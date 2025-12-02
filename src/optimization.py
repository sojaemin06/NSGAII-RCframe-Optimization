
import random
from deap import base, creator, tools, algorithms
import numpy as np
from tqdm import tqdm
from .evaluation import evaluate

def run_ga_optimization(chromosome_structure, DL, LL, Wx, Wy, Ex, Ey, crossover_method, patterns_by_floor, h5_file,
                        num_generations, population_size, CXPB, MUTPB,
                        beam_lengths, col_map, beam_map, num_columns, floors, H, column_locations, beam_connections,
                        column_sections_df, beam_sections_df, FIXED_MIN_COST, FIXED_RANGE_COST, FIXED_MIN_CO2, FIXED_RANGE_CO2,
                        load_combinations, column_sections, beam_sections,
                        initial_pop=None, start_gen=0, logbook=None, hof=None, hof_stats_history=None):
    """
    DEAP 라이브러리를 사용하여 NSGA-II 다중목표 유전 알고리즘을 설정하고 실행하는 함수.
    (제약조건 우선 원칙 적용 최종 버전)
    """
    # --- 1. 제약조건 우선 선택 함수 정의 ---
    def constrained_dominance_selection(individuals, k):
        """제약조건 우선 원칙(Constraint-Dominance Principle)을 적용하는 선택 함수."""
        feasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] == 0.0]
        infeasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] > 0.0]
        
        # 유효해 그룹 내에서 NSGA-II 선택을 먼저 수행
        selected_from_feasible = tools.selNSGA2(feasible_inds, len(feasible_inds))
        
        # 다음 세대를 구성할 리스트
        next_generation = selected_from_feasible
        
        # 유효해만으로 k개를 채우지 못했다면, 위반해로 나머지를 채움
        if len(next_generation) < k:
            num_needed = k - len(next_generation)
            # 위반량이 적은 순으로 위반해들을 정렬
            infeasible_inds.sort(key=lambda ind: ind.detailed_results['violation'])
            # 가장 덜 위반한 해들로 나머지 자리를 채움
            next_generation.extend(infeasible_inds[:num_needed])
            
        return next_generation[:k] # 최종적으로 k개만 반환

    # --- 2. Fitness 계산 헬퍼 함수 정의 ---
    def _assign_fitness(population):
        """주어진 인구집단에 대해 페널티 없는 Fitness를 계산하고 할당"""
        for ind in population:
            res = ind.detailed_results
            if res['cost'] == float('inf'):
                ind.fitness.values = (float('inf'), float('inf'))
                continue
            
            # 고정 스케일로 정규화
            norm_cost = max(0.0, min(1.0, (res['cost'] - FIXED_MIN_COST) / FIXED_RANGE_COST))
            norm_co2 = max(0.0, min(1.0, (res['co2'] - FIXED_MIN_CO2) / FIXED_RANGE_CO2))
            
            # 페널티 없이 목표함수 값 할당
            obj1 = norm_cost + norm_co2
            obj2 = res['mean_strength_ratio']

            ind.fitness.values = (obj1, obj2 if obj2 > 0 else float('inf'))

    # --- 3. DEAP Toolbox 설정 ---
    # 2목표: (정규화된 Cost+CO2) 최소화, (평균 응력비) 최대화
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    num_col_opts, num_beam_opts = len(column_sections), len(beam_sections)
    gene_pool = [lambda: random.randint(0, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
    gene_pool.extend(lambda: random.randint(0, 1) for _ in range(chromosome_structure['col_rot']))
    gene_pool.extend(lambda: random.randint(0, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec']))

    toolbox.register("individual", tools.initCycle, creator.Individual, tuple(gene_pool))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, chromosome_structure=chromosome_structure, DL=DL, LL=LL, Wx=Wx, Wy=Wy, Ex=Ex, Ey=Ey, h5_file=h5_file, patterns_by_floor=patterns_by_floor, beam_lengths=beam_lengths, col_map=col_map, beam_map=beam_map, num_columns=num_columns, floors=floors, H=H, column_locations=column_locations, beam_connections=beam_connections, column_sections_df=column_sections_df, beam_sections_df=beam_sections_df, FIXED_MIN_COST=FIXED_MIN_COST, FIXED_RANGE_COST=FIXED_RANGE_COST, FIXED_MIN_CO2=FIXED_MIN_CO2, FIXED_RANGE_CO2=FIXED_RANGE_CO2, load_combinations=load_combinations, column_sections=column_sections, beam_sections=beam_sections)
    toolbox.register("select", constrained_dominance_selection)
    
    if crossover_method == 'Uniform':
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    elif crossover_method == 'OnePoint':
        toolbox.register("mate", tools.cxOnePoint)
    else:
        toolbox.register("mate", tools.cxTwoPoint)

    def custom_mutate(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                if i < chromosome_structure['col_sec']:
                    individual[i] = random.randint(0, num_col_opts - 1)
                elif i < chromosome_structure['col_sec'] + chromosome_structure['col_rot']:
                    individual[i] = random.randint(0, 1)
                else:
                    individual[i] = random.randint(0, num_beam_opts - 1)
        return individual,
    toolbox.register("mutate", custom_mutate, indpb=0.1)
    toolbox.register("select_offspring", tools.selTournament, tournsize=7)
    
    # 통계 및 로그북 설정
    def get_valid_ratio(population):
        """제약조건을 모두 만족하는 해의 비율을 계산하는 함수"""
        valid_count = sum(1 for ind in population if (
            hasattr(ind, 'detailed_results') and
            # 'violation' 값이 0이면 모든 제약조건을 만족한 유효해
            ind.detailed_results.get('violation', float('inf')) == 0.0
        ))
        return valid_count / len(population) * 100 if population else 0.0
    def get_analysis_success_ratio(population):
        """해석에 성공한 해의 비율을 계산하는 함수"""
        if not population:
            return 0.0
        success_count = sum(1 for ind in population if 
                            hasattr(ind, 'detailed_results') and 
                            ind.detailed_results.get('cost') != float('inf'))
        return success_count / len(population) * 100
    def get_best_invalid_margins(population):
        """
        위반량이 가장 적은 개체를 찾아 해당 개체의 상세 위반 내역을 문자열로 반환.
        (S: Strength, D: Drift, W: Wind, F: deFlection, H: Hierarchy)
        """
        if not population:
            return "N/A"

        # detailed_results가 있고 'violation' 키가 있는 개체들만 필터링
        valid_for_check = [ind for ind in population if hasattr(ind, 'detailed_results') and 'violation' in ind.detailed_results]

        if not valid_for_check:
            return "No detailed results"

        # 위반량이 가장 적은 개체를 찾음
        best_ind = min(valid_for_check, key=lambda ind: ind.detailed_results['violation'])

        margins = best_ind.detailed_results.get('absolute_margins')
        if not margins:
            return "Margins N/A"

        # 각 위반 항목을 축약된 문자열로 포매팅
        margin_str = (
            f"S:{margins.get('strength', 0):.2f} "
            f"D:{margins.get('drift', 0):.2f} "
            f"W:{margins.get('wind_disp', 0):.2f} "
            f"F:{margins.get('deflection', 0):.2f} "
            f"H:{margins.get('hierarchy', 0):.2f}"
        )
        return margin_str
    # --- 4. 최적화 루프 실행 ---
    # 통계 헬퍼 함수 정의 (유효해만 필터링하여 계산)
    def calculate_valid_stat(pop, key, stat_func, default_val=0.0):
        """
        유효해(violation == 0)만 필터링하여 특정 값(key)에 대한 통계(stat_func)를 계산합니다.
        유효해가 없을 경우 default_val을 반환합니다.
        """
        valid_values = [
            ind.detailed_results[key] for ind in pop
            if hasattr(ind, 'detailed_results') and
            ind.detailed_results.get('violation') == 0.0 and
            key in ind.detailed_results
        ]
        return stat_func(valid_values) if valid_values else default_val
    def calculate_margin_stat(pop, margin_dict_key, margin_key, stat_func, default_val=float('inf')):
        """특정 마진(dict_key/key)에 대한 통계(stat_func)를 계산하는 헬퍼 함수"""
        margin_values = []
        for ind in pop:
            if hasattr(ind, 'detailed_results') and isinstance(ind.detailed_results, dict):
                margins_dict = ind.detailed_results.get(margin_dict_key, {})
                val = margins_dict.get(margin_key)
                if val is not None: margin_values.append(val)
        
        valid_values = [v for v in margin_values if v != float('inf')]
        return stat_func(valid_values) if valid_values else default_val
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    # --- 통계 객체 재구성 (전체 항목) ---
    # 그룹 1: Fitness 통계 (전체)
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    fitness_stats.register("avg", np.mean, axis=0)
    fitness_stats.register("max", np.max, axis=0)
    fitness_stats.register("min", np.min, axis=0)
    fitness_stats.register("std", np.std, axis=0)

    # 그룹 2: 모집단 상태 통계
    health_stats = tools.Statistics()
    health_stats.register("success_rate", get_analysis_success_ratio)
    health_stats.register("valid_ratio", get_valid_ratio)

    # 그룹 3: 유효해 값 통계 (전체)
    value_stats = tools.Statistics()
    value_stats.register("avg_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.mean))
    value_stats.register("max_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.max))
    value_stats.register("min_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.min))
    value_stats.register("std_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.std))
    value_stats.register("avg_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.mean))
    value_stats.register("max_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.max))
    value_stats.register("min_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.min))
    value_stats.register("std_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.std))
    value_stats.register("avg_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.mean))
    value_stats.register("max_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.max))
    value_stats.register("min_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.min))
    value_stats.register("std_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.std))

    # 그룹 4: 위반량 상세 통계 (전체)
    margin_stats = tools.Statistics()
    margin_stats.register("best_margins", get_best_invalid_margins)
    margin_keys = ['strength', 'drift', 'wind_disp', 'deflection', 'hierarchy']
    margin_abbrs = ['str', 'drift', 'wind', 'defl', 'hier']
    for key, abbr in zip(margin_keys, margin_abbrs):
        margin_stats.register(f"min_AM_{abbr}", lambda pop, k=key: calculate_margin_stat(pop, 'absolute_margins', k, np.min, default_val=0.0))
    margin_stats.register("min_NM_str", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'strength', np.min))
    margin_stats.register("min_NM_drift", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'drift', np.min))
    margin_stats.register("min_NM_wind", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'wind_disp', np.min))
    margin_stats.register("min_NM_defl", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'deflection', np.min))
    margin_stats.register("min_NM_hier", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'hierarchy', np.min))

    # 그룹 5: Hall of Fame 통계 (유효해 대상)
    def calculate_hypervolume(hof):
        """Hall of Fame의 Hypervolume을 계산 (Reference Point: 1.1, 1.1)"""
        if not hof: return 0.0
        # 유효한 해들의 목적함수 값 추출 (f1, f2)
        front = [ind.fitness.values for ind in hof]
        # DEAP의 hypervolume 함수 사용 (최소화 문제 기준)
        return tools.hypervolume(front, [1.1, 1.1])

    hof_value_stats = tools.Statistics()
    hof_value_stats.register("hof_avg_cost", lambda h: calculate_valid_stat(h, 'cost', np.mean))
    hof_value_stats.register("hof_max_cost", lambda h: calculate_valid_stat(h, 'cost', np.max))
    hof_value_stats.register("hof_min_cost", lambda h: calculate_valid_stat(h, 'cost', np.min))
    hof_value_stats.register("hof_std_cost", lambda h: calculate_valid_stat(h, 'cost', np.std))
    hof_value_stats.register("hof_avg_co2", lambda h: calculate_valid_stat(h, 'co2', np.mean))
    hof_value_stats.register("hof_max_co2", lambda h: calculate_valid_stat(h, 'co2', np.max))
    hof_value_stats.register("hof_min_co2", lambda h: calculate_valid_stat(h, 'co2', np.min))
    hof_value_stats.register("hof_std_co2", lambda h: calculate_valid_stat(h, 'co2', np.std))
    hof_value_stats.register("hof_avg_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.mean))
    hof_value_stats.register("hof_max_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.max))
    hof_value_stats.register("hof_min_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.min))
    hof_value_stats.register("hof_std_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.std))
    hof_value_stats.register("hypervolume", calculate_hypervolume) # HV 추가

    # --- 로그북 헤더 재구성 ---
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + fitness_stats.fields + \
                     ['sep1'] + health_stats.fields + \
                     ['sep2'] + value_stats.fields + \
                     ['sep3'] + margin_stats.fields + \
                     ['sep4', 'hof_size'] + \
                     ['sep5'] + hof_value_stats.fields

    if initial_pop is None:
        # --- 최초 실행 시 초기화 ---
        pop = toolbox.population(n=population_size)
        hof = tools.ParetoFront()
        hof_stats_history = []
        
        print("\n초기 집단 평가 중...")
        eval_results = []
        for ind in tqdm(pop, desc="Initial Population Evaluation", unit="individual"):
            eval_results.append(toolbox.evaluate(ind))
        for ind, res in zip(pop, eval_results):
            ind.detailed_results = res
        _assign_fitness(pop)
        
        feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
        hof.update(feasible_pop)
        if hof:
            best_obj1 = min(ind.fitness.values[0] for ind in hof)
            best_obj2 = min(ind.fitness.values[1] for ind in hof)
            hv = calculate_hypervolume(hof)
            hof_stats_history.append({'gen': 0, 'best_obj1': best_obj1, 'best_obj2': best_obj2, 'hypervolume': hv})

        record = fitness_stats.compile(pop)
        record.update(health_stats.compile(pop))
        record.update(value_stats.compile(pop))
        record.update(margin_stats.compile(pop))
        record.update(hof_value_stats.compile(hof))
        record['hof_size'] = len(hof)
        record['sep1'], record['sep2'], record['sep3'], record['sep4'], record['sep5'] = "|", "|", "|", "|", "|"
        logbook.record(gen=0, nevals=len(pop), **record)
        
        print("최적화 시작...")
        print(logbook.stream)
    else:
        # --- 이어하기 시 상태 복원 ---
        pop = initial_pop
        print(f"\n이전 {start_gen} 세대에서 최적화를 계속합니다...")

    # 메인 진화 루프
    for gen in tqdm(range(start_gen + 1, start_gen + num_generations + 1), desc="세대 진화"):
        offspring = toolbox.select_offspring(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        eval_results = []
        if invalid_ind:
            for ind in tqdm(invalid_ind, desc=f"Gen {gen} Evaluation", unit="ind", leave=False):
                eval_results.append(toolbox.evaluate(ind))
        for ind, res in zip(invalid_ind, eval_results):
            ind.detailed_results = res
            
        _assign_fitness(offspring)
        
        pop = toolbox.select(pop + offspring, k=population_size)
        
        feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
        hof.update(feasible_pop)
        if hof:
            best_obj1 = min(ind.fitness.values[0] for ind in hof)
            best_obj2 = min(ind.fitness.values[1] for ind in hof)
            hv = calculate_hypervolume(hof)
            hof_stats_history.append({'gen': gen, 'best_obj1': best_obj1, 'best_obj2': best_obj2, 'hypervolume': hv})
        record = fitness_stats.compile(pop)
        record.update(health_stats.compile(pop))
        record.update(value_stats.compile(pop))
        record.update(margin_stats.compile(pop))
        record.update(hof_value_stats.compile(hof))
        record['hof_size'] = len(hof)
        record['sep1'], record['sep2'], record['sep3'], record['sep4'], record['sep5'] = "|", "|", "|", "|", "|"
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        tqdm.write(logbook.stream.splitlines()[-1])

    
    return pop, logbook, hof, hof_stats_history
