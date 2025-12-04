
import random
import numpy as np
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from src.config import *
from src.structural_analysis import evaluate

def run_ga_optimization(DL, LL, Wx, Wy, Ex, Ey, crossover_method, patterns_by_floor, h5_file,
                        num_generations, population_size,
                        col_map, beam_map, beam_sections, column_sections, 
                        beam_sections_df, column_sections_df, beam_lengths, 
                        chromosome_structure, num_columns, num_beams,
                        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2,
                        initial_pop=None, start_gen=0, logbook=None, hof=None, hof_stats_history=None):
    """
    DEAP 라이브러리를 사용하여 NSGA-II 다중목표 유전 알고리즘을 설정하고 실행하는 함수.
    """
    # --- 1. 제약조건 우선 선택 함수 정의 ---
    def constrained_dominance_selection(individuals, k):
        """제약조건 우선 원칙(Constraint-Dominance Principle)을 적용하는 선택 함수."""
        feasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] == 0.0]
        infeasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] > 0.0]
        
        selected_from_feasible = tools.selNSGA2(feasible_inds, len(feasible_inds))
        next_generation = selected_from_feasible
        
        if len(next_generation) < k:
            num_needed = k - len(next_generation)
            infeasible_inds.sort(key=lambda ind: ind.detailed_results['violation'])
            next_generation.extend(infeasible_inds[:num_needed])
            
        return next_generation[:k]

    # --- 2. Fitness 계산 헬퍼 함수 정의 ---
    def _assign_fitness(population):
        for ind in population:
            res = ind.detailed_results
            if res['cost'] == float('inf'):
                ind.fitness.values = (float('inf'), float('inf'))
                continue
            
            norm_cost = max(0.0, min(1.0, (res['cost'] - fixed_min_cost) / fixed_range_cost))
            norm_co2 = max(0.0, min(1.0, (res['co2'] - fixed_min_co2) / fixed_range_co2))
            
            obj1 = norm_cost + norm_co2
            obj2 = res['mean_strength_ratio']

            ind.fitness.values = (obj1, obj2 if obj2 > 0 else float('inf'))

    # --- 3. DEAP Toolbox 설정 ---
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    num_col_opts, num_beam_opts = len(column_sections), len(beam_sections)
    gene_pool = [lambda: random.randint(0, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
    gene_pool.extend(lambda: random.randint(0, 1) for _ in range(chromosome_structure['col_rot']))
    gene_pool.extend(lambda: random.randint(0, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec']))

    toolbox.register("individual", tools.initCycle, creator.Individual, tuple(gene_pool))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate, DL=DL, LL=LL, Wx=Wx, Wy=Wy, Ex=Ex, Ey=Ey, h5_file=h5_file, patterns_by_floor=patterns_by_floor,
                     col_map=col_map, beam_map=beam_map, beam_sections=beam_sections, column_sections=column_sections,
                     beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, beam_lengths=beam_lengths,
                     chromosome_structure=chromosome_structure, num_columns=num_columns, num_beams=num_beams)
    
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
    
    # 통계 및 로그북 설정 (헬퍼 함수들은 내부 정의 혹은 utils로 이동 가능하지만 여기 둠)
    def get_valid_ratio(population):
        valid_count = sum(1 for ind in population if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation', float('inf')) == 0.0)
        return valid_count / len(population) * 100 if population else 0.0
    def get_analysis_success_ratio(population):
        if not population: return 0.0
        success_count = sum(1 for ind in population if hasattr(ind, 'detailed_results') and ind.detailed_results.get('cost') != float('inf'))
        return success_count / len(population) * 100
    
    def get_best_invalid_margins(population):
        if not population: return "N/A"
        valid_for_check = [ind for ind in population if hasattr(ind, 'detailed_results') and 'violation' in ind.detailed_results]
        if not valid_for_check: return "No detailed results"
        best_ind = min(valid_for_check, key=lambda ind: ind.detailed_results['violation'])
        margins = best_ind.detailed_results.get('absolute_margins')
        if not margins: return "Margins N/A"
        margin_str = (f"S:{margins.get('strength', 0):.2f} D:{margins.get('drift', 0):.2f} W:{margins.get('wind_disp', 0):.2f} F:{margins.get('deflection', 0):.2f} H:{margins.get('hierarchy', 0):.2f}")
        return margin_str

    def calculate_valid_stat(pop, key, stat_func, default_val=0.0):
        valid_values = [ind.detailed_results[key] for ind in pop if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0 and key in ind.detailed_results]
        return stat_func(valid_values) if valid_values else default_val
    
    def calculate_margin_stat(pop, margin_dict_key, margin_key, stat_func, default_val=float('inf')):
        margin_values = []
        for ind in pop:
            if hasattr(ind, 'detailed_results') and isinstance(ind.detailed_results, dict):
                margins_dict = ind.detailed_results.get(margin_dict_key, {})
                val = margins_dict.get(margin_key)
                if val is not None: margin_values.append(val)
        valid_values = [v for v in margin_values if v != float('inf')]
        return stat_func(valid_values) if valid_values else default_val

    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    fitness_stats.register("avg", np.mean, axis=0)
    fitness_stats.register("max", np.max, axis=0)
    fitness_stats.register("min", np.min, axis=0)
    fitness_stats.register("std", np.std, axis=0)

    health_stats = tools.Statistics()
    health_stats.register("success_rate", get_analysis_success_ratio)
    health_stats.register("valid_ratio", get_valid_ratio)

    value_stats = tools.Statistics()
    value_stats.register("avg_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.mean))
    value_stats.register("min_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.min))
    value_stats.register("avg_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.mean))
    value_stats.register("min_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.min))

    margin_stats = tools.Statistics()
    margin_stats.register("best_margins", get_best_invalid_margins)
    
    hof_value_stats = tools.Statistics()
    hof_value_stats.register("hof_min_cost", lambda h: calculate_valid_stat(h, 'cost', np.min))
    hof_value_stats.register("hof_min_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.min))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + fitness_stats.fields + ['sep1'] + health_stats.fields + ['sep2'] + value_stats.fields + ['sep3'] + margin_stats.fields + ['sep4', 'hof_size'] + ['sep5'] + hof_value_stats.fields

    if initial_pop is None:
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
            hof_stats_history.append({'gen': 0, 'best_obj1': best_obj1, 'best_obj2': best_obj2})

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
        pop = initial_pop
        print(f"\n이전 {start_gen} 세대에서 최적화를 계속합니다...")

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
            hof_stats_history.append({'gen': gen, 'best_obj1': best_obj1, 'best_obj2': best_obj2})
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
