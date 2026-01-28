import random
import numpy as np
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from deap.benchmarks.tools import hypervolume as hv_indicator
from src.config import *
import src.config as cfg
from src.structural_analysis import evaluate

def run_ga_optimization(DL, LL, crossover_method, patterns_by_floor, h5_file,
                        num_generations, population_size,
                        col_map, beam_map, beam_sections, column_sections, 
                        beam_sections_df, column_sections_df, beam_lengths, 
                        chromosome_structure, num_columns, num_beams,
                        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2,
                        tournament_size=7, cxpb=0.9, mutpb=0.1,
                        initial_pop=None, start_gen=0, logbook=None, hof=None, hof_stats_history=None,
                        verbose=True):
    """
    DEAP 라이브러리를 사용하여 NSGA-II 다중목표 유전 알고리즘을 설정하고 실행하는 함수.
    """
    # --- 1. 제약조건 우선 선택 함수 정의 ---
    def constrained_dominance_selection(individuals, k):
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
            obj2 = res['max_drift_ratio']

            ind.fitness.values = (obj1, obj2 if obj2 > 0 else float('inf'))

    # --- 3. DEAP Toolbox 설정 ---
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    num_col_opts, num_beam_opts = len(column_sections), len(beam_sections)
    
    def init_individual():
        # Smart Heuristic Initialization:
        # To find feasible solutions for 6-story irregular frames, we need:
        # 1. Strong columns (Strength constraint)
        # 2. Stiff beams (Deflection constraint)
        # 3. Hierarchy (Col size constraint) -> Hard to enforce perfectly here without map, 
        #    but picking from a narrow range of "Strong" sections reduces the chance of huge violations.
        
        prob = random.random()
        
        if prob < 0.4: # 40% chance: "Heavy Duty" Initialization
            # Pick from top 20% of strongest sections
            min_col = int(num_col_opts * 0.8)
            min_beam = int(num_beam_opts * 0.8)
            col_genes = [random.randint(min_col, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
            beam_genes = [random.randint(min_beam, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec'])]
            
        elif prob < 0.7: # 30% chance: "Medium-Heavy" Initialization
            # Pick from top 50%
            min_col = int(num_col_opts * 0.5)
            min_beam = int(num_beam_opts * 0.5)
            col_genes = [random.randint(min_col, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
            beam_genes = [random.randint(min_beam, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec'])]
            
        else: # 30% chance: Pure Random (Exploration)
            col_genes = [random.randint(0, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
            beam_genes = [random.randint(0, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec'])]
            
        rot_genes = [random.randint(0, 1) for _ in range(chromosome_structure['col_rot'])]
        
        return creator.Individual(col_genes + rot_genes + beam_genes)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate, DL=DL, LL=LL, h5_file=h5_file, patterns_by_floor=patterns_by_floor,
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
        if chromosome_structure['col_rot'] > 0:
            # Scenario A (Proposed): Col + Rot + Beam
            for i in range(len(individual)):
                if random.random() < indpb:
                    if i < chromosome_structure['col_sec']:
                        individual[i] = random.randint(0, num_col_opts - 1)
                    elif i < chromosome_structure['col_sec'] + chromosome_structure['col_rot']:
                        individual[i] = random.randint(0, 1)
                    else:
                        individual[i] = random.randint(0, num_beam_opts - 1)
        else:
            # Scenario B (Conventional): Col + Beam (No rotation variables)
            for i in range(len(individual)):
                if random.random() < indpb:
                    if i < chromosome_structure['col_sec']:
                        individual[i] = random.randint(0, num_col_opts - 1)
                    else:
                        individual[i] = random.randint(0, num_beam_opts - 1)
        return individual,
    
    toolbox.register("mutate", custom_mutate, indpb=0.1)
    toolbox.register("select_offspring", tools.selTournament, tournsize=tournament_size)
    
    # 통계 및 로그북 설정
    def get_valid_ratio(population):
        valid_count = sum(1 for ind in population if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation', float('inf')) == 0.0)
        return valid_count / len(population) * 100 if population else 0.0
    def get_analysis_success_ratio(population):
        if not population: return 0.0
        success_count = sum(1 for ind in population if hasattr(ind, 'detailed_results') and ind.detailed_results.get('cost') != float('inf'))
        return success_count / len(population) * 100
    
    def get_best_invalid_margins(population):
        if not population: return "N/A"
        # Find best individual based on violation score (closest to feasible)
        # If feasible exists, it will show 0.00 for all.
        # If not, show margins of the "least violated" individual.
        best_ind = min(population, key=lambda ind: ind.detailed_results.get('violation', float('inf')))
        margins = best_ind.detailed_results.get('absolute_margins', {})
        
        if not margins: return "Margins N/A"
        
        # S:Strength, D:Drift, W:Wind, F:Defl, H:SCWB, C:ColSize
        margin_str = (f"S:{margins.get('strength', 0):.2f} "
                      f"D:{margins.get('drift', 0):.2f} "
                      f"W:{margins.get('wind_disp', 0):.2f} "
                      f"F:{margins.get('deflection', 0):.2f} "
                      f"H:{margins.get('hierarchy', 0):.2f} "
                      f"C:{margins.get('col_size', 0):.2f}")
        return margin_str

    def calculate_valid_stat(pop, key, stat_func, default_val=0.0):
        valid_values = [ind.detailed_results[key] for ind in pop if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0 and key in ind.detailed_results]
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
    value_stats.register("avg_drift", lambda pop: calculate_valid_stat(pop, 'max_drift_ratio', np.mean))
    value_stats.register("min_drift", lambda pop: calculate_valid_stat(pop, 'max_drift_ratio', np.min))

    # Hypervolume Statistic
    HV_REFERENCE_POINT = [2.5, 2.5]
    def get_hypervolume(population):
        feasible_inds = [ind for ind in population if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation', float('inf')) == 0.0]
        if not feasible_inds: return 0.0
        try:
            return hv_indicator(feasible_inds, HV_REFERENCE_POINT)
        except:
            return 0.0

    margin_stats = tools.Statistics()
    margin_stats.register("best_margins", get_best_invalid_margins)
    margin_stats.register("hypervolume", get_hypervolume)
    
    hof_value_stats = tools.Statistics()
    hof_value_stats.register("hof_min_cost", lambda h: calculate_valid_stat(h, 'cost', np.min))
    hof_value_stats.register("hof_min_drift", lambda h: calculate_valid_stat(h, 'max_drift_ratio', np.min))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + fitness_stats.fields + ['sep1'] + health_stats.fields + ['sep2'] + value_stats.fields + ['sep3'] + margin_stats.fields + ['sep4', 'hof_size'] + ['sep5'] + hof_value_stats.fields

    # --- 초기 모집단 생성 또는 로드 ---
    if initial_pop is None:
        pop = toolbox.population(n=population_size)
        hof = tools.ParetoFront()
        hof_stats_history = []
        
        if verbose: print("\n초기 집단 평가 중...")
        eval_results = []
        for ind in tqdm(pop, desc="Initial Population Evaluation", unit="individual", disable=not verbose):
            eval_results.append(toolbox.evaluate(ind))
        for ind, res in zip(pop, eval_results):
            ind.detailed_results = res
        _assign_fitness(pop)
        
        feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
        hof.update(feasible_pop)
        
        hv_val = 0.0
        if hof:
            try:
                hv_val = hv_indicator(hof, HV_REFERENCE_POINT)
            except:
                hv_val = 0.0
        
        best_obj1 = min([ind.fitness.values[0] for ind in hof]) if hof else float('inf')
        best_obj2 = min([ind.fitness.values[1] for ind in hof]) if hof else float('inf')
        hof_stats_history.append({'gen': 0, 'best_obj1': best_obj1, 'best_obj2': best_obj2, 'hypervolume': hv_val})

        record = fitness_stats.compile(pop)
        record.update(health_stats.compile(pop))
        record.update(value_stats.compile(pop))
        record.update(margin_stats.compile(pop))
        record.update(hof_value_stats.compile(hof))
        record['hof_size'] = len(hof)
        record['sep1'], record['sep2'], record['sep3'], record['sep4'], record['sep5'] = "|", "|", "|", "|", "|"
        logbook.record(gen=0, nevals=len(pop), **record)
        
        if verbose:
            print("최적화 시작...")
            print(logbook.stream)
    else:
        pop = initial_pop
        if verbose: print(f"\n이전 {start_gen} 세대에서 최적화를 계속합니다...")

    # --- 메인 루프 ---
    for gen in tqdm(range(start_gen + 1, start_gen + num_generations + 1), desc="세대 진화", disable=not verbose):
        if pop is None:
            raise ValueError(f"Error: Population became None at Gen {gen}")
            
        offspring = toolbox.select_offspring(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        eval_results = []
        if invalid_ind:
            for ind in tqdm(invalid_ind, desc=f"Gen {gen} Evaluation", unit="ind", leave=False, disable=not verbose):
                eval_results.append(toolbox.evaluate(ind))
        for ind, res in zip(invalid_ind, eval_results):
            ind.detailed_results = res
            
        _assign_fitness(offspring)
        
        pop = toolbox.select(pop + offspring, k=population_size)
        
        feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
        hof.update(feasible_pop)
        
        hv_val = 0.0
        if hof:
            try:
                hv_val = hv_indicator(hof, HV_REFERENCE_POINT)
            except:
                hv_val = 0.0
        
        best_obj1 = min([ind.fitness.values[0] for ind in hof]) if hof else float('inf')
        best_obj2 = min([ind.fitness.values[1] for ind in hof]) if hof else float('inf')
        hof_stats_history.append({'gen': gen, 'best_obj1': best_obj1, 'best_obj2': best_obj2, 'hypervolume': hv_val})
        
        record = fitness_stats.compile(pop)
        record.update(health_stats.compile(pop))
        record.update(value_stats.compile(pop))
        record.update(margin_stats.compile(pop))
        record.update(hof_value_stats.compile(hof))
        record['hof_size'] = len(hof)
        record['sep1'], record['sep2'], record['sep3'], record['sep4'], record['sep5'] = "|", "|", "|", "|", "|"
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose: tqdm.write(logbook.stream.splitlines()[-1])

    return pop, logbook, hof, hof_stats_history