import sys
import os

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import time
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from src.config import *
import src.config as cfg 
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps, visualize_load_patterns
from src.optimization import run_ga_optimization
from src.post_processing import save_results_to_csv, plot_results

# ==================================================================================
# [CORE CONFIGURATION] 제안된 방법론 (Scenario A) 설정
# ==================================================================================

EXAMPLES = {
    'Example_1_4Story': {
        'Floors': 4,
        'Col_Locs': cfg.COLUMN_LOCATIONS_4F,
        'Beam_Conns': cfg.BEAM_CONNECTIONS_4F,
        'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_4F
    },
    'Example_2_6Story': {
        'Floors': 6,
        'Col_Locs': cfg.COLUMN_LOCATIONS_6F,
        'Beam_Conns': cfg.BEAM_CONNECTIONS_6F,
        'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_6F
    },
    'Example_3_8Story': {
        'Floors': 8,
        'Col_Locs': cfg.COLUMN_LOCATIONS_8F,
        'Beam_Conns': cfg.BEAM_CONNECTIONS_8F,
        'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_8F
    }
}

# 실험 파라미터 (Scenario A 고정)
OPT_POP_SIZE = 200 
OPT_GENERATIONS = 100 
OPT_CX_METHOD = cfg.CROSSOVER_STRATEGY 
OPT_CX_PROB = cfg.CXPB 
OPT_MUT_PROB = cfg.MUTPB 
OPT_TOURN_SIZE = 3 

# 시나리오 A를 위해 GROUPING_STRATEGY 강제 설정
cfg.GROUPING_STRATEGY = "Individual"

OUTPUT_BASE_DIR = "Results_Main_Examples_Comparison"

# ==================================================================================

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

def run_example_optimization(ex_name, ex_config):
    output_folder = os.path.join(OUTPUT_BASE_DIR, ex_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"Running Optimization for {ex_name} ({ex_config['Floors']} Floors)...")
    print(f"Strategy: {cfg.GROUPING_STRATEGY} (Automatic Grouping)")
    print("="*60)
    
    # 1. 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    # 2. 구조 정보 설정 (전역 변수 임시 덮어쓰기)
    original_floors = cfg.FLOORS
    original_col_locs = cfg.COLUMN_LOCATIONS
    original_beam_conns = cfg.BEAM_CONNECTIONS
    original_trib_widths = cfg.BEAM_TRIBUTARY_WIDTHS

    cfg.FLOORS = ex_config['Floors']
    cfg.COLUMN_LOCATIONS = ex_config['Col_Locs']
    cfg.BEAM_CONNECTIONS = ex_config['Beam_Conns']
    cfg.BEAM_TRIBUTARY_WIDTHS = ex_config['Trib_Widths']
    
    # 2.1 하중 패턴 선택
    if 'Example_1' in ex_name: dynamic_patterns = cfg.LOAD_PATTERNS_4F
    elif 'Example_2' in ex_name: dynamic_patterns = cfg.LOAD_PATTERNS_6F
    elif 'Example_3' in ex_name: dynamic_patterns = cfg.LOAD_PATTERNS_8F
    else: dynamic_patterns = cfg.PATTERNS_BY_FLOOR
    
    visualize_load_patterns(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS, dynamic_patterns, output_folder)
    
    try:
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        # 3. 그룹핑 맵 재생성 (Individual)
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
        
        # 4. 정규화 스케일 재계산
        total_col_len = num_columns * cfg.H
        total_beam_len = sum(beam_lengths) * cfg.FLOORS
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
            column_sections_df, beam_sections_df, total_col_len, total_beam_len
        )
        
        start_time = time.time()
        pop, logbook, final_hof, hof_stats = run_ga_optimization(
            DL=cfg.DL_AREA_LOAD, LL=cfg.LL_AREA_LOAD,
            crossover_method=OPT_CX_METHOD, 
            patterns_by_floor=dynamic_patterns,
            h5_file=h5_file,
            num_generations=OPT_GENERATIONS, population_size=OPT_POP_SIZE,
            col_map=col_map, beam_map=beam_map, 
            beam_sections=beam_sections, column_sections=column_sections,
            beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, 
            beam_lengths=beam_lengths, chromosome_structure=chromosome_structure, 
            num_columns=num_columns, num_beams=num_beams,
            fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
            fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
            tournament_size=OPT_TOURN_SIZE, cxpb=OPT_CX_PROB, mutpb=OPT_MUT_PROB,
            verbose=True
        )
        elapsed = time.time() - start_time
        print(f"   -> Optimization for {ex_name} completed in {elapsed:.1f}s")

        # 결과 저장 및 시각화
        processed_valid_solutions = []
        for i, ind in enumerate(final_hof):
            if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
                solution_data = ind.detailed_results.copy()
                solution_data['ID'] = i + 1
                solution_data['ind_object'] = ind 
                processed_valid_solutions.append(solution_data)

        if not processed_valid_solutions:
            print(f"No feasible solutions found for {ex_name}.")

        save_results_to_csv(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure)
        plot_results(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure,
                     col_map, beam_map, beam_sections, column_sections)
        
        return {'stats': hof_stats, 'hof': final_hof, 'time': elapsed, 'name': ex_name}
        
    finally:
        h5_file.close()
        cfg.FLOORS = original_floors
        cfg.COLUMN_LOCATIONS = original_col_locs
        cfg.BEAM_CONNECTIONS = original_beam_conns
        cfg.BEAM_TRIBUTARY_WIDTHS = original_trib_widths


def main():
    print("### Main Execution: 3 Building Examples (Proposed Scenario A) ###")
    
    all_results = []
    for name, config in EXAMPLES.items():
        result = run_example_optimization(name, config)
        if result:
            all_results.append(result)
        
    if not all_results:
        print("\nNo examples were run successfully.")
        return

    # 1. Hypervolume Comparison
    plt.figure(figsize=(10, 6))
    for data in all_results:
        hv = [entry['hypervolume'] for entry in data['stats']]
        plt.plot(hv, label=f"{data['name']} ({data['time']:.0f}s)")
    plt.title('Hypervolume Convergence (Proposed Method)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, 'Examples_HV_Comparison.png'))
    
    # 2. Pareto Front Comparison
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5), squeeze=False)
    for i, data in enumerate(all_results):
        hof_solutions = [sol for sol in data['hof'] if hasattr(sol, 'detailed_results') and sol.detailed_results.get('violation') == 0.0]
        fit1 = [sol.fitness.values[0] for sol in hof_solutions] 
        fit2 = [sol.fitness.values[1] for sol in hof_solutions] # Max Drift Ratio
        ax = axes[0, i]
        ax.scatter(fit2, fit1, c='blue', alpha=0.7)
        ax.set_title(data['name'])
        ax.set_xlabel('Max Drift Ratio'); ax.set_ylabel('Norm Cost+CO2'); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, 'Examples_Pareto_Comparison.png'))
    
    # 3. Summary CSV
    summary_list = []
    for data in all_results:
        best_hv = data['stats'][-1]['hypervolume']
        feasible_hof_individuals = [ind for ind in data['hof'] if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0]
        best_ind = min(feasible_hof_individuals, key=lambda x: x.fitness.values[0]) if feasible_hof_individuals else None 
        if best_ind:
            summary_list.append({
                'Example': data['name'],
                'Floors': EXAMPLES[data['name']]['Floors'],
                'Time(s)': round(data['time'], 1),
                'Final_HV': round(best_hv, 4),
                'Best_Cost': round(best_ind.detailed_results['cost'], 0),
                'Best_CO2': round(best_ind.detailed_results['co2'], 0),
                'Max_Drift_Ratio': round(best_ind.detailed_results['max_drift_ratio'], 4),
                'N_types': best_ind.detailed_results['N_types']
            })
    pd.DataFrame(summary_list).to_csv(os.path.join(OUTPUT_BASE_DIR, 'examples_summary.csv'), index=False)
    print(f"\n[Artifacts Saved in {OUTPUT_BASE_DIR}]")

if __name__ == "__main__":
    main()