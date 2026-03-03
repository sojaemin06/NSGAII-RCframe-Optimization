import sys
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

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
        'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_4F,
        'Pop': 500, 'Gen': 100
    },
    # 'Example_2_6Story': {
    #     'Floors': 6,
    #     'Col_Locs': cfg.COLUMN_LOCATIONS_6F,
    #     'Beam_Conns': cfg.BEAM_CONNECTIONS_6F,
    #     'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_6F,
    #     'Pop': 500, 'Gen': 100
    # },
    # 'Example_3_8Story': {
    #     'Floors': 8,
    #     'Col_Locs': cfg.COLUMN_LOCATIONS_8F,
    #     'Beam_Conns': cfg.BEAM_CONNECTIONS_8F,
    #     'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_8F,
    #     'Pop': 500, 'Gen': 100
    # }
}

# 실험 파라미터 (공통 전략)
OPT_CX_METHOD = cfg.CROSSOVER_STRATEGY 
OPT_CX_PROB = cfg.CXPB 
OPT_MUT_PROB = cfg.MUTPB 
OPT_TOURN_SIZE = cfg.TOURNAMENT_SIZE 

# 시나리오 A를 위해 GROUPING_STRATEGY 강제 설정
cfg.GROUPING_STRATEGY = "Hybrid"

# [수정] 출력 폴더명 변경
OUTPUT_BASE_DIR = "Results_Optimization_Paper_Final"

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
    print(f"Strategy: {cfg.GROUPING_STRATEGY} (Hybrid Grouping)")
    print(f"Params: Pop={ex_config['Pop']}, Gen={ex_config['Gen']}")
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
    
    # [수정] Figures 폴더에 시각화 저장
    figures_dir = os.path.join(output_folder, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    visualize_load_patterns(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS, dynamic_patterns, figures_dir)
    
    try:
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        # 3. 그룹핑 맵 재생성 (Hybrid)
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
            num_generations=ex_config['Gen'], population_size=ex_config['Pop'], # 인자 값 사용
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
            if hasattr(ind, 'detailed_results'):
                res = ind.detailed_results
                # 1. 제약 조건 위반이 없고(0.0)
                # 2. 피트니스 값이 유한한(not inf/nan) 경우만 최종 결과로 수집
                is_feasible = (res.get('violation', float('inf')) == 0.0)
                is_finite = not (np.any(np.isinf(ind.fitness.values)) or np.any(np.isnan(ind.fitness.values)))
                
                if is_feasible and is_finite:
                    solution_data = res.copy()
                    solution_data['ID'] = i + 1
                    solution_data['ind_object'] = ind 
                    processed_valid_solutions.append(solution_data)


        if not processed_valid_solutions:
            print(f"No feasible solutions found for {ex_name}.")

        save_results_to_csv(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure)
        plot_results(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure,
                     col_map, beam_map, beam_sections, column_sections, ex_name=ex_name)
        
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

    # [수정] Summary 폴더 생성 및 저장
    summary_dir = os.path.join(OUTPUT_BASE_DIR, "Summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Hypervolume Comparison
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    for i, data in enumerate(all_results):
        hv = [entry['hypervolume'] for entry in data['stats']]
        plt.plot(hv, label=f"{data['name']}", marker=markers[i%3], linestyle=linestyles[i%3], markersize=4)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume Indicator')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'Examples_HV_Comparison.png'))
    
    # 2. Pareto Front Comparison
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 5), squeeze=False)
    for i, data in enumerate(all_results):
        # inf 값 배제 필터링 강화
        hof_solutions = [
            sol for sol in data['hof'] 
            if hasattr(sol, 'detailed_results') 
            and sol.detailed_results.get('violation') == 0.0 
            and not np.isinf(sol.detailed_results.get('max_drift_ratio', np.inf))
        ]
        fit1 = [sol.fitness.values[0] for sol in hof_solutions] # Norm Cost+CO2
        fit2 = [sol.fitness.values[1] for sol in hof_solutions] # Raw Drift
        
        ax = axes[0, i]
        ax.scatter(fit2, fit1, c='blue', marker='o', s=40, alpha=0.7, edgecolors='k') # Blue circles
        # [수정] 제목 삭제
        # ax.set_title(data['name'], fontsize=12)
        ax.set_xlabel('Objective 2 (Max. Inter-story Drift Ratio)')
        ax.set_ylabel(r'Objective 1 (Normalized Cost + CO$_2$)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'Examples_Pareto_Comparison.png'))
    
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
                'N_types': best_ind.detailed_results.get('N_types', 0)
            })
    pd.DataFrame(summary_list).to_csv(os.path.join(summary_dir, 'examples_summary.csv'), index=False)
    
    # 4. Save dedicated Timing Report
    with open(os.path.join(OUTPUT_BASE_DIR, 'optimization_timing_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Optimization Timing Report ===\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 35 + "\n")
        for data in summary_list:
            f.write(f"{data['Example']}: {data['Time(s)']} seconds\n")
        f.write("-" * 35 + "\n")
        total_time = sum(data['Time(s)'] for data in summary_list)
        f.write(f"Total Combined Time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)\n")

    print(f"\n[Artifacts Saved in {OUTPUT_BASE_DIR}]")

if __name__ == "__main__":
    main()
