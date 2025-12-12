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
import random # For reproducibility
from src.config import *
import src.config as cfg # For accessing new variables
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps, visualize_load_patterns, generate_load_patterns
from src.optimization import run_ga_optimization
from src.post_processing import save_results_to_csv, plot_results

# ==================================================================================
# [USER CONFIGURATION] 3가지 구조물 예시 정의 (4층, 6층, 8층 - Different Plans)
# ==================================================================================

EXAMPLES = {
    'Example_1_4Story': {
        'Floors': 4,
        'Col_Locs': cfg.COLUMN_LOCATIONS_4F,
        'Beam_Conns': cfg.BEAM_CONNECTIONS_4F,
        'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_4F
    # },
    # 'Example_2_6Story': {
    #     'Floors': 6,
    #     'Col_Locs': cfg.COLUMN_LOCATIONS_6F,
    #     'Beam_Conns': cfg.BEAM_CONNECTIONS_6F,
    #     'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_6F
    # },
    # 'Example_3_8Story': {
    #     'Floors': 8,
    #     'Col_Locs': cfg.COLUMN_LOCATIONS_8F,
    #     'Beam_Conns': cfg.BEAM_CONNECTIONS_8F,
    #     'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_8F
    }
}

# 실험 파라미터
OPT_POP_SIZE = 200 
OPT_GENERATIONS = 0 
OPT_CX_METHOD = cfg.CROSSOVER_STRATEGY 
OPT_CX_PROB = cfg.CXPB 
OPT_MUT_PROB = cfg.MUTPB 
OPT_TOURN_SIZE = 3 

OUTPUT_BASE_DIR = "Results_Main_Examples_Comparison"

# ==================================================================================

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

def run_example_optimization(ex_name, ex_config):
    # 난수 시드 고정 제거 (매 실행마다 다른 결과 도출)

    output_folder = os.path.join(OUTPUT_BASE_DIR, ex_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"Running Optimization for {ex_name} ({ex_config['Floors']} Floors)...")
    print("="*60)
    
    # 1. 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    # 2. 구조 정보 설정 (전역 변수 임시 덮어쓰기)
    original_floors = cfg.FLOORS
    original_col_locs = cfg.COLUMN_LOCATIONS
    original_beam_conns = cfg.BEAM_CONNECTIONS
    original_trib_widths = cfg.BEAM_TRIBUTARY_WIDTHS
    # original_patterns... 는 어차피 동적 생성하므로 백업 불필요

    cfg.FLOORS = ex_config['Floors']
    cfg.COLUMN_LOCATIONS = ex_config['Col_Locs']
    cfg.BEAM_CONNECTIONS = ex_config['Beam_Conns']
    cfg.BEAM_TRIBUTARY_WIDTHS = ex_config['Trib_Widths']
    
    # 2.1 하중 패턴 선택 (Hardcoded from config.py)
    if 'Example_1' in ex_name:
        dynamic_patterns = cfg.LOAD_PATTERNS_4F
    elif 'Example_2' in ex_name:
        dynamic_patterns = cfg.LOAD_PATTERNS_6F
    elif 'Example_3' in ex_name:
        dynamic_patterns = cfg.LOAD_PATTERNS_8F
    else:
        dynamic_patterns = cfg.PATTERNS_BY_FLOOR # Default fallback
    
    # 패턴 시각화 저장
    visualize_load_patterns(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS, dynamic_patterns, output_folder)
    
    try:
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        # 3. 그룹핑 맵 재생성
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups if cfg.GROUPING_STRATEGY != "Expanded_DB_No_Rot" else 0, 'beam_sec': num_beam_groups}
        
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
            patterns_by_floor=dynamic_patterns, # 동적 생성된 패턴 사용
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
                solution_data['ID'] = i + 1  # Unique ID for each solution
                solution_data['ind_object'] = ind # Include the individual object for fitness values
                processed_valid_solutions.append(solution_data)

        if not processed_valid_solutions:
            print(f"No feasible solutions found for {ex_name}.")

        save_results_to_csv(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure)
        plot_results(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure,
                     col_map, beam_map, beam_sections, column_sections)
        
        return {'stats': hof_stats, 'hof': final_hof, 'time': elapsed, 'name': ex_name}
        
    finally:
        h5_file.close()
        # 원래 config.py의 설정으로 되돌리기
        cfg.FLOORS = original_floors
        cfg.COLUMN_LOCATIONS = original_col_locs
        cfg.BEAM_CONNECTIONS = original_beam_conns
        cfg.BEAM_TRIBUTARY_WIDTHS = original_trib_widths


def main():
    print("### Main Execution: 3 Building Examples Optimization (Varying Plans & Heights) ###")
    
    all_results = []
    
    # 각 예시 실행
    for name, config in EXAMPLES.items():
        result = run_example_optimization(name, config)
        if result:
            all_results.append(result)
        
    if not all_results:
        print("\nNo examples were run successfully. Exiting.")
        return

    # --- 전체 결과 시각화 ---
    
    # 1. Hypervolume Comparison
    plt.figure(figsize=(10, 6))
    for data in all_results:
        hv = [entry['hypervolume'] for entry in data['stats']]
        plt.plot(hv, label=f"{data['name']} ({data['time']:.0f}s)")
    plt.title('Hypervolume Convergence by Building Example')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, 'Examples_HV_Comparison.png'))
    
    # 2. Pareto Front Comparison (Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 만약 예제가 1개라면 axes는 배열이 아닐 수 있음. 리스트로 변환
    if len(all_results) == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    for i, data in enumerate(all_results):
        # hof_data는 individual 객체가 아니라 result dict의 리스트이므로 ind_object에서 fitness 값을 가져옵니다.
        hof_solutions = [sol for sol in data['hof'] if hasattr(sol, 'detailed_results') and sol.detailed_results.get('violation') == 0.0]
        fit1 = [sol.fitness.values[0] for sol in hof_solutions] # Norm Cost+CO2
        fit2 = [sol.fitness.values[1] for sol in hof_solutions] # Mean DCR
        
        ax = axes[i] if len(all_results) > 1 else axes # 1개일 경우 axes 자체가 ax일수도, axes[0]일수도
        # axes가 1d array인 경우와 subplot 1개인 경우 처리 주의
        # plt.subplots(1, 3) -> axes shape (3,)
        # plt.subplots(1, 1) -> ax object (not array) unless squeeze=False
        
        # 여기서 간단히 처리:
        if isinstance(axes, np.ndarray):
            ax = axes.flatten()[i]
        
        ax.scatter(fit2, fit1, c='blue', alpha=0.7)
        ax.set_title(data['name'])
        ax.set_xlabel('Mean DCR')
        ax.set_ylabel('Norm Cost+CO2')
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, 'Examples_Pareto_Comparison.png'))
    
    # 3. Summary CSV
    summary_list = []
    for data in all_results:
        best_hv = data['stats'][-1]['hypervolume']
        # 'hof'는 이제 개별 객체 리스트이므로, 여기서 직접 필터링
        feasible_hof_individuals = [ind for ind in data['hof'] if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0]
        best_ind = min(feasible_hof_individuals, key=lambda x: x.fitness.values[0]) if feasible_hof_individuals else None # Cost 최소 해

        if best_ind:
            summary_list.append({
                'Example': data['name'],
                'Floors': EXAMPLES[data['name']]['Floors'],
                'Time(s)': round(data['time'], 1),
                'Final_HV': round(best_hv, 4),
                'Best_Cost': round(best_ind.detailed_results['cost'], 0),
                'Best_CO2': round(best_ind.detailed_results['co2'], 0),
                'Mean_DCR': round(best_ind.detailed_results['mean_strength_ratio'], 4)
            })
    
    df_sum = pd.DataFrame(summary_list)
    df_sum.to_csv(os.path.join(OUTPUT_BASE_DIR, 'examples_summary.csv'), index=False)
    print(f"\n[Artifacts Saved in {OUTPUT_BASE_DIR}]")

    print("\nTotal Main execution finished.")

if __name__ == "__main__":
    main()
