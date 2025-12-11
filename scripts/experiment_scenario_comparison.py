import os
import time
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import *
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization

# ==================================================================================
# [USER CONFIGURATION] 시나리오별 파일 경로 및 설정
# ==================================================================================

# Scenario A: 기존 방식 (회전 유전자 사용)
SCENARIO_A = {
    'name': 'Scenario A (Gene Rotation)',
    'beam_csv': 'beam_sections_simple02.csv',
    'col_csv': 'column_sections_simple02.csv',
    'mat_file': 'pm_dataset_simple02.mat',
    'use_rotation_gene': True
}

# Scenario B: 확장 DB 방식 (회전된 단면 포함 DB 사용, 회전 유전자 미사용)
SCENARIO_B = {
    'name': 'Scenario B (Expanded DB)',
    'beam_csv': 'beam_sections_simple02.csv', # 보는 기존과 동일하다고 가정
    'col_csv': 'column_sections_expanded_rotated.csv', 
    'mat_file': 'pm_dataset_expanded_rotated.mat',     
    'use_rotation_gene': False
}

# 공통 실험 파라미터
POP_SIZE = 50     # 비교용이므로 조금 작게 설정 (필요시 수정)
GENERATIONS = 50 
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.1
CROSSOVER_METHOD = 'TwoPoint'

OUTPUT_DIR = "Results_Scenario_Comparison"

# ==================================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

def run_scenario(scenario_config):
    print(f"\n>>> Running {scenario_config['name']} ...")
    
    # 1. 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data(
        beam_path=scenario_config['beam_csv'],
        col_path=scenario_config['col_csv']
    )
    
    # 2. 구조 정보 생성
    num_locations = len(COLUMN_LOCATIONS)
    num_columns = num_locations * FLOORS
    num_beams = len(BEAM_CONNECTIONS) * FLOORS
    beam_lengths = get_beam_lengths(COLUMN_LOCATIONS, BEAM_CONNECTIONS)
    
    num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
        GROUPING_STRATEGY, num_locations, num_columns, num_beams, FLOORS, BEAM_CONNECTIONS, COLUMN_LOCATIONS
    )
    
    # 3. 유전자 구조 설정 (Scenario B에서는 col_rot 제거)
    chromosome_structure = {
        'col_sec': num_col_groups,
        'col_rot': num_col_groups if scenario_config['use_rotation_gene'] else 0, # <--- 핵심 차이
        'beam_sec': num_beam_groups
    }
    
    # 4. 정규화 스케일 계산
    total_col_len = (len(COLUMN_LOCATIONS) * FLOORS) * H
    total_beam_len = sum(beam_lengths) * FLOORS
    fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
        column_sections_df, beam_sections_df, total_col_len, total_beam_len
    )
    
    # 5. GA 실행
    # .mat 파일이 없을 경우 예외처리 안내
    if not os.path.exists(scenario_config['mat_file']):
        print(f"[Error] MAT file not found: {scenario_config['mat_file']}")
        print("Please place the file in the correct directory or update the script path.")
        return None, None, None

    with h5py.File(scenario_config['mat_file'], 'r') as h5_file:
        start_time = time.time()
        pop, logbook, final_hof, hof_stats = run_ga_optimization(
            DL=DL_AREA_LOAD, LL=LL_AREA_LOAD,
            crossover_method=CROSSOVER_METHOD, 
            patterns_by_floor=PATTERNS_BY_FLOOR, h5_file=h5_file,
            num_generations=GENERATIONS, population_size=POP_SIZE,
            col_map=col_map, beam_map=beam_map, 
            beam_sections=beam_sections, column_sections=column_sections,
            beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, 
            beam_lengths=beam_lengths, chromosome_structure=chromosome_structure, 
            num_columns=num_columns, num_beams=num_beams,
            fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
            fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
            tournament_size=TOURNAMENT_SIZE, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB,
            verbose=True
        )
        elapsed = time.time() - start_time
        print(f"   Done in {elapsed:.1f}s")
        
        return hof_stats, final_hof, elapsed

def main():
    print("### Scenario Comparison Experiment: Gene Rotation vs Expanded DB ###")
    
    # Run Scenario A
    stats_a, hof_a, time_a = run_scenario(SCENARIO_A)
    
    # Run Scenario B (Optional check)
    stats_b, hof_b, time_b = run_scenario(SCENARIO_B)
    
    if stats_a is None or stats_b is None:
        print("\n[Comparison Aborted] One or more scenarios failed to run.")
        return

    # --- Plot 1: Hypervolume Convergence ---
    plt.figure(figsize=(10, 6))
    hv_a = [entry['hypervolume'] for entry in stats_a]
    hv_b = [entry['hypervolume'] for entry in stats_b]
    
    plt.plot(hv_a, label=f"Scenario A (Gene Rot) - {time_a:.0f}s", color='blue', marker='o', markersize=3)
    plt.plot(hv_b, label=f"Scenario B (Expanded DB) - {time_b:.0f}s", color='red', marker='x', markersize=3)
    
    plt.title('Comparison of Convergence Speed (Hypervolume)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Scenario_Comparison_HV.png'))
    
    # --- Plot 2: Pareto Front Comparison ---
    plt.figure(figsize=(10, 8))
    
    fit1_a = [ind.fitness.values[0] for ind in hof_a]
    fit2_a = [ind.fitness.values[1] for ind in hof_a]
    plt.scatter(fit2_a, fit1_a, c='blue', label='Scenario A', alpha=0.6, s=40)
    
    fit1_b = [ind.fitness.values[0] for ind in hof_b]
    fit2_b = [ind.fitness.values[1] for ind in hof_b]
    plt.scatter(fit2_b, fit1_b, c='red', label='Scenario B', alpha=0.6, marker='x', s=40)
    
    plt.title('Comparison of Pareto Fronts')
    plt.xlabel('Structural Conservatism (Mean DCR)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Scenario_Comparison_Pareto.png'))
    
    # --- Save Summary CSV ---
    summary_data = []
    # A Data
    for ind in hof_a:
        summary_data.append({'Scenario': 'A', 'Obj1': ind.fitness.values[0], 'Obj2': ind.fitness.values[1], 'Cost': ind.detailed_results.get('cost',0), 'CO2': ind.detailed_results.get('co2',0)})
    # B Data
    for ind in hof_b:
        summary_data.append({'Scenario': 'B', 'Obj1': ind.fitness.values[0], 'Obj2': ind.fitness.values[1], 'Cost': ind.detailed_results.get('cost',0), 'CO2': ind.detailed_results.get('co2',0)})
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, 'scenario_comparison_summary.csv'), index=False)
    
    print(f"\n[Artifacts Saved in {OUTPUT_DIR}]")

if __name__ == "__main__":
    main()
