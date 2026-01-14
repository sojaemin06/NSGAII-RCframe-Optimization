import sys
import os

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import time
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from src.config import *
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization
import openseespy.opensees as ops

# ==================================================================================
# [PART 2] Population Size Experiment (500 ~ 700)
# ==================================================================================

# 1. 설정
PART_NAME = "Step5_PopSize_Part2"
POP_RANGE = [500, 600, 700]       # 실행할 모집단 크기 범위
STEP5_GEN = 100                   # 세대 수

# 이전 단계 최적 파라미터 (고정)
BEST_PARAMS = {
    'Crossover': 'TwoPoint', 
    'Tournament': 3,         
    'CXPB': 0.9,             
    'MUTPB': 0.7            
}
OUTPUT_ROOT = "Results_Param_Optimization"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

def run_single_experiment(exp_name, pop_size, tourn_size, crossover, cxpb, mutpb, num_gen, common_data, pbar_desc=""):
    ops.wipe()
    random.seed(42)
    np.random.seed(42)

    (beam_sections_df, column_sections_df, beam_sections, column_sections,
     h5_file, col_map, beam_map, beam_lengths, chromosome_structure,
     num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2) = common_data

    print(f"{pbar_desc:<50} | Pop:{pop_size:<4} ... ", end='', flush=True)
    start_time = time.time()
    
    final_pop, _, final_hof, hof_stats_history = run_ga_optimization(
        DL=DL_AREA_LOAD, LL=LL_AREA_LOAD,
        crossover_method=crossover, patterns_by_floor=PATTERNS_BY_FLOOR, h5_file=h5_file,
        num_generations=num_gen, population_size=pop_size,
        col_map=col_map, beam_map=beam_map, beam_sections=beam_sections, column_sections=column_sections,
        beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, beam_lengths=beam_lengths,
        chromosome_structure=chromosome_structure, num_columns=num_columns, num_beams=num_beams,
        fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
        fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
        tournament_size=tourn_size, cxpb=cxpb, mutpb=mutpb,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    final_hv = hof_stats_history[-1]['hypervolume'] if hof_stats_history else 0.0
    print(f"Done ({elapsed:.1f}s) | HV: {final_hv:.4f}")
    
    for entry in hof_stats_history:
        entry.update({
            'Experiment': exp_name, 'PopSize': pop_size, 
            'Tournament': tourn_size, 'Crossover': crossover, 
            'CXPB': cxpb, 'MUTPB': mutpb
        })
        
    return hof_stats_history, final_hof, final_pop

def save_part_results(all_history_data, all_hof_data, step_name, output_dir):
    """중간 결과 저장 (Part별 분리 저장)"""
    df_hist = pd.DataFrame(all_history_data)
    hist_path = os.path.join(output_dir, f'{step_name}_HV_History.csv')
    df_hist.to_csv(hist_path, index=False)
    print(f"   -> [Saved] History to {hist_path}")

    new_hof_rows = []
    for param, inds in all_hof_data.items():
        subset = df_hist[df_hist['PopSize'] == param]
        final_hv = subset.iloc[-1]['hypervolume'] if not subset.empty else 0.0
        
        for ind in inds:
            new_hof_rows.append({
                'Parameter': param,
                'Obj1_NormCostCO2': ind.fitness.values[0],
                'Obj2_MaxDrift': ind.fitness.values[1],
                'Cost': ind.detailed_results.get('cost', 0),
                'CO2': ind.detailed_results.get('co2', 0),
                'Max_Drift': ind.detailed_results.get('max_drift_ratio', 0),
                'Hypervolume': final_hv
            })
    
    df_hof = pd.DataFrame(new_hof_rows)
    hof_path = os.path.join(output_dir, f'{step_name}_Pareto_Data.csv')
    df_hof.to_csv(hof_path, index=False)
    print(f"   -> [Saved] HOF Data to {hof_path}")

def main():
    print(f"### Experiment: {PART_NAME} Started ###")
    print(f"Target Populations: {POP_RANGE}")
    
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    try:
        forced_grouping_strategy = "Hybrid"
        num_locations = len(COLUMN_LOCATIONS)
        num_columns = num_locations * FLOORS
        num_beams = len(BEAM_CONNECTIONS) * FLOORS
        beam_lengths = get_beam_lengths(COLUMN_LOCATIONS, BEAM_CONNECTIONS)
        
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            forced_grouping_strategy, num_locations, num_columns, num_beams, FLOORS, BEAM_CONNECTIONS, COLUMN_LOCATIONS
        )
        chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
        
        total_col_len = (len(COLUMN_LOCATIONS) * FLOORS) * H
        total_beam_len = sum(beam_lengths) * FLOORS
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
            column_sections_df, beam_sections_df, total_col_len, total_beam_len
        )
        
        common_data = (beam_sections_df, column_sections_df, beam_sections, column_sections,
                       h5_file, col_map, beam_map, beam_lengths, chromosome_structure,
                       num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2)
        
        history, hof_data = [], {}
        
        for i, val in enumerate(POP_RANGE):
            exp_name = f"PopSize_{val}"
            desc = f"({i+1}/{len(POP_RANGE)}) Testing PopSize={val}"
            
            hist, hof, _ = run_single_experiment(
                exp_name, val, BEST_PARAMS['Tournament'], BEST_PARAMS['Crossover'], 
                BEST_PARAMS['CXPB'], BEST_PARAMS['MUTPB'], STEP5_GEN, common_data, pbar_desc=desc
            )
            history.extend(hist)
            hof_data[val] = hof
            
        save_part_results(history, hof_data, PART_NAME, OUTPUT_ROOT)
        print("\n[Part 2 Complete]")

    finally:
        h5_file.close()

if __name__ == "__main__":
    main()
