import sys
import os
import time
import h5py
import pandas as pd
import numpy as np
import random
import openseespy.opensees as ops

# Add project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import *
import src.config as cfg
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization

# [FIXED PARAMETERS from Phase 3.0 Specs]
BEST_PARAMS = {
    'Crossover': 'Uniform',
    'Tournament': 3,
    'CXPB': 0.9,
    'MUTPB': 0.7
}
GEN = 100  # Generation count for Population experiment
OUTPUT_ROOT = "Results_Param_Optimization"

def run_pop_batch(pop_list):
    """지정된 Population List에 대해 순차적으로 실험 수행"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 1. Load Data & Config
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_path = os.path.join(project_root, 'pm_dataset_simple02.mat')
    h5_file = h5py.File(h5_path, 'r')

    try:
        # Configuration for Example 1 (4 Story)
        cfg.FLOORS = 4
        cfg.COLUMN_LOCATIONS = cfg.COLUMN_LOCATIONS_4F
        cfg.BEAM_CONNECTIONS = cfg.BEAM_CONNECTIONS_4F
        cfg.BEAM_TRIBUTARY_WIDTHS = cfg.BEAM_TRIBUTARY_WIDTHS_4F
        cfg.GROUPING_STRATEGY = "Hybrid"

        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)

        _, _, col_map, beam_map = get_grouping_maps(
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        chromosome_structure = {
            'col_sec': len(set(col_map.values())), 
            'col_rot': len(set(col_map.values())), 
            'beam_sec': len(set(beam_map.values()))
        }

        total_col_len = (len(cfg.COLUMN_LOCATIONS) * cfg.FLOORS) * cfg.H
        total_beam_len = sum(beam_lengths) * cfg.FLOORS
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
            column_sections_df, beam_sections_df, total_col_len, total_beam_len
        )

        # 2. Run Experiment Loop
        for pop_size in pop_list:
            print(f"\n>>> Starting Experiment: Population Size = {pop_size}")
            print(f"    Params: {BEST_PARAMS}, Gen: {GEN}")
            
            # Reset System
            ops.wipe()
            random.seed(42)
            np.random.seed(42)
            
            start_time = time.time()
            
            # Run GA
            final_pop, _, final_hof, hof_stats_history = run_ga_optimization(
                DL=DL_AREA_LOAD, LL=LL_AREA_LOAD,
                crossover_method=BEST_PARAMS['Crossover'], 
                patterns_by_floor=cfg.LOAD_PATTERNS_4F, 
                h5_file=h5_file,
                num_generations=GEN, 
                population_size=pop_size,
                col_map=col_map, beam_map=beam_map, 
                beam_sections=beam_sections, column_sections=column_sections,
                beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, 
                beam_lengths=beam_lengths,
                chromosome_structure=chromosome_structure, 
                num_columns=num_columns, num_beams=num_beams,
                fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
                fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
                tournament_size=BEST_PARAMS['Tournament'], 
                cxpb=BEST_PARAMS['CXPB'], 
                mutpb=BEST_PARAMS['MUTPB'],
                verbose=True
            )
            
            elapsed = time.time() - start_time
            print(f"    Done in {elapsed:.1f}s. Saving results...")

            # 3. Save Individual Results
            # (1) History
            df_hist = pd.DataFrame(hof_stats_history)
            df_hist['PopSize'] = pop_size
            hist_path = os.path.join(OUTPUT_ROOT, f"Step5_Pop_{pop_size}_History.csv")
            df_hist.to_csv(hist_path, index=False)
            
            # (2) Pareto Front (HOF)
            hof_rows = []
            final_hv = hof_stats_history[-1]['hypervolume'] if hof_stats_history else 0.0
            for ind in final_hof:
                # Violation check included in detailed_results
                if ind.detailed_results.get('violation', 0) == 0:
                    hof_rows.append({
                        'Parameter': pop_size,
                        'Obj1_NormCostCO2': ind.fitness.values[0],
                        'Obj2_MaxDrift': ind.fitness.values[1],
                        'Cost': ind.detailed_results.get('cost', 0),
                        'CO2': ind.detailed_results.get('co2', 0),
                        'Max_Drift': ind.detailed_results.get('max_drift_ratio', 0),
                        'Hypervolume': final_hv
                    })
            
            df_pareto = pd.DataFrame(hof_rows)
            pareto_path = os.path.join(OUTPUT_ROOT, f"Step5_Pop_{pop_size}_Pareto.csv")
            df_pareto.to_csv(pareto_path, index=False)
            print(f"    -> Saved: {hist_path}")
            print(f"    -> Saved: {pareto_path}")

    finally:
        h5_file.close()
