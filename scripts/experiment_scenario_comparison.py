import sys
import os
import warnings
import random

# Suppress all warnings
warnings.filterwarnings("ignore")

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import src.config as cfg
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization
from src.post_processing import save_results_to_csv

# --- [Configuration] ---
RANDOM_SEED = 42  
OUTPUT_BASE_DIR = "Results_Scenario_Comparison"
POP_SIZE = 500
GENERATIONS = 100

# Example Configuration List
EXAMPLE_CONFIGS = [
    {
        'id': 'Example_1_4Story',
        'floors': 4,
        'col_locs': cfg.COLUMN_LOCATIONS_4F,
        'beam_conn': cfg.BEAM_CONNECTIONS_4F,
        'beam_trib': cfg.BEAM_TRIBUTARY_WIDTHS_4F,
        'patterns': cfg.LOAD_PATTERNS_4F
    },
    # {
    #     'id': 'Example_2_6Story',
    #     'floors': 6,
    #     'col_locs': cfg.COLUMN_LOCATIONS_6F,
    #     'beam_conn': cfg.BEAM_CONNECTIONS_6F,
    #     'beam_trib': cfg.BEAM_TRIBUTARY_WIDTHS_6F,
    #     'patterns': cfg.LOAD_PATTERNS_6F
    # },
    # {
    #     'id': 'Example_3_8Story',
    #     'floors': 8,
    #     'col_locs': cfg.COLUMN_LOCATIONS_8F,
    #     'beam_conn': cfg.BEAM_CONNECTIONS_8F,
    #     'beam_trib': cfg.BEAM_TRIBUTARY_WIDTHS_8F,
    #     'patterns': cfg.LOAD_PATTERNS_8F
    # }
]

def update_config(config):
    """Update global config based on the selected example."""
    cfg.FLOORS = config['floors']
    cfg.COLUMN_LOCATIONS = config['col_locs']
    cfg.BEAM_CONNECTIONS = config['beam_conn']
    cfg.BEAM_TRIBUTARY_WIDTHS = config['beam_trib']
    
    # Recalculate dimensions for wind load
    all_x = [p[0] for p in cfg.COLUMN_LOCATIONS]
    all_y = [p[1] for p in cfg.COLUMN_LOCATIONS]
    cfg.BUILDING_WIDTH_X = max(all_x) - min(all_x)
    cfg.BUILDING_WIDTH_Y = max(all_y) - min(all_y)
    
    return config['patterns']

def run_scenario_a(example_dir, fixed_scale_info, dynamic_patterns):
    """Run Scenario A (Proposed): Standard DB + Separate Rotation."""
    scenario_name = "Scenario_A_Proposed"
    print(f"\nRunning {scenario_name}...")
    
    # --- Set Seed for Fair Comparison ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load Standard DB (simple02)
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data(
        col_path="column_sections_simple02.csv"
    )
    
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    try:
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            "Hybrid", num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        
        # Scenario A: Separate rotation variable
        chromosome_structure = {
            'col_sec': num_col_groups,
            'col_rot': num_col_groups, 
            'beam_sec': num_beam_groups
        }
        
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = fixed_scale_info
        
        start_time = time.time()
        pop, logbook, final_hof, hof_stats = run_ga_optimization(
            DL=cfg.DL_AREA_LOAD, LL=cfg.LL_AREA_LOAD,
            crossover_method=cfg.CROSSOVER_STRATEGY, 
            patterns_by_floor=dynamic_patterns,
            h5_file=h5_file,
            num_generations=GENERATIONS, population_size=POP_SIZE,
            col_map=col_map, beam_map=beam_map, 
            beam_sections=beam_sections, column_sections=column_sections,
            beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, 
            beam_lengths=beam_lengths, chromosome_structure=chromosome_structure, 
            num_columns=num_columns, num_beams=num_beams,
            fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
            fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
            tournament_size=cfg.TOURNAMENT_SIZE, cxpb=cfg.CXPB, mutpb=cfg.MUTPB,
            verbose=True
        )
        elapsed = time.time() - start_time
        
        output_dir = os.path.join(example_dir, scenario_name)
        os.makedirs(output_dir, exist_ok=True)
        
        processed_solutions = []
        for i, ind in enumerate(final_hof):
            if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
                sol = ind.detailed_results.copy()
                sol['ID'] = i + 1
                sol['ind_object'] = ind
                processed_solutions.append(sol)
        
        save_results_to_csv(output_dir, processed_solutions, logbook, hof_stats, chromosome_structure)
        
        return {
            'name': scenario_name,
            'logbook': logbook,
            'chromosome_len': sum(chromosome_structure.values()),
            'time': elapsed
        }
    finally:
        h5_file.close()

def run_scenario_b(example_dir, fixed_scale_info, dynamic_patterns):
    """Run Scenario B (Conventional): Expanded DB + Integrated Rotation."""
    scenario_name = "Scenario_B_Conventional"
    print(f"\nRunning {scenario_name}...")
    
    # --- Set Seed for Fair Comparison ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load Expanded DB
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data(
        col_path="column_sections_expanded_rotated.csv"
    )
    
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    try:
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            "Hybrid", num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        
        # Scenario B: No separate rotation variable
        chromosome_structure = {
            'col_sec': num_col_groups,
            'col_rot': 0, 
            'beam_sec': num_beam_groups
        }
        
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = fixed_scale_info
        
        start_time = time.time()
        pop, logbook, final_hof, hof_stats = run_ga_optimization(
            DL=cfg.DL_AREA_LOAD, LL=cfg.LL_AREA_LOAD,
            crossover_method=cfg.CROSSOVER_STRATEGY, 
            patterns_by_floor=dynamic_patterns,
            h5_file=h5_file,
            num_generations=GENERATIONS, population_size=POP_SIZE,
            col_map=col_map, beam_map=beam_map, 
            beam_sections=beam_sections, column_sections=column_sections,
            beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, 
            beam_lengths=beam_lengths, chromosome_structure=chromosome_structure, 
            num_columns=num_columns, num_beams=num_beams,
            fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
            fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
            tournament_size=cfg.TOURNAMENT_SIZE, cxpb=cfg.CXPB, mutpb=cfg.MUTPB,
            verbose=True
        )
        elapsed = time.time() - start_time
        
        output_dir = os.path.join(example_dir, scenario_name)
        os.makedirs(output_dir, exist_ok=True)
        
        processed_solutions = []
        for i, ind in enumerate(final_hof):
            if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
                sol = ind.detailed_results.copy()
                sol['ID'] = i + 1
                sol['ind_object'] = ind
                processed_solutions.append(sol)
        
        save_results_to_csv(output_dir, processed_solutions, logbook, hof_stats, chromosome_structure)
        
        return {
            'name': scenario_name,
            'logbook': logbook,
            'chromosome_len': sum(chromosome_structure.values()),
            'time': elapsed
        }
    finally:
        h5_file.close()

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    for config in EXAMPLE_CONFIGS:
        example_id = config['id']
        example_dir = os.path.join(OUTPUT_BASE_DIR, example_id)
        os.makedirs(example_dir, exist_ok=True)
        
        print(f"\n{'#'*80}")
        print(f"### SCENARIO COMPARISON EXPERIMENT: {example_id} ###")
        print(f"### RANDOM SEED: {RANDOM_SEED} ###")
        print(f"{'#'*80}")
        
        # 1. Update Config and Load Patterns
        dynamic_patterns = update_config(config)
        
        # 2. Calculate Normalization Scale for the current building size
        # Use Scenario B's DB for scale calculation to be consistent (covers full range)
        b_df, c_df, _, _ = load_section_data(col_path="column_sections_expanded_rotated.csv")
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        total_col_len = (len(cfg.COLUMN_LOCATIONS) * cfg.FLOORS) * cfg.H
        total_beam_len = sum(beam_lengths) * cfg.FLOORS
        fixed_scale_info = calculate_fixed_scale(c_df, b_df, total_col_len, total_beam_len)

        # 3. Run Scenario A (Proposed)
        results_A = run_scenario_a(example_dir, fixed_scale_info, dynamic_patterns)
        
        # 4. Run Scenario B (Conventional)
        results_B = run_scenario_b(example_dir, fixed_scale_info, dynamic_patterns)

        # 5. Summary and Plotting
        results_list = [results_A, results_B]
        summary = []
        for res in results_list:
            summary.append({
                'Scenario': res['name'],
                'Chromosome_Length': res['chromosome_len'],
                'Time(s)': round(res['time'], 1),
                'Final_HV': round(res['logbook'][-1]['hypervolume'], 4)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(example_dir, f"Summary_{example_id}.csv"), index=False)
        print(f"\nSummary for {example_id}:")
        print(summary_df.to_string(index=False))

        # --- Plotting to match the reference image ---
        plt.figure(figsize=(10, 7), dpi=150)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        # Legend order and styles: B (Red dashed) then A (Blue solid)
        # Note: We plot them in the order we want them to appear in the legend
        
        # Plot Scenario B
        gens_B = [e['gen'] for e in results_B['logbook']]
        hvs_B = [e['hypervolume'] for e in results_B['logbook']]
        label_B = f"Scenario B:\nUsing Expanded Column DB (Len={results_B['chromosome_len']})"
        plt.plot(gens_B, hvs_B, color='#C0392B', linestyle='--', linewidth=2.5, label=label_B)
        
        # Plot Scenario A
        gens_A = [e['gen'] for e in results_A['logbook']]
        hvs_A = [e['hypervolume'] for e in results_A['logbook']]
        label_A = f"Scenario A:\nUsing Rotation Genes (Len={results_A['chromosome_len']})"
        plt.plot(gens_A, hvs_A, color='#2C3E50', linestyle='-', linewidth=2.5, label=label_A)

        plt.xlabel('Generation', fontweight='bold', fontsize=14)
        plt.ylabel('Hypervolume Indicator', fontweight='bold', fontsize=14)
        
        # Grid settings
        plt.grid(True, which='both', linestyle='-', alpha=0.4, color='#CCCCCC')
        
        # Legend settings
        plt.legend(
            loc='lower right', 
            frameon=True, 
            edgecolor='black', 
            facecolor='white',
            framealpha=1.0,
            fontsize=11,
            labelspacing=0.8
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(example_dir, f"Comparison_HV_{example_id}.png"))
        plt.close()
        
        print(f"\nExperiment for {example_id} complete.")

if __name__ == "__main__":
    main()
