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
import src.config as cfg 
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps, visualize_load_patterns
from src.optimization import run_ga_optimization
from src.post_processing import save_results_to_csv, plot_results

# ==================================================================================
# SMOKE TEST CONFIGURATION
# ==================================================================================
EXAMPLES = {
    'Example_1_4Story': {
        'Floors': 4,
        'Col_Locs': cfg.COLUMN_LOCATIONS_4F,
        'Beam_Conns': cfg.BEAM_CONNECTIONS_4F,
        'Trib_Widths': cfg.BEAM_TRIBUTARY_WIDTHS_4F
    }
}

# Reduced parameters for quick verification
OPT_POP_SIZE = 8 # Small population
OPT_GENERATIONS = 2 # Short generation
OPT_CX_METHOD = cfg.CROSSOVER_STRATEGY 
OPT_CX_PROB = cfg.CXPB 
OPT_MUT_PROB = cfg.MUTPB 
OPT_TOURN_SIZE = cfg.TOURNAMENT_SIZE 

cfg.GROUPING_STRATEGY = "Hybrid"
OUTPUT_BASE_DIR = "Results_Smoke_Test_Review"
# ==================================================================================

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman' # Font check

def run_example_optimization(ex_name, ex_config):
    output_folder = os.path.join(OUTPUT_BASE_DIR, ex_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n[SMOKE TEST] Running {ex_name}...")
    
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    original_floors = cfg.FLOORS
    original_col_locs = cfg.COLUMN_LOCATIONS
    original_beam_conns = cfg.BEAM_CONNECTIONS
    original_trib_widths = cfg.BEAM_TRIBUTARY_WIDTHS

    cfg.FLOORS = ex_config['Floors']
    cfg.COLUMN_LOCATIONS = ex_config['Col_Locs']
    cfg.BEAM_CONNECTIONS = ex_config['Beam_Conns']
    cfg.BEAM_TRIBUTARY_WIDTHS = ex_config['Trib_Widths']
    
    if 'Example_1' in ex_name: dynamic_patterns = cfg.LOAD_PATTERNS_4F
    else: dynamic_patterns = cfg.PATTERNS_BY_FLOOR
    
    # Check Figure Generation
    figures_dir = os.path.join(output_folder, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    visualize_load_patterns(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS, dynamic_patterns, figures_dir)
    
    try:
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
        
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
        print(f"   -> Optimization Completed in {elapsed:.1f}s")
        
        # Save Results & Check Plotting
        processed_valid_solutions = []
        for i, ind in enumerate(final_hof):
            if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
                solution_data = ind.detailed_results.copy()
                solution_data['ID'] = i + 1
                solution_data['ind_object'] = ind 
                processed_valid_solutions.append(solution_data)
        
        if not processed_valid_solutions:
            print("   [WARNING] No feasible solutions found in Smoke Test (Expected due to low Gen/Pop).")
        
        save_results_to_csv(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure)
        plot_results(output_folder, processed_valid_solutions, logbook, hof_stats, chromosome_structure,
                     col_map, beam_map, beam_sections, column_sections)
        print("   -> Results Saved & Plotted.")
        
        return True
        
    finally:
        h5_file.close()
        cfg.FLOORS = original_floors
        cfg.COLUMN_LOCATIONS = original_col_locs
        cfg.BEAM_CONNECTIONS = original_beam_conns
        cfg.BEAM_TRIBUTARY_WIDTHS = original_trib_widths

def main():
    print("### SMOKE TEST RE-RUN: Output Verification ###")
    for name, config in EXAMPLES.items():
        run_example_optimization(name, config)
    print("\n[SMOKE TEST FINISHED]")

if __name__ == "__main__":
    main()
