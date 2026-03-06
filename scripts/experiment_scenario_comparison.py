import sys
import os
import warnings

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

# --- [Configuration] Set Example Index to run (0: Example 1, 1: Example 2, 2: Example 3) ---
EXAMPLE_INDEX = 1  

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
    {
        'id': 'Example_2_6Story',
        'floors': 6,
        'col_locs': cfg.COLUMN_LOCATIONS_6F,
        'beam_conn': cfg.BEAM_CONNECTIONS_6F,
        'beam_trib': cfg.BEAM_TRIBUTARY_WIDTHS_6F,
        'patterns': cfg.LOAD_PATTERNS_6F
    },
    {
        'id': 'Example_3_8Story',
        'floors': 8,
        'col_locs': cfg.COLUMN_LOCATIONS_8F,
        'beam_conn': cfg.BEAM_CONNECTIONS_8F,
        'beam_trib': cfg.BEAM_TRIBUTARY_WIDTHS_8F,
        'patterns': cfg.LOAD_PATTERNS_8F
    }
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

def load_scenario_a_results(example_id):
    """Load existing Scenario A (Proposed) results from Results_Optimization_Paper_Final."""
    source_dir = os.path.join("Results_Optimization_Paper_Final", example_id)
    print(f"\nLoading Scenario A (Proposed) from: {source_dir}")
    
    log_path = os.path.join(source_dir, "Data", "optimization_log.csv")
    if not os.path.exists(log_path):
        print(f"Warning: Existing results not found at {log_path}")
        return None
        
    try:
        df_log = pd.read_csv(log_path)
        logbook = [row.to_dict() for _, row in df_log.iterrows()]
        
        # Infer chromosome length
        design_vars_path = os.path.join(source_dir, "Data", "design_variables.csv")
        chromosome_len = 0
        if os.path.exists(design_vars_path):
            dv_df = pd.read_csv(design_vars_path, nrows=1)
            cols = dv_df.columns
            n_col_grps = sum(1 for c in cols if c.startswith('col_grp_') and c.endswith('_ID'))
            n_beam_grps = sum(1 for c in cols if c.startswith('beam_grp_') and c.endswith('_ID'))
            has_rot = any(c.startswith('col_grp_') and c.endswith('_Rot') for c in cols)
            chromosome_len = (n_col_grps * (2 if has_rot else 1)) + n_beam_grps
            
        return {
            'name': 'Scenario_A_Proposed',
            'logbook': logbook,
            'chromosome_len': chromosome_len,
            'time': 0.0
        }
    except Exception as e:
        print(f"Error loading Scenario A: {e}")
        return None

def run_scenario_b(example_dir, fixed_scale_info, dynamic_patterns):
    """Run Scenario B (Conventional): Expanded DB + Integrated Rotation."""
    scenario_name = "Scenario_B_Conventional"
    print(f"\nRunning {scenario_name}")
    
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
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
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
    if EXAMPLE_INDEX >= len(EXAMPLE_CONFIGS):
        print(f"Invalid EXAMPLE_INDEX: {EXAMPLE_INDEX}")
        return

    config = EXAMPLE_CONFIGS[EXAMPLE_INDEX]
    example_id = config['id']
    example_dir = os.path.join(OUTPUT_BASE_DIR, example_id)
    os.makedirs(example_dir, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"### SCENARIO COMPARISON EXPERIMENT: {example_id} ###")
    print(f"{'#'*80}")
    
    # 1. Update Config and Load Patterns
    dynamic_patterns = update_config(config)
    
    # 2. Calculate Normalization Scale for the current building size
    b_df, c_df, _, _ = load_section_data(col_path="column_sections_expanded_rotated.csv")
    beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
    total_col_len = (len(cfg.COLUMN_LOCATIONS) * cfg.FLOORS) * cfg.H
    total_beam_len = sum(beam_lengths) * cfg.FLOORS
    fixed_scale_info = calculate_fixed_scale(c_df, b_df, total_col_len, total_beam_len)

    # 3. Load Scenario A Results (Proposed)
    results_A = load_scenario_a_results(example_id)
    
    # 4. Run Scenario B (Conventional)
    results_B = run_scenario_b(example_dir, fixed_scale_info, dynamic_patterns)

    # 5. Summary and Plotting
    results_list = []
    if results_A: results_list.append(results_A)
    results_list.append(results_B)

    summary = []
    for res in results_list:
        summary.append({
            'Scenario': res['name'],
            'Chromosome_Length': res['chromosome_len'],
            'Time(s)': round(res['time'], 1),
            'Final_HV': round(res['logbook'][-1]['hypervolume'], 4)
        })
    pd.DataFrame(summary).to_csv(os.path.join(example_dir, f"Summary_{example_id}.csv"), index=False)

    plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    
    colors = ['#2C3E50', '#C0392B']
    linestyles = ['-', '--']
    
    for res, color, ls in zip(results_list, colors, linestyles):
        gens = [e['gen'] for e in res['logbook']]
        hvs = [e['hypervolume'] for e in res['logbook']]
        label = f"{res['name']} (Len={res['chromosome_len']})"
        plt.plot(gens, hvs, color=color, linestyle=ls, linewidth=2, label=label)
        
    plt.xlabel('Generation', fontweight='bold')
    plt.ylabel('Hypervolume Indicator', fontweight='bold')
    plt.title(f'Hypervolume Convergence Comparison - {example_id}', fontweight='bold')
    plt.legend(frameon=True, loc='lower right', edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(example_dir, f"Comparison_HV_{example_id}.png"))
    
    print(f"\nExperiment for {example_id} complete.")
    print(f"Results saved in: {example_dir}")

if __name__ == "__main__":
    main()
