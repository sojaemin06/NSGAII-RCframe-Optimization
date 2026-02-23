
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

# --- Experiment Configuration ---
OUTPUT_BASE_DIR = "Results_Scenario_Comparison"
POP_SIZE = 500
GENERATIONS = 100
EXAMPLE_NAME = 'Example_1_4Story'

# Use Example 1 Config
cfg.FLOORS = 4
cfg.COLUMN_LOCATIONS = cfg.COLUMN_LOCATIONS_4F
cfg.BEAM_CONNECTIONS = cfg.BEAM_CONNECTIONS_4F
cfg.BEAM_TRIBUTARY_WIDTHS = cfg.BEAM_TRIBUTARY_WIDTHS_4F
DYNAMIC_PATTERNS = cfg.LOAD_PATTERNS_4F
cfg.GROUPING_STRATEGY = "Hybrid" # Match past success, more tractable search space

def run_scenario(scenario_name, use_expanded_db, use_separate_rotation, fixed_scale_info):
    print(f"\n{'='*60}")
    print(f"Running {scenario_name}")
    print(f" - DB: {'Expanded (1600 items)' if use_expanded_db else 'Reduced (800 items)'}")
    print(f" - Rotation Variables: {'Included (Separate Gene)' if use_separate_rotation else 'Excluded (Integrated in DB)'}")
    print(f" - Grouping Strategy: {cfg.GROUPING_STRATEGY}")
    print(f"{'='*60}")
    
    # 1. Load Appropriate Data
    col_db_path = "column_sections_expanded_rotated.csv" if use_expanded_db else "column_sections_reduced.csv"
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data(col_path=col_db_path)
    
    print(f"Loaded Column DB with {len(column_sections)} entries.")
    
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    try:
        # 2. Setup Chromosome Structure
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        
        chromosome_structure = {
            'col_sec': num_col_groups,
            'col_rot': num_col_groups if use_separate_rotation else 0, 
            'beam_sec': num_beam_groups
        }
        
        total_genes = chromosome_structure['col_sec'] + chromosome_structure['col_rot'] + chromosome_structure['beam_sec']
        print(f"Chromosome Length: {total_genes} (Col: {chromosome_structure['col_sec']}, Rot: {chromosome_structure['col_rot']}, Beam: {chromosome_structure['beam_sec']})")

        # 3. Use Provided Normalization Scale
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = fixed_scale_info
        
        start_time = time.time()
        
        # 4. Run Optimization
        pop, logbook, final_hof, hof_stats = run_ga_optimization(
            DL=cfg.DL_AREA_LOAD, LL=cfg.LL_AREA_LOAD,
            crossover_method=cfg.CROSSOVER_STRATEGY, 
            patterns_by_floor=DYNAMIC_PATTERNS,
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
        print(f"Scenario {scenario_name} completed in {elapsed:.2f}s")
        
        # Save Results
        output_dir = os.path.join(OUTPUT_BASE_DIR, scenario_name)
        os.makedirs(output_dir, exist_ok=True)
        
        processed_valid_solutions = []
        for i, ind in enumerate(final_hof):
            if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
                solution_data = ind.detailed_results.copy()
                solution_data['ID'] = i + 1
                solution_data['ind_object'] = ind 
                processed_valid_solutions.append(solution_data)
        
        save_results_to_csv(output_dir, processed_valid_solutions, logbook, hof_stats, chromosome_structure)
        
        return {
            'name': scenario_name,
            'time': elapsed,
            'logbook': logbook,
            'stats': hof_stats,
            'chromosome_len': total_genes
        }

    finally:
        h5_file.close()

def load_existing_results(scenario_name, source_dir):
    print(f"\n{'='*60}")
    print(f"Attempting to Load Existing Results for {scenario_name}")
    print(f"Source: {source_dir}")
    
    log_path = os.path.join(source_dir, "Data", "optimization_log.csv")
    if not os.path.exists(log_path):
        print(f" - Optimization log not found at {log_path}")
        return None
        
    try:
        # Load logbook from CSV using pandas
        df_log = pd.read_csv(log_path)
        
        # Create a dummy logbook-like object (list of dicts)
        logbook = []
        for _, row in df_log.iterrows():
            logbook.append(row.to_dict())
            
        # Get Final HV
        final_hv = df_log.iloc[-1]['hypervolume']
        
        # Chromosome length inference
        design_vars_path = os.path.join(source_dir, "Data", "design_variables.csv")
        chromosome_len = 208 # Default fallback
        
        if os.path.exists(design_vars_path):
            try:
                dv_df = pd.read_csv(design_vars_path, nrows=1) # Read header only
                cols = dv_df.columns
                
                n_col_grps = sum(1 for c in cols if c.startswith('col_grp_') and c.endswith('_ID'))
                n_beam_grps = sum(1 for c in cols if c.startswith('beam_grp_') and c.endswith('_ID'))
                has_rot = any(c.startswith('col_grp_') and c.endswith('_Rot') for c in cols)
                
                chromosome_len = (n_col_grps * (2 if has_rot else 1)) + n_beam_grps
                print(f" - Inferred Chromosome Length from Data: {chromosome_len}")
            except Exception as e:
                print(f" - Warning: Failed to infer chromosome length from CSV ({e}). Using default 208.")
        else:
             print(" - Warning: design_variables.csv not found. Using default 208.")
        
        print(f" - Successfully loaded existing results.")
        return {
            'name': scenario_name,
            'time': 0.0,
            'logbook': logbook,
            'stats': [{'hypervolume': final_hv}],
            'chromosome_len': chromosome_len
        }
    except Exception as e:
        print(f" - Error loading existing results: {e}")
        return None

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # --- SCI Paper Style Configuration ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '-' 
    plt.rcParams['grid.alpha'] = 0.7
    
    # --- 0. Calculate Global Normalization Scale (Using Expanded DB for full range) ---
    print("Calculating Global Normalization Scale...")
    b_df, c_df, _, _ = load_section_data(col_path="column_sections_expanded_rotated.csv")
    num_locations = len(cfg.COLUMN_LOCATIONS)
    num_columns = num_locations * cfg.FLOORS
    beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
    total_col_len = num_columns * cfg.H
    total_beam_len = sum(beam_lengths) * cfg.FLOORS
    fixed_scale_info = calculate_fixed_scale(c_df, b_df, total_col_len, total_beam_len)

    # --- Scenario A: Proposed (Load Existing or Run New) ---
    # Try multiple possible paths for Example 1 results
    possible_paths = [
        os.path.join(project_root, "Results_Optimization_Paper_Final", "Example_1_4Story"),
        os.path.join(project_root, "Results", "Example_1_4Story")
    ]
    
    results_A = None
    for path in possible_paths:
        if os.path.exists(path):
            results_A = load_existing_results("Scenario_A_Proposed", path)
            if results_A: break
            
    if results_A is None:
        print("\nScenario A: No valid existing results found. Running from scratch (this may take time)...")
        results_A = run_scenario("Scenario_A_Proposed", use_expanded_db=False, use_separate_rotation=True, fixed_scale_info=fixed_scale_info)
    
    # --- Scenario B: Conventional (Run New) ---
    # DB: Expanded (1600), Rot Variables: No
    results_B = run_scenario("Scenario_B_Conventional", use_expanded_db=True, use_separate_rotation=False, fixed_scale_info=fixed_scale_info)
    
    # --- Comparison Visualization ---
    plt.figure(figsize=(10, 6))
    
    # Handle different logbook formats (DEAP Logbook vs List of Dicts)
    def get_data(res):
        if isinstance(res['logbook'], list): # Loaded from CSV
            gens = [entry['gen'] for entry in res['logbook']]
            hvs = [entry['hypervolume'] for entry in res['logbook']]
        else: # DEAP Logbook
            gens = res['logbook'].select('gen')
            hvs = [entry['hypervolume'] for entry in res['logbook']]
        return gens, hvs

    gen_A, hv_A = get_data(results_A)
    gen_B, hv_B = get_data(results_B)
    
    plt.plot(gen_A, hv_A, 'b-o', label=f"Scenario A (Proposed): Hybrid Grouping + Reduced DB\n(GenLen={results_A['chromosome_len']})")
    plt.plot(gen_B, hv_B, 'r-x', label=f"Scenario B (Conventional): Hybrid Grouping + Expanded DB\n(GenLen={results_B['chromosome_len']})")
    
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume Indicator')
    # plt.title('Optimization Efficiency Comparison') # Removed for paper style
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, "Comparison_Hypervolume.png"))
    
    # Summary CSV
    summary = []
    for res in [results_A, results_B]:
        # Handle stats access safely
        if isinstance(res['stats'], list):
             best_hv = res['stats'][-1]['hypervolume']
        else:
             best_hv = res['stats'][-1]['hypervolume']
             
        summary.append({
            'Scenario': res['name'],
            'Time(s)': round(res['time'], 1),
            'Chromosome_Length': res['chromosome_len'],
            'Final_HV': round(best_hv, 4)
        })
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_BASE_DIR, "Comparison_Summary.csv"), index=False)
    print("\nExperiment Complete. Check Results_Scenario_Comparison folder.")

if __name__ == "__main__":
    main()
