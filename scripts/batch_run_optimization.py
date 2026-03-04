import os
import sys
import time
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import *
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization

OPTIMAL_PARAMS = {
    'Population_Size': 500,
    'Tournament_Size': 3,
    'Crossover_Method': 'Uniform',
    'Crossover_Prob': 0.9,
    'Mutation_Prob': 0.7,
    'Generations': 100
}

NUM_REPEATS = 10 
OUTPUT_DIR = "Results_Statistical_Validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

def run_validation_instance(run_id, common_data):
    (beam_sections_df, column_sections_df, beam_sections, column_sections,
     h5_file_path, col_map, beam_map, beam_lengths, chromosome_structure,
     num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2) = common_data

    with h5py.File(h5_file_path, 'r') as h5_file:
        start_time = time.time()
        pop, logbook, final_hof, hof_stats = run_ga_optimization(
            DL=DL_AREA_LOAD, LL=LL_AREA_LOAD,
            crossover_method=OPTIMAL_PARAMS['Crossover_Method'], 
            patterns_by_floor=PATTERNS_BY_FLOOR, h5_file=h5_file,
            num_generations=OPTIMAL_PARAMS['Generations'], 
            population_size=OPTIMAL_PARAMS['Population_Size'],
            col_map=col_map, beam_map=beam_map, 
            beam_sections=beam_sections, column_sections=column_sections,
            beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, 
            beam_lengths=beam_lengths, chromosome_structure=chromosome_structure, 
            num_columns=num_columns, num_beams=num_beams,
            fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
            fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
            tournament_size=OPTIMAL_PARAMS['Tournament_Size'], 
            cxpb=OPTIMAL_PARAMS['Crossover_Prob'], 
            mutpb=OPTIMAL_PARAMS['Mutation_Prob'],
            verbose=True,
            tqdm_pos=run_id
        )
        elapsed = time.time() - start_time
        best_hv = hof_stats[-1]['hypervolume']
        best_cost_ind = min(final_hof, key=lambda x: x.fitness.values[0])
        pareto_front = []
        for ind in final_hof:
            pareto_front.append({
                'Run_ID': run_id,
                'Obj1': ind.fitness.values[0], 
                'Obj2': ind.fitness.values[1], 
                'Cost': ind.detailed_results.get('cost', 0),
                'CO2': ind.detailed_results.get('co2', 0),
                'Max_Drift': ind.detailed_results.get('max_drift_ratio', 0)
            })
        return {
            'Run_ID': run_id, 'Hypervolume': best_hv, 'Best_Cost': best_cost_ind.detailed_results['cost'],
            'Best_CO2': best_cost_ind.detailed_results['co2'], 'Best_Drift': best_cost_ind.detailed_results['max_drift_ratio'],
            'Elapsed_Time': elapsed, 'HV_History': [e['hypervolume'] for entry in hof_stats], 'Pareto_Solutions': pareto_front
        }

def main():
    print(f"### Statistical Validation ({NUM_REPEATS} runs) ###")
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    num_locs = len(COLUMN_LOCATIONS); n_cols, n_beams = num_locs*FLOORS, len(BEAM_CONNECTIONS)*FLOORS
    beam_lens = get_beam_lengths(COLUMN_LOCATIONS, BEAM_CONNECTIONS)
    num_col_g, num_beam_g, col_map, beam_map = get_grouping_maps(GROUPING_STRATEGY, num_locs, n_cols, n_beams, FLOORS, BEAM_CONNECTIONS, COLUMN_LOCATIONS)
    chrom_struct = {'col_sec': num_col_g, 'col_rot': num_col_g, 'beam_sec': num_beam_g}
    f_min_c, f_range_c, f_min_e, f_range_e = calculate_fixed_scale(column_sections_df, beam_sections_df, n_cols*H, sum(beam_lens)*FLOORS)
    common_data = (beam_sections_df, column_sections_df, beam_sections, column_sections, 'pm_dataset_simple02.mat', col_map, beam_map, beam_lens, chrom_struct, n_cols, n_beams, f_min_c, f_range_c, f_min_e, f_range_e)

    summary_results, all_hv, all_pareto = [], [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_validation_instance, i+1, common_data): i for i in range(NUM_REPEATS)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_REPEATS, desc="Validation Progress"):
            res = future.result()
            summary_results.append({k: v for k, v in res.items() if k not in ['HV_History', 'Pareto_Solutions']})
            all_hv.append(res['HV_History'])
            all_pareto.extend(res['Pareto_Solutions'])

    df_results = pd.DataFrame(summary_results).sort_values('Run_ID')
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'statistical_summary.csv'), index=False)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, m in enumerate(['Hypervolume', 'Best_Cost', 'Best_CO2', 'Best_Drift']):
        axes[i].boxplot(df_results[m], patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[i].set_title(m)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_metrics.png'))

    df_hv = pd.DataFrame(all_hv); df_hv.to_csv(os.path.join(OUTPUT_DIR, 'convergence_history.csv'), index_label="Run_ID")
    plt.figure(figsize=(10, 6))
    plt.plot(df_hv.mean(axis=0), label='Mean HV', color='blue')
    plt.fill_between(range(df_hv.shape[1]), df_hv.mean(axis=0)-df_hv.std(axis=0), df_hv.mean(axis=0)+df_hv.std(axis=0), alpha=0.2)
    plt.xlabel('Generation'); plt.ylabel('Hypervolume'); plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_plot.png'))

    df_p = pd.DataFrame(all_pareto); df_p.to_csv(os.path.join(OUTPUT_DIR, 'all_pareto_solutions.csv'), index=False)
    plt.figure(figsize=(10, 8))
    plt.scatter(df_p['Obj2'], df_p['Obj1'], c='gray', alpha=0.3, s=20)
    plt.xlabel('Objective 2 (Normalized Max. Story Drift Ratio)')
    plt.ylabel('Objective 1 (Normalized Cost + CO2)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'accumulated_pareto_plot.png'))

if __name__ == "__main__":
    main()
