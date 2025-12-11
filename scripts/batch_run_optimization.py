import os
import time
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

from src.config import *
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization

# ==================================================================================
# [USER CONFIGURATION] 최적 파라미터 설정 (실험 완료 후 업데이트 필요)
# ==================================================================================
OPTIMAL_PARAMS = {
    'Population_Size': 100,      # (예시값) Step 5 결과
    'Tournament_Size': 3,        # (예시값) Step 2 결과
    'Crossover_Method': 'TwoPoint', # Step 1 결과
    'Crossover_Prob': 0.9,       # (예시값) Step 3 결과
    'Mutation_Prob': 0.1,        # (예시값) Step 4 결과
    'Generations': 200           # 최종 검증용 세대 수
}

NUM_REPEATS = 30 # 통계적 검증을 위한 반복 횟수 (보통 30회 이상 권장)
OUTPUT_DIR = "Results_Statistical_Validation"

# ==================================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5

def run_validation_instance(run_id, common_data):
    """단일 검증 실행"""
    (beam_sections_df, column_sections_df, beam_sections, column_sections,
     h5_file_path, col_map, beam_map, beam_lengths, chromosome_structure,
     num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2) = common_data

    # 각 프로세스에서 파일 별도 오픈
    with h5py.File(h5_file_path, 'r') as h5_file:
        start_time = time.time()
        
        # GA 실행 (verbose=False로 조용히 실행)
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
            verbose=False
        )
        
        elapsed = time.time() - start_time
        
        # 결과 요약 추출
        best_hv = hof_stats[-1]['hypervolume']
        best_ind = min(final_hof, key=lambda x: x.fitness.values[0])
        
        # Pareto Front Solutions 추출 (Objective Values Only)
        pareto_front = []
        for ind in final_hof:
            pareto_front.append({
                'Run_ID': run_id,
                'Obj1': ind.fitness.values[0], # Cost+CO2
                'Obj2': ind.fitness.values[1], # DCR
                'Cost': ind.detailed_results.get('cost', 0),
                'CO2': ind.detailed_results.get('co2', 0)
            })
            
        # Convergence History 추출 (HV per Generation)
        hv_history = [entry['hypervolume'] for entry in hof_stats]

        return {
            'Run_ID': run_id,
            'Hypervolume': best_hv,
            'Best_Cost': best_ind.detailed_results['cost'],
            'Best_CO2': best_ind.detailed_results['co2'],
            'Best_DCR': best_ind.detailed_results['mean_strength_ratio'],
            'Elapsed_Time': elapsed,
            'HV_History': hv_history,
            'Pareto_Solutions': pareto_front
        }

def main():
    print(f"### Starting Statistical Validation ({NUM_REPEATS} runs) ###")
    print(f"Target Directory: {OUTPUT_DIR}")
    print("Parameters:", OPTIMAL_PARAMS)
    
    # 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file_path = 'pm_dataset_simple02.mat'
    
    num_locations = len(COLUMN_LOCATIONS)
    num_columns = num_locations * FLOORS
    num_beams = len(BEAM_CONNECTIONS) * FLOORS
    beam_lengths = get_beam_lengths(COLUMN_LOCATIONS, BEAM_CONNECTIONS)
    
    num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
        GROUPING_STRATEGY, num_locations, num_columns, num_beams, FLOORS, BEAM_CONNECTIONS, COLUMN_LOCATIONS
    )
    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    
    total_col_len = (len(COLUMN_LOCATIONS) * FLOORS) * H
    total_beam_len = sum(beam_lengths) * FLOORS
    fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
        column_sections_df, beam_sections_df, total_col_len, total_beam_len
    )
    
    common_data = (beam_sections_df, column_sections_df, beam_sections, column_sections,
                   h5_file_path, col_map, beam_map, beam_lengths, chromosome_structure,
                   num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2)

    # 병렬 실행
    summary_results = []
    all_hv_histories = []
    all_pareto_solutions = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_validation_instance, i+1, common_data): i for i in range(NUM_REPEATS)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_REPEATS, desc="Validation Progress"):
            try:
                res = future.result()
                # 요약 정보 저장
                summary_results.append({k: v for k, v in res.items() if k not in ['HV_History', 'Pareto_Solutions']})
                # 상세 정보 수집
                all_hv_histories.append(res['HV_History'])
                all_pareto_solutions.extend(res['Pareto_Solutions'])
            except Exception as e:
                print(f"Error in run: {e}")

    # 1. 요약 통계 저장 (CSV & Boxplot)
    df_results = pd.DataFrame(summary_results).sort_values('Run_ID')
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'statistical_summary.csv'), index=False)
    
    metrics = ['Hypervolume', 'Best_Cost', 'Best_CO2', 'Best_DCR']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, metric in enumerate(metrics):
        axes[i].boxplot(df_results[metric], patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[i].set_title(metric)
        axes[i].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_metrics.png'))

    # 2. 수렴성 그래프 (Convergence Plot)
    # 데이터 프레임 변환 (Rows: Runs, Cols: Generations)
    df_hv_hist = pd.DataFrame(all_hv_histories)
    df_hv_hist.to_csv(os.path.join(OUTPUT_DIR, 'convergence_history.csv'), index_label="Run_ID")
    
    mean_hv = df_hv_hist.mean(axis=0)
    std_hv = df_hv_hist.std(axis=0)
    generations = np.arange(len(mean_hv))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_hv, label='Mean Hypervolume', color='blue', linewidth=2)
    plt.fill_between(generations, mean_hv - std_hv, mean_hv + std_hv, color='blue', alpha=0.2, label='±1 Std. Dev.')
    plt.title(f'Convergence Stability over {NUM_REPEATS} Runs')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_plot.png'))
    
    # 3. 누적 파레토 프론트 (Accumulated Pareto Front)
    df_pareto = pd.DataFrame(all_pareto_solutions)
    df_pareto.to_csv(os.path.join(OUTPUT_DIR, 'all_pareto_solutions.csv'), index=False)
    
    plt.figure(figsize=(10, 8))
    # 모든 런의 해를 회색 점으로 표시
    plt.scatter(df_pareto['Obj2'], df_pareto['Obj1'], c='gray', alpha=0.3, s=20, label='All Runs Solutions')
    
    # (선택) 전체 해 중 Non-dominated 해를 찾아서 강조할 수도 있음 (여기선 생략하고 전체 분포만 표시)
    
    plt.title(f'Accumulated Pareto Fronts ({NUM_REPEATS} Runs)')
    plt.xlabel('Structural Conservatism (Mean DCR)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accumulated_pareto_plot.png'))

    print("\n[Artifacts Generated]")
    print(f"1. Summary CSV: {os.path.join(OUTPUT_DIR, 'statistical_summary.csv')}")
    print(f"2. Boxplot: {os.path.join(OUTPUT_DIR, 'boxplot_metrics.png')}")
    print(f"3. Convergence Plot: {os.path.join(OUTPUT_DIR, 'convergence_plot.png')}")
    print(f"4. Pareto Plot: {os.path.join(OUTPUT_DIR, 'accumulated_pareto_plot.png')}")

if __name__ == "__main__":
    main()
