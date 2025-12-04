import os
import time
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.config import *
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization

# --- 실험 설정 ---
STEP1_GEN = 50     # 교배 전략 비교 (빠른 탐색)
STEP2_GEN = 50     # 토너먼트 크기 비교
STEP3_GEN = 50     # 교배 확률 비교
STEP4_GEN = 100    # 변이 확률 비교 (다양성 중요하므로 조금 더 길게)
STEP5_GEN = 200    # 모집단 크기 비교 (최종 수렴 성능)

BASE_POP = 100     # 초기 기준 모집단
BASE_TOURN = 3     # 초기 기준 토너먼트
BASE_CXPB = 0.9    # 초기 기준 교배 확률
BASE_MUTPB = 0.1   # 초기 기준 변이 확률

OUTPUT_ROOT = "Results_Param_Optimization"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 그래프 스타일 설정
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5

def run_single_experiment(exp_name, pop_size, tourn_size, crossover, cxpb, mutpb, num_gen, common_data):
    """단일 실험 수행 후 HOF 통계 및 최종 HOF 개체 리스트 반환"""
    print(f"\n>>> Running: {exp_name} (Pop:{pop_size}, Tourn:{tourn_size}, CX:{crossover}, P_c:{cxpb}, P_m:{mutpb})")
    
    (beam_sections_df, column_sections_df, beam_sections, column_sections,
     h5_file, col_map, beam_map, beam_lengths, chromosome_structure,
     num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2) = common_data

    start_time = time.time()
    _, _, final_hof, hof_stats_history = run_ga_optimization(
        DL=DL_AREA_LOAD, LL=LL_AREA_LOAD, Wx=WX_RAND, Wy=WY_RAND, Ex=EX_RAND, Ey=EY_RAND,
        crossover_method=crossover, patterns_by_floor=PATTERNS_BY_FLOOR, h5_file=h5_file,
        num_generations=num_gen, population_size=pop_size,
        col_map=col_map, beam_map=beam_map, beam_sections=beam_sections, column_sections=column_sections,
        beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, beam_lengths=beam_lengths,
        chromosome_structure=chromosome_structure, num_columns=num_columns, num_beams=num_beams,
        fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
        fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
        tournament_size=tourn_size, cxpb=cxpb, mutpb=mutpb
    )
    elapsed = time.time() - start_time
    print(f"    Done in {elapsed:.1f}s")
    
    # 메타데이터 추가
    for entry in hof_stats_history:
        entry.update({
            'Experiment': exp_name, 'PopSize': pop_size, 'Tournament': tourn_size, 
            'Crossover': crossover, 'CXPB': cxpb, 'MUTPB': mutpb
        })
        
    return hof_stats_history, final_hof

def analyze_and_plot(all_history_data, all_hof_data, step_name, param_key, output_dir):
    """실험 결과 분석: HV 그래프 및 Pareto Front 비교"""
    df = pd.DataFrame(all_history_data)
    
    # 1. Hypervolume 수렴 그래프
    plt.figure(figsize=(10, 6))
    final_hvs = {}
    
    for label, group in df.groupby(param_key):
        plt.plot(group['gen'], group['hypervolume'], marker='o', markersize=3, label=f"{label}")
        final_hvs[label] = group.iloc[-1]['hypervolume']
    
    plt.title(f'{step_name} - Hypervolume Convergence')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend(title=param_key)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{step_name}_HV.png'))
    plt.close()
    
    # 2. 최적 파라미터 선정
    # Series나 numpy 타입 등을 안전하게 처리하기 위해 str로 변환하여 비교하거나 직접 값 사용
    best_param = max(final_hvs, key=final_hvs.get)
    worst_param = min(final_hvs, key=final_hvs.get)
    print(f"\n[{step_name} Result]")
    print(f"  - Best {param_key}: {best_param} (HV: {final_hvs[best_param]:.4f})")
    print(f"  - Worst {param_key}: {worst_param} (HV: {final_hvs[worst_param]:.4f})")

    # 3. Pareto Front 비교 그래프 (Best vs Worst)
    plt.figure(figsize=(8, 7))
    
    # Best Param Plot
    best_hof = all_hof_data[best_param]
    fit1_best = [ind.fitness.values[0] for ind in best_hof] # Norm Cost+CO2
    fit2_best = [ind.fitness.values[1] for ind in best_hof] # Mean DCR
    plt.scatter(fit2_best, fit1_best, c='blue', label=f'Best ({best_param})', alpha=0.7, s=50)

    # Worst Param Plot
    worst_hof = all_hof_data[worst_param]
    fit1_worst = [ind.fitness.values[0] for ind in worst_hof]
    fit2_worst = [ind.fitness.values[1] for ind in worst_hof]
    plt.scatter(fit2_worst, fit1_worst, c='red', label=f'Worst ({worst_param})', alpha=0.4, marker='x', s=40)
    
    plt.title(f'{step_name} - Pareto Front Comparison (Best vs Worst)')
    plt.xlabel('Structural Conservatism (Mean DCR)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{step_name}_Pareto_BestVsWorst.png'))
    plt.close()
    
    return best_param

def main():
    print("### Starting Comprehensive Parameter Optimization Experiment ###")
    
    # 공통 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
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
                   h5_file, col_map, beam_map, beam_lengths, chromosome_structure,
                   num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2)
    
    final_summary = {}

    try:
        # --- Step 1: 교배 전략 (Crossover Strategy) ---
        print("\n" + "="*60 + "\nStep 1: Crossover Strategy\n" + "="*60)
        crossovers = ["OnePoint", "TwoPoint", "Uniform"]
        history_s1, hof_s1 = [], {}
        for cx in crossovers:
            hist, hof = run_single_experiment(f"CX_{cx}", BASE_POP, BASE_TOURN, cx, BASE_CXPB, BASE_MUTPB, STEP1_GEN, common_data)
            history_s1.extend(hist)
            hof_s1[cx] = hof
        best_cx = analyze_and_plot(history_s1, hof_s1, "Step1_Crossover", "Crossover", OUTPUT_ROOT)
        final_summary['Best_Crossover'] = best_cx

        # --- Step 2: 토너먼트 크기 (Tournament Size) ---
        print("\n" + "="*60 + f"\nStep 2: Tournament Size (Fixed CX: {best_cx})\n" + "="*60)
        tournaments = [2, 3, 5, 7, 9, 11]
        history_s2, hof_s2 = [], {}
        for t in tournaments:
            hist, hof = run_single_experiment(f"Tourn_{t}", BASE_POP, t, best_cx, BASE_CXPB, BASE_MUTPB, STEP2_GEN, common_data)
            history_s2.extend(hist)
            hof_s2[t] = hof
        best_tourn = analyze_and_plot(history_s2, hof_s2, "Step2_Tournament", "Tournament", OUTPUT_ROOT)
        final_summary['Best_Tournament'] = best_tourn
        
        # --- Step 3: 교배 확률 (Crossover Probability) ---
        print("\n" + "="*60 + f"\nStep 3: Crossover Prob (Fixed Tourn: {best_tourn})\n" + "="*60)
        cx_probs = [0.7, 0.8, 0.9, 1.0]
        history_s3, hof_s3 = [], {}
        for cp in cx_probs:
            hist, hof = run_single_experiment(f"CXPB_{cp}", BASE_POP, best_tourn, best_cx, cp, BASE_MUTPB, STEP3_GEN, common_data)
            history_s3.extend(hist)
            hof_s3[cp] = hof
        best_cxpb = analyze_and_plot(history_s3, hof_s3, "Step3_CXPB", "CXPB", OUTPUT_ROOT)
        final_summary['Best_CXPB'] = best_cxpb

        # --- Step 4: 변이 확률 (Mutation Probability) ---
        print("\n" + "="*60 + f"\nStep 4: Mutation Prob (Fixed CXPB: {best_cxpb})\n" + "="*60)
        mut_probs = [0.05, 0.1, 0.2, 0.3]
        history_s4, hof_s4 = [], {}
        for mp in mut_probs:
            hist, hof = run_single_experiment(f"MUTPB_{mp}", BASE_POP, best_tourn, best_cx, best_cxpb, mp, STEP4_GEN, common_data)
            history_s4.extend(hist)
            hof_s4[mp] = hof
        best_mutpb = analyze_and_plot(history_s4, hof_s4, "Step4_MUTPB", "MUTPB", OUTPUT_ROOT)
        final_summary['Best_MUTPB'] = best_mutpb

        # --- Step 5: 모집단 크기 (Population Size) ---
        print("\n" + "="*60 + f"\nStep 5: Population Size (Fixed All Params)\n" + "="*60)
        pop_sizes = list(range(100, 1001, 100))
        history_s5, hof_s5 = [], {}
        for pop in pop_sizes:
            hist, hof = run_single_experiment(f"Pop_{pop}", pop, best_tourn, best_cx, best_cxpb, best_mutpb, STEP5_GEN, common_data)
            history_s5.extend(hist)
            hof_s5[pop] = hof
        best_pop = analyze_and_plot(history_s5, hof_s5, "Step5_PopSize", "PopSize", OUTPUT_ROOT)
        final_summary['Best_PopSize'] = best_pop

        # 최종 결과 저장
        df_all = pd.DataFrame(history_s1 + history_s2 + history_s3 + history_s4 + history_s5)
        df_all.to_csv(os.path.join(OUTPUT_ROOT, "all_optimization_steps_results.csv"), index=False)

        print("\n" + "="*60)
        print("### FINAL OPTIMIZATION PARAMETERS ###")
        for k, v in final_summary.items():
            print(f"{k:<20} : {v}")
        print("="*60)

    finally:
        h5_file.close()

if __name__ == "__main__":
    main()