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
import src.config as cfg
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps
from src.optimization import run_ga_optimization

# ==================================================================================
# [USER CONFIGURATION] 실험 실행 제어 및 이전 결과 입력
# ==================================================================================

# 1. 실행할 단계 선택 (True: 실행, False: 건너뛰기)
RUN_STEPS = {
    1: True,   # Step 1: Crossover Strategy (현재 실행)
    2: False,  # Step 2: Tournament Size
    3: False,  # Step 3: Crossover Probability
    4: False,  # Step 4: Mutation Probability
    5: False   # Step 5: Population Size
}

# 2. 이전 단계에서 결정된 최적 파라미터 (건너뛴 단계의 결과값을 여기에 입력하세요)
PREV_BEST_PARAMS = {
    'Best_Crossover': 'TwoPoint', 
    'Best_Tournament': 3,         
    'Best_CXPB': 0.9,             
    'Best_MUTPB': 0.7,            
    'Best_PopSize': 100           
}

# 3. 실험 파라미터 설정
STEP1_GEN = 50     # 교배 전략 비교 (빠른 탐색)
STEP2_GEN = 50     # 토너먼트 크기 비교
STEP3_GEN = 50     # 교배 확률 비교
STEP4_GEN = 50     # 변이 확률 비교
STEP5_GEN = 100     # 모집단 크기 비교 (조금 더 길게)

BASE_POP = 100     # 초기 기준 모집단
BASE_TOURN = 3     
BASE_CXPB = 0.8    
BASE_MUTPB = 0.2   

OUTPUT_ROOT = "Results_Param_Optimization"

# ==================================================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5

import openseespy.opensees as ops

def run_single_experiment(exp_name, pop_size, tourn_size, crossover, cxpb, mutpb, num_gen, common_data, pbar_desc=""):
    """단일 실험 수행 및 로그 간소화"""
    # [Fairness] OpenSees 상태 완전 초기화
    ops.wipe()
    
    # [Fairness] 공정한 비교를 위해 매 실험마다 난수 시드 고정
    random.seed(42)
    np.random.seed(42)

    (beam_sections_df, column_sections_df, beam_sections, column_sections,
     h5_file, col_map, beam_map, beam_lengths, chromosome_structure,
     num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2) = common_data

    # 로그 출력 포맷팅
    print(f"{pbar_desc:<50} | Pop:{pop_size:<4} Tourn:{tourn_size:<2} CX:{crossover:<8} Pc:{cxpb:<4} Pm:{mutpb:<4} ... ", end='', flush=True)

    start_time = time.time()
    
    # [수정] Example 1 전용 하중 패턴 사용
    target_patterns = cfg.LOAD_PATTERNS_4F 

    # verbose=False로 설정하여 내부 로그 억제
    final_pop, _, final_hof, hof_stats_history = run_ga_optimization(
        DL=DL_AREA_LOAD, LL=LL_AREA_LOAD,
        crossover_method=crossover, patterns_by_floor=target_patterns, h5_file=h5_file,
        num_generations=num_gen, population_size=pop_size,
        col_map=col_map, beam_map=beam_map, beam_sections=beam_sections, column_sections=column_sections,
        beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, beam_lengths=beam_lengths,
        chromosome_structure=chromosome_structure, num_columns=num_columns, num_beams=num_beams,
        fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
        fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
        tournament_size=tourn_size, cxpb=cxpb, mutpb=mutpb,
        verbose=True # <--- 핵심: 진행바 활성화 (텍스트 로그는 optimization.py에서 끔)
    )
    
    elapsed = time.time() - start_time
    final_hv = hof_stats_history[-1]['hypervolume'] if hof_stats_history else 0.0
    print(f"Done ({elapsed:.1f}s) | HV: {final_hv:.4f}") # 결과 요약 출력
    
    # 메타데이터 추가
    for entry in hof_stats_history:
        entry.update({
            'Experiment': exp_name, 'PopSize': pop_size, 'Tournament': tourn_size, 
            'Crossover': crossover, 'CXPB': cxpb, 'MUTPB': mutpb
        })
        
    return hof_stats_history, final_hof, final_pop

def analyze_and_plot(all_history_data, all_hof_data, step_name, param_key, output_dir):
    """실험 결과 분석 및 시각화 (기존 CSV 병합 기능 추가)"""
    df = pd.DataFrame(all_history_data)
    
    # 1. Hypervolume 수렴 그래프 (현재 실행된 실험만 표시)
    plt.figure(figsize=(10, 6))
    final_hvs = {}
    for label, group in df.groupby(param_key):
        plt.plot(group['gen'], group['hypervolume'], marker='o', markersize=3, label=f"{label}")
        final_hvs[label] = group.iloc[-1]['hypervolume']
    
    plt.title(f'{step_name} - Hypervolume Convergence (Current Run)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend(title=param_key)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{step_name}_HV.png'))
    plt.close()

    # [추가] 수렴 이력 데이터 저장 (그래프 재현용)
    history_csv_path = os.path.join(output_dir, f'{step_name}_HV_History.csv')
    df.to_csv(history_csv_path, index=False)
    print(f"   -> [Saved] Convergence history saved to {history_csv_path}")
    
    # 3. Pareto Front 데이터 처리 (HOF Only)
    new_hof_rows = []
    for param, inds in all_hof_data.items():
        for ind in inds:
            # [제약조건 필터링] violation > 0인 해는 제외
            if ind.detailed_results.get('violation', 0) > 0:
                continue
                
            new_hof_rows.append({
                'Parameter': param,
                'Obj1_NormCostCO2': ind.fitness.values[0],
                'Obj2_MaxDrift': ind.fitness.values[1],
                'Cost': ind.detailed_results.get('cost', 0),
                'CO2': ind.detailed_results.get('co2', 0),
                'Max_Drift': ind.detailed_results.get('max_drift_ratio', 0),
                'Hypervolume': final_hvs.get(param, 0.0)
            })
    
    df_new = pd.DataFrame(new_hof_rows)
    csv_path = os.path.join(output_dir, f'{step_name}_Pareto_Data.csv')
    
    # [병합 로직 삭제] 항상 새로운 데이터로 덮어쓰기
    df_combined = df_new

    # 저장
    df_combined.to_csv(csv_path, index=False)
    print(f"   -> [Saved] Pareto data saved to {csv_path}")

    # 4. Pareto Front 비교 그래프 (Feasible Only)
    plt.figure(figsize=(10, 8))
    
    if df_combined.empty:
        print("   -> [Warning] No HOF solutions found to plot.")
        plt.close()
        return list(all_hof_data.keys())[0] if all_hof_data else "None"

    # 병합된 데이터에서 파라미터 목록 추출 및 정렬
    all_params = sorted(df_combined['Parameter'].unique())
    
    # Best/Worst 재산정 (병합된 데이터 기준)
    param_hvs = {}
    for p in all_params:
        rows = df_combined[df_combined['Parameter'] == p]
        if not rows.empty:
            param_hvs[p] = rows.iloc[0]['Hypervolume']
            
    best_param = max(param_hvs, key=param_hvs.get) if param_hvs else all_params[0]
    worst_param = min(param_hvs, key=param_hvs.get) if param_hvs else all_params[-1]
    
    print(f"   -> [Result (Combined)] Best: {best_param} (HV={param_hvs.get(best_param, 0):.4f}), Worst: {worst_param}")

    # 파라미터 개수에 맞춰 색상 생성 (viridis 컬러맵 사용)
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_params)))
    
    for param, color in zip(all_params, colors):
        subset = df_combined[df_combined['Parameter'] == param]
        if subset.empty: continue

        fit1 = subset['Obj1_NormCostCO2']
        fit2 = subset['Obj2_MaxDrift']
        
        # 라벨 생성 (HV 포함)
        label_str = f"{param}"
        
        if param == best_param:
            # Best Param: 빨간색 별표, 크기 키움, 최상단 표시
            plt.scatter(fit2, fit1, c='red', label=f'{label_str} (Best)', s=100, marker='*', edgecolors='black', zorder=10)
        elif param == worst_param:
            # Worst Param: 회색 X표, 투명도 낮춤
            plt.scatter(fit2, fit1, c='gray', label=f'{label_str} (Worst)', s=40, marker='x', alpha=0.5, zorder=1)
        else:
            # 그 외: 컬러맵 색상, 원형
            plt.scatter(fit2, fit1, color=color, label=label_str, s=50, alpha=0.7, zorder=5)
    
    plt.title(f'{step_name} - Pareto Front Comparison (HOF Only)')
    plt.xlabel('Resilience (Max Drift Ratio)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{step_name}_Pareto_All.png'))
    plt.close()
    
    return best_param

def run_step_logic(step_num, step_title, param_list, param_name, current_best_params, common_data, gen_count):
    """각 Step의 공통 실행 로직"""
    if not RUN_STEPS[step_num]:
        fixed_val = PREV_BEST_PARAMS[f'Best_{param_name}']
        print(f"\n[Step {step_num}] {step_title} -> SKIPPED (Using fixed: {fixed_val})")
        return fixed_val

    print(f"\n" + "="*80)
    print(f"[Step {step_num}] {step_title}")
    print("="*80)
    
    history, hof_data = [], {}
    
    # 파라미터별 루프
    for i, val in enumerate(param_list):
        exp_name = f"{param_name}_{val}"
        desc = f"({i+1}/{len(param_list)}) Testing {param_name}={val}"
        
        # 현재 Step의 파라미터(val)를 적용하고, 나머지는 current_best_params 값 사용 (순차적 최적화 반영)
        # 동적으로 인자 생성
        args = {
            'pop_size': val if param_name == 'PopSize' else current_best_params.get('Best_PopSize', BASE_POP),
            'tourn_size': val if param_name == 'Tournament' else current_best_params.get('Best_Tournament', BASE_TOURN),
            'crossover': val if param_name == 'Crossover' else current_best_params.get('Best_Crossover', 'OnePoint'),
            'cxpb': val if param_name == 'CXPB' else current_best_params.get('Best_CXPB', BASE_CXPB),
            'mutpb': val if param_name == 'MUTPB' else current_best_params.get('Best_MUTPB', BASE_MUTPB)
        }
        
        hist, hof, _ = run_single_experiment(
            exp_name, args['pop_size'], args['tourn_size'], args['crossover'], args['cxpb'], args['mutpb'], 
            gen_count, common_data, pbar_desc=desc
        )
        history.extend(hist)
        hof_data[val] = hof
        
    best_val = analyze_and_plot(history, hof_data, f"Step{step_num}_{param_name}", param_name, OUTPUT_ROOT)
    return best_val

def main():
    print("### Parameter Optimization Experiment Started (Sequential & Independent Mode) ###")
    print(f"Output Directory: {OUTPUT_ROOT}")
    
    # 공통 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    try:
        # [Sync with main.py] Example 1 (4층) 환경 강제 설정
        import src.config as cfg
        cfg.FLOORS = 4
        cfg.COLUMN_LOCATIONS = cfg.COLUMN_LOCATIONS_4F
        cfg.BEAM_CONNECTIONS = cfg.BEAM_CONNECTIONS_4F
        cfg.BEAM_TRIBUTARY_WIDTHS = cfg.BEAM_TRIBUTARY_WIDTHS_4F
        cfg.GROUPING_STRATEGY = "Hybrid"

        # [수정] 강제 그룹핑 전략 설정 (Hybrid) -> cfg 사용
        # forced_grouping_strategy = "Hybrid"
        
        num_locations = len(cfg.COLUMN_LOCATIONS)
        num_columns = num_locations * cfg.FLOORS
        num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
        beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
        
        num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
            cfg.GROUPING_STRATEGY, num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
        )
        chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
        
        total_col_len = (len(cfg.COLUMN_LOCATIONS) * cfg.FLOORS) * cfg.H
        total_beam_len = sum(beam_lengths) * cfg.FLOORS
        fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
            column_sections_df, beam_sections_df, total_col_len, total_beam_len
        )
        
        common_data = (beam_sections_df, column_sections_df, beam_sections, column_sections,
                       h5_file, col_map, beam_map, beam_lengths, chromosome_structure,
                       num_columns, num_beams, fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2)
        
        # 현재까지의 최적 파라미터를 추적하는 딕셔너리 (초기값은 PREV_BEST_PARAMS로 시작)
        current_best = PREV_BEST_PARAMS.copy()

        # --- Step 1: Crossover Strategy ---
        best_cx = run_step_logic(1, "Crossover Strategy", ["OnePoint", "TwoPoint", "Uniform"], "Crossover", current_best, common_data, STEP1_GEN)
        current_best['Best_Crossover'] = best_cx

        # --- Step 2: Tournament Size ---
        best_tourn = run_step_logic(2, "Tournament Size", [2, 3, 5, 7, 9, 11], "Tournament", current_best, common_data, STEP2_GEN)
        current_best['Best_Tournament'] = best_tourn
        
        # --- Step 3: Crossover Probability ---
        best_cxpb = run_step_logic(3, "Crossover Probability", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "CXPB", current_best, common_data, STEP3_GEN)
        current_best['Best_CXPB'] = best_cxpb

        # --- Step 4: Mutation Probability ---
        # [변경] 추가 실험 (0.7 ~ 1.0)
        best_mutpb = run_step_logic(4, "Mutation Probability", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "MUTPB", current_best, common_data, STEP4_GEN)
        current_best['Best_MUTPB'] = best_mutpb

        # --- Step 5: Population Size ---
        best_pop = run_step_logic(5, "Population Size", [400], "PopSize", current_best, common_data, STEP5_GEN)
        current_best['Best_PopSize'] = best_pop

        # 최종 결과 출력
        print("\n" + "="*80)
        print("### FINAL OPTIMIZATION PARAMETERS ###")
        for k, v in current_best.items():
            print(f"{k:<20} : {v}")
        print("="*80)

    finally:
        h5_file.close()

if __name__ == "__main__":
    main()
