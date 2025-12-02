import pandas as pd
import numpy as np
import h5py
import math
import os
import sys
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.grouping import get_grouping_maps
from src.modeling import build_model_for_section
from src.evaluation import evaluate
from src.optimization import run_ga_optimization
from src.visualization import plot_Structure, visualize_load_patterns
from src.utils import get_precalculated_strength, load_pm_data_for_column, get_pm_capacity_from_df, calculate_normalization_constants


def main():
    import argparse

    parser = argparse.ArgumentParser(description='NSGA-II Optimization for RC Frames')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--pop', type=int, default=600, help='Population size')
    parser.add_argument('--gen', type=int, default=200, help='Number of generations')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode (no user input)')
    # 하중 인자 추가 (기본값은 기존 코드와 동일하게 유지)
    parser.add_argument('--ex', type=float, default=40, help='Seismic Load X (kN)')
    parser.add_argument('--ey', type=float, default=40, help='Seismic Load Y (kN)')
    parser.add_argument('--wx', type=float, default=35, help='Wind Load X (kN)')
    parser.add_argument('--wy', type=float, default=38, help='Wind Load Y (kN)')
    args = parser.parse_args()

    # =================================================================
    # ===                  라이브러리 임포트                          ===
    # =================================================================
    import openseespy.opensees as ops
    from deap import base, creator, tools, algorithms
    import time # 시간 측정을 위해 추가

    start_time = time.time() # 시작 시간 기록

    # --- ✅ 1. Matplotlib 전역 글꼴 설정: Times New Roman ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12  # 기본 글꼴 크기 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random Seed set to: {args.seed}")

    # =================================================================
    # ===              1. 최적화 및 모델링 주요 설정                  ===
    # =================================================================

    # --- 1.1. 그룹핑 및 교배 전략 선택 ---
    GROUPING_STRATEGY = "Hybrid"
    CROSSOVER_STRATEGY = "OnePoint"  # "OnePoint", "TwoPoint", "Uniform"

    # --- 1.2. 건물 기본 정보 ---
    floors = 4
    H = 4.0

    # --- 1.3. 건물 형상 정보 ---
    column_locations = [(0, 0), (5, 0), (10, 0), (15, 0),
                        (0, 6), (5, 6), (10, 6), (15, 6),
                        (0, 10), (5, 10), (10, 10), (15, 10),
                        (5, 15), (10, 15), (15, 15)]
    beam_connections = [(0, 1), (1, 2), (2, 3),
                        (4, 5), (5, 6), (6, 7),
                        (8, 9), (9, 10), (10, 11),
                        (12, 13), (13, 14),
                        (0, 4), (4, 8),
                        (1, 5), (5, 9), (9, 12),
                        (2, 6), (6, 10), (10, 13),
                        (3, 7), (7, 11), (11, 14)]

    DL_rand=25
    LL_rand=20
    # 인자로 받은 하중 값 적용
    Wx_rand = args.wx
    Wy_rand = args.wy
    Ex_rand = args.ex
    Ey_rand = args.ey

    # --- 1.5. 유전 알고리즘 파라미터 ---
    POPULATION_SIZE = args.pop
    NUM_GENERATIONS = args.gen
    CXPB, MUTPB = 0.8, 0.2

    # --- 1.6. 부재 단면 정보 로드 ---
    beam_sections_df = pd.read_csv("../data/beam_sections_simple02.csv")
    column_sections_df = pd.read_csv("../data/column_sections_simple02.csv")
    h5_file = h5py.File('../data/pm_dataset_simple02.mat', 'r') # 데이터로드용

    load_combinations = [
        # 1. 1.4D
        ("ACI-1", {"DL": 1.4, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 0, "Ey": 0}),
        # 2. 1.2D + 1.6L
        ("ACI-2", {"DL": 1.2, "LL": 1.6, "Wx": 0, "Wy": 0, "Ex": 0, "Ey": 0}),
        # 3. 1.2D + 1.0L + 1.0W
        ("ACI-3", {"DL": 1.2, "LL": 1.0, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-4", {"DL": 1.2, "LL": 1.0, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-5", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
        ("ACI-6", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
        # 4. 1.2D + 1.0L + 1.0E (직교 효과 포함)
        ("ACI-7", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": 0.3}),
        ("ACI-8", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": -0.3}),
        ("ACI-9", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": 0.3}),
        ("ACI-10", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": -0.3}),
        ("ACI-11", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": 1.0}),
        ("ACI-12", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": -1.0}),
        ("ACI-13", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": 1.0}),
        ("ACI-14", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": -1.0}),
        # 5. 0.9D + 1.0W
        ("ACI-15", {"DL": 0.9, "LL": 0, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-16", {"DL": 0.9, "LL": 0, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-17", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
        ("ACI-18", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
        # 6. 0.9D + 1.0E (직교 효과 포함)
        ("ACI-19", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": 0.3}),
        ("ACI-20", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": -0.3}),
        ("ACI-21", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": 0.3}),
        ("ACI-22", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": -0.3}),
        ("ACI-23", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": 1.0}),
        ("ACI-24", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": -1.0}),
        ("ACI-25", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": 1.0}),
        ("ACI-26", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": -1.0}),
        # --- ✅ ASCE 7 사용성 검토용 하중조합 (신규 추가 및 수정) ---
        # 1. 풍하중(W) 변위 검토용: D + 0.5L + W
        ("ASCE-S-W1", {"DL": 1.0, "LL": 0.5, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-W2", {"DL": 1.0, "LL": 0.5, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-W3", {"DL": 1.0, "LL": 0.5, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-W4", {"DL": 1.0, "LL": 0.5, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
        # 2. 지진하중(E) 층간변위 검토용: D + L + 0.7E (직교 효과 포함)
        ("ASCE-S-E1", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.7, "Ey": 0.21}),
        ("ASCE-S-E2", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.7, "Ey": -0.21}),
        ("ASCE-S-E3", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.7, "Ey": 0.21}),
        ("ASCE-S-E4", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.7, "Ey": -0.21}),
        ("ASCE-S-E5", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.21, "Ey": 0.7}),
        ("ASCE-S-E6", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.21, "Ey": -0.7}),
        ("ASCE-S-E7", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.21, "Ey": 0.7}),
        ("ASCE-S-E8", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.21, "Ey": -0.7}),
    ]

    # --- 2.1. 그룹핑 전략 실행 ---
    col_map, beam_map, num_col_groups, num_beam_groups = get_grouping_maps(GROUPING_STRATEGY, floors, column_locations, beam_connections)
    num_columns = len(column_locations) * floors
    num_beams = len(beam_connections) * floors
    
    # --- 2.2. 유전 정보(Chromosome) 구조 정의 ---
    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    chromosome_len = sum(chromosome_structure.values())
    beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
    column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
    beam_lengths = [math.sqrt((column_locations[p2][0] - column_locations[p1][0])**2 + (column_locations[p2][1] - column_locations[p1][1])**2) for p1, p2 in beam_connections]

    # --- 데이터 기반 고정 스케일 자동 계산 (최종안 - 거푸집 비용 포함) ---
    total_column_length = (len(column_locations) * floors) * H
    total_beam_length = sum(beam_lengths) * floors
    
    FIXED_MIN_COST, FIXED_RANGE_COST, FIXED_MIN_CO2, FIXED_RANGE_CO2 = calculate_normalization_constants(
        column_sections_df, beam_sections_df, total_column_length, total_beam_length
    )

    # =================================================================
    # ===                    5. 메인 실행 블록                        ===
    # =================================================================
    if args.output:
        output_folder = args.output
    else:
        output_folder = f"../results/Results_main_하중증가,모집단{POPULATION_SIZE}_06_200세대"
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n결과는 '{output_folder}' 폴더에 저장됩니다.")
    print("\n" + "="*90)
    print("### 최적화 실행 조건 요약 ###")
    print(f"- 그룹핑 전략: {GROUPING_STRATEGY}, 교배 전략: {CROSSOVER_STRATEGY}")
    print(f"- 인구: {POPULATION_SIZE}, 세대: {NUM_GENERATIONS}, 교배/변이율: {CXPB}/{MUTPB}")
    print("\n[하중 기본값]")
    print(f"- 추가 고정하중(DL): {DL_rand} kN/m, 활하중(LL): {LL_rand} kN/m")
    print(f"- 지진하중(Ex, Ey): {Ex_rand}, {Ey_rand} kN | 풍하중(Wx, Wy): {Wx_rand}, {Wy_rand} kN")
    print("- 자중: 단면적에 따라 자동 계산")
    
    patterns_by_floor = {
        1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20],
        2: [1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        3: [3, 4, 6, 7, 9, 12, 14, 15, 17, 18],
        4: [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 19, 20]
    }


    print("\n[부분 재하 상세 (OpenSees 보 Element ID)]")
    beams_per_floor = len(beam_connections)
    num_locations = len(column_locations)
    for floor_num, conn_indices in patterns_by_floor.items():
        if conn_indices:
            start_eid_for_floor = num_columns + (floor_num - 1) * beams_per_floor
            loaded_beam_eids = [start_eid_for_floor + conn_idx + 1 for conn_idx in conn_indices]
            print(f"- {floor_num}층 적용 보 ID: {sorted(loaded_beam_eids)}")
        else:
            print(f"- {floor_num}층: 부분 재하 없음")

    print("\n[횡력의 절점 하중 상세 (역삼각형 분포, kN)]")
    lateral_force_dist_sum = sum(range(1, floors + 1))
    print(f"  {'Floor':<5} | {'Nodal Fx (Seismic)':<20} | {'Nodal Fy (Seismic)':<20} | {'Nodal Fx (Wind)':<17} | {'Nodal Fy (Wind)':<17}")
    print("-" * 87)
    for k in range(1, floors + 1):
        floor_multiplier = k / lateral_force_dist_sum
        seismic_fx = (Ex_rand * floor_multiplier) / num_locations
        seismic_fy = (Ey_rand * floor_multiplier) / num_locations
        wind_fx = (Wx_rand * floor_multiplier) / num_locations
        wind_fy = (Wy_rand * floor_multiplier) / num_locations
        print(f"  {k:<5} | {seismic_fx:<20.3f} | {seismic_fy:<20.3f} | {wind_fx:<17.3f} | {wind_fy:<17.3f}")
    print("="*90)

    # --- 최초 최적화 실행 ---
    population, logbook, pareto_front, hof_stats_history = run_ga_optimization(
        chromosome_structure, DL_rand, LL_rand, Wx_rand, Wy_rand, Ex_rand, Ey_rand, 
        CROSSOVER_STRATEGY, patterns_by_floor, h5_file,
        NUM_GENERATIONS, POPULATION_SIZE, CXPB, MUTPB,
        beam_lengths, col_map, beam_map, num_columns, floors, H, column_locations, beam_connections,
        column_sections_df, beam_sections_df, FIXED_MIN_COST, FIXED_RANGE_COST, FIXED_MIN_CO2, FIXED_RANGE_CO2,
        load_combinations, column_sections, beam_sections,
        initial_pop=None, start_gen=0, logbook=None, hof=None, hof_stats_history=None
    )
    
    # --- 추가 세대 진행 여부 확인 루프 (Batch 모드에서는 건너뜀) ---
    while not args.batch:
        try:
            last_gen_num = logbook.select("gen")[-1]
            more_gens_str = input(f"\n현재 {last_gen_num} 세대까지 완료. 추가로 진행할 세대 수를 입력하세요 (종료하려면 Enter): ")
            
            if not more_gens_str.strip():
                print("최적화를 종료합니다.")
                break
            
            additional_gens = int(more_gens_str)
            if additional_gens <= 0:
                print("0보다 큰 정수를 입력해야 합니다.")
                continue

            population, logbook, pareto_front, hof_stats_history = run_ga_optimization(
                chromosome_structure, DL_rand, LL_rand, Wx_rand, Wy_rand, Ex_rand, Ey_rand,
                CROSSOVER_STRATEGY, patterns_by_floor, h5_file,
                additional_gens, POPULATION_SIZE, CXPB, MUTPB,
                beam_lengths, col_map, beam_map, num_columns, floors, H, column_locations, beam_connections,
                column_sections_df, beam_sections_df, FIXED_MIN_COST, FIXED_RANGE_COST, FIXED_MIN_CO2, FIXED_RANGE_CO2,
                load_combinations, column_sections, beam_sections,
                initial_pop=population,
                start_gen=last_gen_num,
                logbook=logbook,
                hof=pareto_front,
                hof_stats_history=hof_stats_history
            )

        except (ValueError):
            print("잘못된 입력입니다. 숫자를 입력하거나 Enter를 눌러 종료하세요.")
        except (EOFError, KeyboardInterrupt):
            print("\n사용자에 의해 중단되었습니다. 최적화를 종료합니다.")
            break
    
    print("\n" + "="*80)
    print("### 최종 최적화 완료 ###")
    
    valid_solutions = []
    for ind in pareto_front:
        # 유효해 판별 기준을 'violation' 값으로 통일
        if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
            valid_solutions.append(ind)
            
    print(f"총 {len(valid_solutions)}개의 유효한 파레토 최적해를 찾았습니다.")
    print("="*80)

    # =================================================================
    # ===             공통: 최적화 로그 및 수렴 데이터 저장             ===
    # =================================================================
    print("\n\n" + "="*90)
    print(f"### 최적화 결과 데이터를 '{output_folder}' 폴더에 저장합니다. ###")

    # 1. 최적화 과정 로그 (optimization_log.csv) - 무조건 저장
    log_df = pd.DataFrame(logbook)
    log_df = log_df.drop(columns=[col for col in log_df.columns if 'sep' in str(col)], errors='ignore')
    log_df.to_csv(os.path.join(output_folder, "optimization_log.csv"), index=False)
    print("- 'optimization_log.csv' 저장 완료.")

    # 2. Hall of Fame 수렴도 데이터 저장 (hof_convergence.csv) - 무조건 저장
    if hof_stats_history:
        hof_df = pd.DataFrame(hof_stats_history)
        hof_df.to_csv(os.path.join(output_folder, "hof_convergence.csv"), index=False)
        print("- 'hof_convergence.csv' 저장 완료.")

        # 2.3: Hall of Fame 수렴도 그래프 생성
        fig_conv, ax_conv = plt.subplots(figsize=(8, 7))
        gen_hof = [s['gen'] for s in hof_stats_history]
        hof_best_obj1 = [s['best_obj1'] for s in hof_stats_history]
        hof_best_obj2 = [s['best_obj2'] for s in hof_stats_history]
        color1 = 'tab:blue'
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Fitness1 (Normalized Cost+CO2)", color=color1)
        ax_conv.plot(gen_hof, hof_best_obj1, color=color1, marker='o', linestyle='-', label="Best Fitness1")
        ax_conv.tick_params(axis='y', labelcolor=color1)
        ax_conv.grid(True)
        ax_conv_twin = ax_conv.twinx()
        color2 = 'tab:red'
        ax_conv_twin.set_ylabel("Fitness2 (Mean DCR)", color=color2)
        ax_conv_twin.plot(gen_hof, hof_best_obj2, color=color2, marker='s', linestyle='-', label="Best Fitness2")
        ax_conv_twin.tick_params(axis='y', labelcolor=color2)
        ax_conv.set_title('Convergence of the Hall of Fame', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "analysis_convergence_hof.png"))
        print("- 'analysis_convergence_hof.png' 저장 완료.")

    # 2.4: 유효해 비율 수렴도 그래프 - 무조건 저장
    fig_valid, ax_valid = plt.subplots(figsize=(8, 7))
    gen = logbook.select("gen")
    valid_ratios_log = logbook.select("valid_ratio")
    ax_valid.plot(gen, valid_ratios_log, marker='o', linestyle='-', color='g')
    ax_valid.set_title("Ratio of Feasible Solutions per Generation", fontsize=14)
    ax_valid.set_xlabel("Generation")
    ax_valid.set_ylabel("Feasible Solutions (%)")
    ax_valid.set_ylim(0, 105)
    ax_valid.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "analysis_feasible_ratio.png"))
    print("- 'analysis_feasible_ratio.png' 저장 완료.")

    all_results = []
    if not valid_solutions:
        print("\n[Warning] 모든 제약조건을 만족하는 해를 찾지 못했습니다.")
    else:
        # 정렬 기준을 첫 번째 목표값(피트니스)으로 변경
        sorted_pareto = sorted(valid_solutions, key=lambda ind: ind.fitness.values[0])
        for i, ind in enumerate(sorted_pareto):
            result_dict = ind.detailed_results
            result_dict['ID'] = f"Sol #{i+1}"
            result_dict['individual'] = ind
            result_dict['ind_object'] = ind 
            all_results.append(result_dict)

        print("\n### 파레토 최적해 요약 ###")
        summary_data_list = []
        for r in all_results:
            ind_obj = r['ind_object']
            summary_data_list.append({
                'ID': r['ID'],
                'Cost': f"{r['cost']:,.0f}",
                'CO2': f"{r['co2']:,.0f}",
                'Mean_DCR': f"{r['mean_strength_ratio']:.4f}",
                'Fit1(NormCost+CO2)': f"{ind_obj.fitness.values[0]:.4f}",
                'Fit2(Mean_DCR)': f"{ind_obj.fitness.values[1]:.4f}"
            })
        headers = list(summary_data_list[0].keys())
        col_widths = {'ID': 8, 'Cost': 14, 'CO2': 14, 'Mean_DCR': 12, 'Fit1(NormCost+CO2)': 20, 'Fit2(Mean_DCR)': 18}
        header_string = ""; 
        for h in headers: header_string += f"{h:<{col_widths[h]}} "
        print(header_string); print("-" * (sum(col_widths.values()) + len(col_widths)))
        for row_data in summary_data_list:
            row_string = ""
            for h in headers: row_string += f"{row_data[h]:<{col_widths[h]}} "
            print(row_string)
    
    if all_results:
        # =================================================================
        # ===             추가: 최적화 결과 데이터 CSV 저장             ===
        # =================================================================
        
        # 3. 파레토 최적해 요약 (pareto_summary.csv)
        summary_df = pd.DataFrame(summary_data_list)
        summary_df.to_csv(os.path.join(output_folder, "pareto_summary.csv"), index=False)
        print("- 'pareto_summary.csv' 저장 완료.")

        # 4. 최적해별 설계 변수 (design_variables.csv)
        design_vars_data = []
        for r in all_results:
            ind = r['individual']
            len_col_sec, len_col_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
            
            row_data = {'Solution_ID': r['ID']}
            for i in range(len_col_sec):
                row_data[f'col_sec_grp_{i}'] = ind[i]
            for i in range(len_col_rot):
                row_data[f'col_rot_grp_{i}'] = ind[len_col_sec + i]
            for i in range(chromosome_structure['beam_sec']):
                row_data[f'beam_sec_grp_{i}'] = ind[len_col_sec + len_col_rot + i]
            design_vars_data.append(row_data)
            
        design_vars_df = pd.DataFrame(design_vars_data)
        design_vars_df.to_csv(os.path.join(output_folder, "design_variables.csv"), index=False)
        print("- 'design_variables.csv' 저장 완료.")

        # 5. 최적해별 변위/변형 검토 결과 (displacement_checks_all.csv)
        displacement_data = []
        for r in all_results:
            sol_id = r['ID']
            if drifts_x := r.get('story_drifts_x'):
                for i, val in enumerate(drifts_x): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Seismic_Drift_X', 'Story': i + 1, 'Value': val})
            if drifts_y := r.get('story_drifts_y'):
                for i, val in enumerate(drifts_y): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Seismic_Drift_Y', 'Story': i + 1, 'Value': val})
            if disps_x := r.get('wind_displacements_x'):
                for i, val in enumerate(disps_x): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Wind_Disp_X', 'Story': i + 1, 'Value': val})
            if disps_y := r.get('wind_displacements_y'):
                for i, val in enumerate(disps_y): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Wind_Disp_Y', 'Story': i + 1, 'Value': val})

        if displacement_data:
            displacement_df = pd.DataFrame(displacement_data); displacement_df.to_csv(os.path.join(output_folder, "displacement_checks_all.csv"), index=False)
            print("- 'displacement_checks_all.csv' 저장 완료.")

        # 6. 최적해별 부재 DCR (dcr_by_element.csv)
        dcr_data = []
        for r in all_results:
            sol_id = r['ID']; ratios = r.get('strength_ratios', [])
            for i, ratio in enumerate(ratios):
                elem_id = i + 1; elem_type = 'Column' if elem_id <= num_columns else 'Beam'
                dcr_data.append({'Solution_ID': sol_id, 'Element_ID': elem_id, 'ElementType': elem_type, 'DCR': ratio})
        if dcr_data:
            dcr_df = pd.DataFrame(dcr_data); dcr_df.to_csv(os.path.join(output_folder, "dcr_by_element.csv"), index=False)
            print("- 'dcr_by_element.csv' 저장 완료.")
        print("="*90)

        print("\n\n### 파레토 최적해별 부재 설계 내역 ###")
        for result in all_results:
            print(f"\n--- {result['ID']} ---")
            ind = result['individual']; ratios = result['strength_ratios']; len_col_sec = chromosome_structure['col_sec']; len_col_rot = chromosome_structure['col_rot']
            col_indices = ind[:len_col_sec]; col_rotations = ind[len_col_sec : len_col_sec + len_col_rot]; beam_indices = ind[len_col_sec + len_col_rot :]
            member_data = []
            for i in range(num_columns):
                group_idx = col_map[i + 1]; sec_idx = col_indices[group_idx]; rot_flag = col_rotations[group_idx]; section_row = column_sections_df.iloc[sec_idx]
                member_data.append({'Type': 'Column', 'Elem. ID': i + 1, 'Sec. ID': sec_idx, 'Group':group_idx, 'Section': f"{int(section_row['b'])}x{int(section_row['h'])}",'Rot': rot_flag, 'DCR': ratios[i]})
            for i in range(num_beams):
                group_idx = beam_map[num_columns + i + 1]; sec_idx = beam_indices[group_idx]; section_row = beam_sections_df.iloc[sec_idx]
                member_data.append({'Type': 'Beam', 'Elem. ID': num_columns + i + 1, 'Sec. ID': sec_idx, 'Group':group_idx, 'Section': f"{int(section_row['b'])}x{int(section_row['h'])}",'Rot': '-', 'DCR': ratios[num_columns + i]})
            member_df = pd.DataFrame(member_data); member_df['DCR'] = member_df['DCR'].map('{:.3f}'.format); print(member_df.to_string(index=False))
        
        print("\n\n" + "="*90)
        print("### 파레토 최적해별 변위 검토 결과 ###")
        for result in all_results:
            print(f"\n--- {result['ID']} ---")
            print("\n[지진하중 층간 변위]")
            print(f"{ 'Floor':<10} {'Drift-X (m)':<15} {'Drift-Y (m)':<15} {'Drift Ratio-X':<15} {'Drift Ratio-Y':<15} {'Result'}")
            allowable_drift_ratio = 0.015
            drifts_x_ratio = result.get('story_drifts_x', []); drifts_y_ratio = result.get('story_drifts_y', [])
            for i in range(floors):
                drift_x_r = drifts_x_ratio[i] if i < len(drifts_x_ratio) else 0.0
                drift_y_r = drifts_y_ratio[i] if i < len(drifts_y_ratio) else 0.0
                drift_x_m = drift_x_r * H; drift_y_m = drift_y_r * H
                status = "OK" if drift_x_r <= allowable_drift_ratio and drift_y_r <= allowable_drift_ratio else "NG"
                print(f"{ 'Story ' + str(i+1):<10} {drift_x_m:<15.5f} {drift_y_m:<15.5f} {drift_x_r:<15.5f} {drift_y_r:<15.5f} {status}")

            print("\n[풍하중 층별 변위]")
            print(f"{ 'Floor':<10} {'Disp-X (m)':<15} {'Disp-Y (m)':<15} {'Allowable (m)':<15} {'Result'}")
            disps_x = result.get('wind_displacements_x', []); disps_y = result.get('wind_displacements_y', [])
            for i in range(floors):
                disp_x = disps_x[i] if i < len(disps_x) else 0.0; disp_y = disps_y[i] if i < len(disps_y) else 0.0
                allowable_disp = ((i + 1) * H) / 400.0
                status = "OK" if disp_x <= allowable_disp and disp_y <= allowable_disp else "NG"
                print(f"{ 'Story ' + str(i+1):<10} {disp_x:<15.5f} {disp_y:<15.5f} {allowable_disp:<15.5f} {status}")
        
        print("\n\n" + "="*80)
        print("### 파레토 최적해별 구조물 총 자중 ###")
        for result in all_results:
            print(f"\n--- {result['ID']} ---")
            ind = result['individual']
            len_col_sec = chromosome_structure['col_sec']
            len_col_rot = chromosome_structure['col_rot']
            col_indices = ind[:len_col_sec]
            beam_indices = ind[len_col_sec + len_col_rot:]

            total_sw_kn = 0
            # 기둥 자중 계산
            for i in range(num_columns):
                group_idx = col_map[i + 1]
                sec_idx = col_indices[group_idx]
                b, h = column_sections[sec_idx]  # 단면 크기 (m)
                unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']  # 단위중량 (kN/m^3)
                total_sw_kn += b * h * H * unit_weight
            # 보 자중 계산
            for k in range(floors):
                for i in range(len(beam_connections)):
                    abs_beam_idx = k * len(beam_connections) + i
                    group_idx = beam_map[num_columns + abs_beam_idx + 1]
                    sec_idx = beam_indices[group_idx]
                    b, h = beam_sections[sec_idx]  # 단면 크기 (m)
                    unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']  # 단위중량 (kN/m^3)
                    total_sw_kn += b * h * beam_lengths[i] * unit_weight
            
            # kN을 ton으로 변환 (g = 9.81 m/s^2 기준)
            total_sw_tons = total_sw_kn / 9.81
            print(f"- 구조물 총 자중: {total_sw_kn:,.2f} kN ({total_sw_tons:,.2f} tons)")
        print("="*80)


        # =================================================================
        # ===             추가: 상세 부재력 및 성능 분석 출력 (최종판)     ===
        # =================================================================
        print("\n\n" + "="*90)
        print("### 파레토 최적해별 상세 부재력 및 성능 분석 (지배 부재력 포함) ###")

        

        for result in all_results:
            print(f"\n--- {result['ID']} (Cost: {result['cost']:,.0f}, CO2: {result['co2']:,.0f}) ---")
            
            forces_df = result.get('forces_df')
            if forces_df is None or forces_df.empty:
                print("부재력 데이터를 찾을 수 없습니다.")
                continue

            ind = result['individual']
            len_col_sec, len_col_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
            col_indices, col_rotations = ind[:len_col_sec], ind[len_col_sec:len_col_sec + len_col_rot]
            beam_indices = ind[len_col_sec + len_col_rot:]
            
            column_elem_ids, beam_elem_ids, _, _ = build_model_for_section(floors, H, column_locations, beam_connections, col_indices, col_rotations, beam_indices, col_map, beam_map, column_sections, beam_sections, num_columns)
            
            output_data_long = []
            
            # 1. 모든 부재력 성분에 대한 DCR 계산 및 데이터 수집
            for _, row in forces_df.iterrows():
                elem_id = int(row['ElementID'])
                elem_type = row['ElementType']
                
                force_components_all = ['Axial (kN)', 'Moment-y (kNm)', 'Moment-z (kNm)', 'Shear-y (kN)', 'Shear-z (kN)', 'Torsion (kNm)']

                Pn_actual_z, Mn_actual_z = 0, 0
                Pn_actual_y, Mn_actual_y = 0, 0
                rotation_flag = '-' # 보(Beam)의 경우 회전이 없으므로 기본값 설정
                if elem_type == 'Column':
                    abs_col_idx = column_elem_ids.index(elem_id)
                    group_idx = col_map[abs_col_idx + 1]
                    sec_idx = col_indices[group_idx]
                    rotation_flag = col_rotations[group_idx] # 기둥의 회전 정보 가져오기
                    section_info = column_sections_df.iloc[sec_idx]
                    section_name = f"{int(section_info['b'])}x{int(section_info['h'])}"
                    strengths = get_precalculated_strength('Column', sec_idx, column_sections_df, beam_sections_df)
                    pm_df = load_pm_data_for_column(h5_file, sec_idx)

                    # 현재 부재력 수요(demand)
                    P_u = row['Axial (kN)']
                    M_u_y = row['Moment-z (kNm)'] # 약축 모멘트
                    M_u_z = row['Moment-y (kNm)'] # 강축 모멘트

                    # 상호작용을 고려한 실제 공칭강도(Pn, Mn) 계산
                    Pn_actual_z, Mn_actual_z = get_pm_capacity_from_df(abs(P_u)/(abs(M_u_z)+1e-9), pm_df, axis='z')
                    Pn_actual_y, Mn_actual_y = get_pm_capacity_from_df(abs(P_u)/(abs(M_u_y)+1e-9), pm_df, axis='y')
                # --- 수정 끝 ---
                else: # Beam
                    abs_beam_idx = beam_elem_ids.index(elem_id)
                    group_idx = beam_map[num_columns + abs_beam_idx + 1]
                    sec_idx = beam_indices[group_idx]
                    section_info = beam_sections_df.iloc[sec_idx]
                    section_name = f"{int(section_info['b'])}x{int(section_info['h'])}"
                    strengths = get_precalculated_strength('Beam', sec_idx, column_sections_df, beam_sections_df)
                
                for force_name in force_components_all:
                    demand = row[force_name]
                    combo = row.get(f'{force_name}_Combo', 'N/A')
                    capacity, dcr_val = 0, 0

                    # Capacity 및 DCR 계산
                    if elem_type == 'Column':
                        if force_name == 'Axial (kN)':
                            # --- 제안: 축력을 강축(z)/약축(y) 상호작용에 따라 분리하여 평가 ---
                            # 1. z축(강축) 모멘트와의 상호작용 고려
                            capacity_z = Pn_actual_z
                            dcr_z = abs(demand) / (capacity_z + 1e-9) if capacity_z > 0 else 0
                            output_data_long.append({
                                "ID": elem_id, "Type": elem_type[:3], "Sec. ID": sec_idx, "Section": section_name, "Rot": rotation_flag,
                                "Force": "Axial-z (Interaction)",
                                "Demand": f"{demand:.2f}", "Combo": combo,
                                "Capacity": f"{capacity_z:.2f}",
                                "DCR": dcr_z
                            })
                            # 2. y축(약축) 모멘트와의 상호작용 고려
                            capacity_y = Pn_actual_y
                            dcr_y = abs(demand) / (capacity_y + 1e-9) if capacity_y > 0 else 0
                            output_data_long.append({
                                "ID": elem_id, "Type": elem_type[:3], "Sec. ID": sec_idx, "Section": section_name, "Rot": rotation_flag,
                                "Force": "Axial-y (Interaction)",
                                "Demand": f"{demand:.2f}", "Combo": combo,
                                "Capacity": f"{capacity_y:.2f}",
                                "DCR": dcr_y
                            })
                            continue # 축력은 별도 처리했으므로 다음 부재력으로 넘어감
                        elif force_name == 'Moment-y (kNm)': capacity = Mn_actual_z # 강축 모멘트
                        elif force_name == 'Moment-z (kNm)': capacity = Mn_actual_y # 약축 모멘트
                        elif force_name == 'Shear-y (kN)': capacity = strengths.get('Vn_y', 0)
                        elif force_name == 'Shear-z (kN)': capacity = strengths.get('Vn_z', 0)
                        else: capacity = float('inf')
                    else: # Beam
                        if force_name == 'Moment-y (kNm)': capacity = strengths.get('Mn_z', 0)
                        elif force_name == 'Shear-z (kN)': capacity = strengths.get('Vn_z', 0)
                        else: capacity = float('inf') # Beam의 약축/비틀림 등은 고려 안 함
                    
                    dcr_val = abs(demand) / (capacity + 1e-9) if capacity > 0 and capacity != float('inf') else 0

                    output_data_long.append({
                        "ID": elem_id, "Type": elem_type[:3], "Sec. ID": sec_idx, "Section": section_name, "Rot": rotation_flag,
                        "Force": force_name.replace(' (kN)','').replace(' (kNm)',''),
                        "Demand": f"{demand:.2f}", "Combo": combo,
                        "Capacity": f"{capacity:.2f}" if capacity != float('inf') else "N/A",
                        "DCR": dcr_val
                    })

            # 2. 각 부재별 최대 DCR 찾기 및 'Governing' 항목 추가
            if output_data_long:
                max_dcrs = {}
                for item in output_data_long:
                    elem_id = item['ID']
                    dcr = item['DCR']
                    if elem_id not in max_dcrs or dcr > max_dcrs[elem_id]:
                        max_dcrs[elem_id] = dcr
                
                for item in output_data_long:
                    item['Governing'] = '✅' if item['DCR'] == max_dcrs.get(item['ID']) and item['DCR'] > 0 else ''
                    item['DCR'] = f"{item['DCR']:.3f}" # 최종 출력을 위해 문자열로 변환

                # 3. 데이터프레임으로 변환 및 출력
                report_df_final = pd.DataFrame(output_data_long)
                print(report_df_final.to_string(index=False))

        
        print("="*90)


        # =================================================================
        # ===                 6. 최종 결과 시각화 (개별 저장)           ===
        # =================================================================
        print("\n\n" + "="*80)
        print("### 최종 결과 시각화 ###")
        
        # --- 그래프 1: 최적 구조물 형상 (개별 저장) ---
        print("\n- 최적 구조물 형상 그래프 생성 중...")
        ind_vis = all_results[0]['individual']
        col_ind_vis = ind_vis[:chromosome_structure['col_sec']]
        col_rot_vis = ind_vis[chromosome_structure['col_sec']:chromosome_structure['col_sec']+chromosome_structure['col_rot']]
        beam_ind_vis = ind_vis[chromosome_structure['col_sec']+chromosome_structure['col_rot']:]
        build_model_for_section(floors, H, column_locations, beam_connections, col_ind_vis, col_rot_vis, beam_ind_vis, col_map, beam_map, column_sections, beam_sections, num_columns)
        
        # 1.1: 2D Plan View
        fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
        plot_Structure(title='2D Plan View of Optimal Design', view='2D_plan', ax=ax_2d)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "optimal_structure_2D_plan.png"))
        # plt.show()

        # 1.2: 3D Node Numbering
        fig_3d_node = plt.figure(figsize=(10, 8))
        ax_3d_node = fig_3d_node.add_subplot(111, projection='3d')
        opsv.plot_model(node_labels=1, element_labels=0, az_el=(-60, 30), ax=ax_3d_node)
        ax_3d_node.set_title('Node Numbering of the Optimal Structural Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "optimal_structure_3D_nodes.png"))
        # plt.show()

        # 1.3: 3D Element Numbering
        fig_3d_elem = plt.figure(figsize=(10, 8))
        ax_3d_elem = fig_3d_elem.add_subplot(111, projection='3d')
        opsv.plot_model(node_labels=0, element_labels=1, az_el=(-60, 30), ax=ax_3d_elem)
        ax_3d_elem.set_title('Element Numbering of the Optimal Structural Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "optimal_structure_3D_elements.png"))
        # plt.show()

        visualize_load_patterns(column_locations, beam_connections, patterns_by_floor, output_folder)

        # --- 그래프 2: 최적화 과정 분석 (개별 저장) ---
        print("- 최적화 과정 분석 그래프 생성 중...")
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

        # 2.1: 목표 공간 파레토 전선
        fig_p_obj, ax_p_obj = plt.subplots(figsize=(8, 7))
        fitness1_vals = [r['ind_object'].fitness.values[0] for r in all_results]
        fitness2_vals = [r['ind_object'].fitness.values[1] for r in all_results]
        ax_p_obj.scatter(fitness2_vals, fitness1_vals, c=colors, s=80, edgecolors='k', alpha=0.8)
        ax_p_obj.set_title('Pareto Front in the Objective Space', fontsize=14)
        ax_p_obj.set_xlabel('Structural Conservatism ($f_2$, Mean DCR)')
        ax_p_obj.set_ylabel('Economic & Environmental Demand ($f_1$)')
        ax_p_obj.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "analysis_pareto_objective_space.png"))
        # plt.show()

        # 2.2: 실제 값 공간 파레토 전선
        fig_p_sol, ax_p_sol = plt.subplots(figsize=(9, 7))
        real_costs = [r['cost'] for r in all_results]
        real_co2s = [r['co2'] for r in all_results]
        sc = ax_p_sol.scatter(real_co2s, real_costs, c=fitness2_vals, cmap='viridis', s=80, edgecolors='k', alpha=0.8)
        fig_p_sol.colorbar(sc, ax=ax_p_sol).set_label('Mean DCR')
        ax_p_sol.set_title('Pareto Front in the Solution Space (Cost vs. CO2)', fontsize=14)
        ax_p_sol.set_xlabel('Total CO2')
        ax_p_sol.set_ylabel('Total Cost')
        ax_p_sol.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "analysis_pareto_solution_space.png"))
        # plt.show()

        # 2.3: Hall of Fame 수렴도
        fig_conv, ax_conv = plt.subplots(figsize=(8, 7))
        gen_hof = [s['gen'] for s in hof_stats_history]
        hof_best_obj1 = [s['best_obj1'] for s in hof_stats_history]
        hof_best_obj2 = [s['best_obj2'] for s in hof_stats_history]
        color1 = 'tab:blue'
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Fitness1 (Normalized Cost+CO2)", color=color1)
        ax_conv.plot(gen_hof, hof_best_obj1, color=color1, marker='o', linestyle='-', label="Best Fitness1")
        ax_conv.tick_params(axis='y', labelcolor=color1)
        ax_conv.grid(True)
        ax_conv_twin = ax_conv.twinx()
        color2 = 'tab:red'
        ax_conv_twin.set_ylabel("Fitness2 (Mean DCR)", color=color2)
        ax_conv_twin.plot(gen_hof, hof_best_obj2, color=color2, marker='s', linestyle='-', label="Best Fitness2")
        ax_conv_twin.tick_params(axis='y', labelcolor=color2)
        ax_conv.set_title('Convergence of the Hall of Fame', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "analysis_convergence_hof.png"))
        # plt.show()

        # 2.4: 유효해 비율 수렴도
        fig_valid, ax_valid = plt.subplots(figsize=(8, 7))
        gen = logbook.select("gen")
        valid_ratios_log = logbook.select("valid_ratio")
        ax_valid.plot(gen, valid_ratios_log, marker='o', linestyle='-', color='g')
        ax_valid.set_title("Ratio of Feasible Solutions per Generation", fontsize=14)
        ax_valid.set_xlabel("Generation")
        ax_valid.set_ylabel("Feasible Solutions (%)")
        ax_valid.set_ylim(0, 105)
        ax_valid.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "analysis_feasible_ratio.png"))
        # plt.show()
        
        # --- 그래프 3: 횡하중에 의한 제약조건 그래프 (개별 저장) ---
        print("- 횡하중 제약조건 그래프 생성 중...")
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
        
        # 3.1: 지진하중 층간 변위 (X-dir)
        fig_drift_x, ax_drift_x = plt.subplots(figsize=(7, 6))
        ax_drift_x.axvline(x=0.015, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(all_results):
            if drifts := r.get('story_drifts_x'):
                ax_drift_x.plot(drifts, range(1, len(drifts) + 1), marker='o', color=colors[i], alpha=0.7)
        ax_drift_x.set_title('Seismic Inter-story Drift (X-dir)'); ax_drift_x.set_xlabel('Drift Ratio'); ax_drift_x.set_ylabel('Story'); ax_drift_x.grid(True); ax_drift_x.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "displacement_check_seismic_drift_x.png"))
        # plt.show()

        # 3.2: 지진하중 층간 변위 (Y-dir)
        fig_drift_y, ax_drift_y = plt.subplots(figsize=(7, 6))
        ax_drift_y.axvline(x=0.015, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(all_results):
            if drifts := r.get('story_drifts_y'):
                ax_drift_y.plot(drifts, range(1, len(drifts) + 1), marker='o', color=colors[i], alpha=0.7)
        ax_drift_y.set_title('Seismic Inter-story Drift (Y-dir)'); ax_drift_y.set_xlabel('Drift Ratio'); ax_drift_y.set_ylabel('Story'); ax_drift_y.grid(True); ax_drift_y.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "displacement_check_seismic_drift_y.png"))
        # plt.show()

        # 3.3: 풍하중 층별 변위 (X-dir)
        fig_wind_x, ax_wind_x = plt.subplots(figsize=(7, 6))
        ax_wind_x.axvline(x=(floors * H) / 400.0, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(all_results):
            if disps := r.get('wind_displacements_x'):
                ax_wind_x.plot(disps, range(1, len(disps) + 1), marker='o', color=colors[i], alpha=0.7)
        ax_wind_x.set_title('Wind Lateral Displacement (X-dir)'); ax_wind_x.set_xlabel('Displacement (m)'); ax_wind_x.set_ylabel('Story'); ax_wind_x.grid(True); ax_wind_x.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "displacement_check_wind_disp_x.png"))
        # plt.show()

        # 3.4: 풍하중 층별 변위 (Y-dir)
        fig_wind_y, ax_wind_y = plt.subplots(figsize=(7, 6))
        ax_wind_y.axvline(x=(floors * H) / 400.0, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(all_results):
            if disps := r.get('wind_displacements_y'):
                ax_wind_y.plot(disps, range(1, len(disps) + 1), marker='o', color=colors[i], alpha=0.7)
        ax_wind_y.set_title('Wind Lateral Displacement (Y-dir)'); ax_wind_y.set_xlabel('Displacement (m)'); ax_wind_y.set_ylabel('Story'); ax_wind_y.grid(True); ax_wind_y.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "displacement_check_wind_disp_y.png"))
        # plt.show()

        # --- 그래프 4: 최적해 다각도 성능 비교 (개별 저장) ---
        print("- 최적해 다각도 성능 비교 그래프 생성 중...")
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

        # 4.1: Radar Chart
        fig_radar, ax_radar = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
        radar_metrics = ['cost', 'co2', 'mean_strength_ratio', 'max_strength_ratio']
        radar_labels = ['Cost Demand', 'Co2 Demand', 'Average Strength Ratio', 'Max Strength Ratio']
        radar_data = pd.DataFrame({m: [r[m] for r in all_results] for m in radar_metrics})
        radar_data_normalized = pd.DataFrame(index=radar_data.index)
        cost_co2_metrics = ['cost', 'co2']
        range_cost_co2 = radar_data[cost_co2_metrics].max() - radar_data[cost_co2_metrics].min()
        range_cost_co2[range_cost_co2 == 0] = 1.0
        radar_data_normalized[cost_co2_metrics] = (radar_data[cost_co2_metrics].max() - radar_data[cost_co2_metrics]) / range_cost_co2
        dcr_metrics = ['mean_strength_ratio', 'max_strength_ratio']
        range_dcr = radar_data[dcr_metrics].max() - radar_data[dcr_metrics].min()
        range_dcr[range_dcr == 0] = 1.0
        radar_data_normalized[dcr_metrics] = (radar_data[dcr_metrics].max() - radar_data[dcr_metrics]) / range_dcr
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
        for i, result in enumerate(all_results):
            values = radar_data_normalized.iloc[i].values.flatten().tolist()
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, color=colors[i])
            ax_radar.fill(angles, values, color=colors[i], alpha=0.15)
        ax_radar.set_title('Performance Comparison via Radar Chart', fontsize=14, y=1.15)
        ax_radar.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "solution_comparison_radar.png"))
        # plt.show()

        # 4.2: Box Plot
        fig_box, ax_box = plt.subplots(figsize=(12, 9))
        col_indices_per_sol = [r['individual'][:chromosome_structure['col_sec']] for r in all_results]
        beam_indices_per_sol = [r['individual'][chromosome_structure['col_sec']+chromosome_structure['col_rot']:] for r in all_results]
        positions_col = np.array(range(len(all_results))) * 2.0 - 0.4
        positions_beam = np.array(range(len(all_results))) * 2.0 + 0.4
        bp_col = ax_box.boxplot(col_indices_per_sol, positions=positions_col, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightblue')); bp_beam = ax_box.boxplot(beam_indices_per_sol, positions=positions_beam, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax_box.legend([bp_col["boxes"][0], bp_beam["boxes"][0]], ['Column Sections', 'Beam Sections'])
        ax_box.set_xticks(range(0, len(all_results) * 2, 2))
        ax_box.set_xticklabels([r['ID'] for r in all_results], rotation=45, ha='right')
        ax_box.set_title('Distribution of Design Variables for Each Pareto Solution', fontsize=14)
        ax_box.set_xlabel('Pareto Solution ID')
        ax_box.set_ylabel('Section Index ID'); ax_box.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "solution_comparison_boxplot.png"))
        # plt.show()
        
        # --- 그래프 5: 상세 응력비 분석 ---
        print("- 상세 응력비 분석 그래프 생성 중...")
        num_solutions = len(all_results)
        if num_solutions > 0:
            fig_stress, axs_stress = plt.subplots(num_solutions, 3, figsize=(20, 5 * num_solutions), squeeze=False)
            fig_stress.suptitle('Detailed Stress Ratio (DCR) Analysis per Solution', fontsize=20, y=1.0)
            for i, result in enumerate(all_results):
                ax_dist, ax_bar_col, ax_bar_beam = axs_stress[i, 0], axs_stress[i, 1], axs_stress[i, 2]
                ratios = result['strength_ratios']
                
                # --- 1열: 정렬된 DCR 분포 ---
                sorted_ratios = sorted(ratios, reverse=True); mean_ratio = result['mean_strength_ratio']
                ax_dist.plot(sorted_ratios, marker='.', linestyle='-', color='gray'); ax_dist.axhline(y=1.0, color='r', linestyle='--', label='Allowable Ratio (1.0)'); ax_dist.axhline(y=mean_ratio, color='b', linestyle='--', label=f'Avg DCR: {mean_ratio:.3f}')
                if sorted_ratios: ax_dist.plot(0, sorted_ratios[0], 'o', color='red', markersize=8, label=f'Max DCR: {sorted_ratios[0]:.3f}')
                ax_dist.set_title(f"[{result['ID']}] Sorted DCR Distribution", fontsize=12); ax_dist.set_xlabel('Members (Sorted)', fontsize=10); ax_dist.set_ylabel('DCR', fontsize=10)
                ax_dist.set_ylim(bottom=0, top=1.2); ax_dist.legend(loc='upper right', fontsize='small'); ax_dist.grid(axis='y', linestyle=':', alpha=0.7)
                # --- 2열 & 3열: 그룹별 최대 DCR 막대그래프 ---
                max_dcr_by_col_group = {i: 0.0 for i in range(num_col_groups)}
                max_dcr_by_beam_group = {i: 0.0 for i in range(num_beam_groups)}
                for c_idx in range(num_columns):
                    group_id = col_map[c_idx + 1]
                    max_dcr_by_col_group[group_id] = max(max_dcr_by_col_group[group_id], ratios[c_idx])
                for b_idx in range(num_beams):
                    group_id = beam_map[num_columns + b_idx + 1]
                    max_dcr_by_beam_group[group_id] = max(max_dcr_by_beam_group[group_id], ratios[num_columns + b_idx])
                if max_dcr_by_col_group:
                    col_groups = sorted(max_dcr_by_col_group.keys()); col_values = [max_dcr_by_col_group[g] for g in col_groups]
                    ax_bar_col.bar(col_groups, col_values, color='royalblue', edgecolor='black')
                    ax_bar_col.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
                    ax_bar_col.set_title(f"[{result['ID']}] Max DCR per Column Group", fontsize=12); ax_bar_col.set_xlabel('Column Group ID', fontsize=10); ax_bar_col.set_ylabel('Max DCR', fontsize=10)
                    ax_bar_col.set_xticks(col_groups); ax_bar_col.set_ylim(bottom=0, top=1.2); ax_bar_col.grid(axis='y', linestyle=':', alpha=0.7)
                if max_dcr_by_beam_group:
                    beam_groups = sorted(max_dcr_by_beam_group.keys()); beam_values = [max_dcr_by_beam_group[g] for g in beam_groups]
                    ax_bar_beam.bar(beam_groups, beam_values, color='seagreen', edgecolor='black')
                    ax_bar_beam.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
                    ax_bar_beam.set_title(f"[{result['ID']}] Max DCR per Beam Group", fontsize=12); ax_bar_beam.set_xlabel('Beam Group ID', fontsize=10)
                    ax_bar_beam.set_xticks(beam_groups); ax_bar_beam.set_ylim(bottom=0, top=1.2); ax_bar_beam.grid(axis='y', linestyle=':', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(output_folder, "stress_ratios_distribution.png")); plt.show()

        # =================================================================
        # ===                 7. 대표해 상세 시각화                       ===
        # =================================================================
        print("\n\n" + "="*90)
        print("### 대표 최적해 분석 및 시각화 ###")

        # --- 대표해 선정 (양 끝점, 중간점) ---
        representative_results = []
        if len(all_results) > 0:
            # 1. 최소 비용 해
            res1 = all_results[0]
            res1['Rep_ID'] = 'Rep_MinCost'
            representative_results.append(res1)

            # 2. 중간 해 (3개 이상일 때만 의미가 있음)
            if len(all_results) >= 3:
                res2 = all_results[len(all_results) // 2]
                res2['Rep_ID'] = 'Rep_MidPoint'
                representative_results.append(res2)

            # 3. 최대 보수성 해 (2개 이상일 때만 의미가 있음)
            if len(all_results) >= 2:
                res3 = all_results[-1]
                res3['Rep_ID'] = 'Rep_MaxConserv'
                representative_results.append(res3)
        
        # Rep_ID를 기준으로 중복 제거 (결과가 2개 이하일 때 중복될 수 있음)
        unique_reps = {r['ID']: r for r in representative_results}
        representative_results = list(unique_reps.values())

        print("\n[대표 최적해 선정 결과]")
        for r in representative_results:
            print(f"- {r['Rep_ID']}: {r['ID']} (Cost: {r['cost']:,.0f}, CO2: {r['co2']:,.0f})")
        print("="*90)

        # --- 그래프 6: 대표해 횡하중 제약조건 그래프 (개별 저장) ---
        print("\n- 대표해 횡하중 제약조건 그래프 생성 중...")
        colors_rep = plt.cm.jet(np.linspace(0, 1, len(representative_results)))

        # 6.1: 대표해 지진하중 층간 변위 (X-dir)
        fig_rep_drift_x, ax_rep_drift_x = plt.subplots(figsize=(7, 6))
        ax_rep_drift_x.axvline(x=0.015, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(representative_results):
            if drifts := r.get('story_drifts_x'):
                ax_rep_drift_x.plot(drifts, range(1, len(drifts) + 1), marker='o', color=colors_rep[i], label=r['Rep_ID'], alpha=0.8)
        ax_rep_drift_x.set_title('Seismic Inter-story Drift (X-dir) for Rep. Solutions'); ax_rep_drift_x.set_xlabel('Drift Ratio'); ax_rep_drift_x.set_ylabel('Story'); ax_rep_drift_x.grid(True); ax_rep_drift_x.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "representative_displacement_check_seismic_drift_x.png"))
        # plt.show()

        # 6.2: 대표해 지진하중 층간 변위 (Y-dir)
        fig_rep_drift_y, ax_rep_drift_y = plt.subplots(figsize=(7, 6))
        ax_rep_drift_y.axvline(x=0.015, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(representative_results):
            if drifts := r.get('story_drifts_y'):
                ax_rep_drift_y.plot(drifts, range(1, len(drifts) + 1), marker='o', color=colors_rep[i], label=r['Rep_ID'], alpha=0.8)
        ax_rep_drift_y.set_title('Seismic Inter-story Drift (Y-dir) for Rep. Solutions'); ax_rep_drift_y.set_xlabel('Drift Ratio'); ax_rep_drift_y.set_ylabel('Story'); ax_rep_drift_y.grid(True); ax_rep_drift_y.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "representative_displacement_check_seismic_drift_y.png"))
        # plt.show()

        # 6.3: 대표해 풍하중 층별 변위 (X-dir)
        fig_rep_wind_x, ax_rep_wind_x = plt.subplots(figsize=(7, 6))
        ax_rep_wind_x.axvline(x=(floors * H) / 400.0, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(representative_results):
            if disps := r.get('wind_displacements_x'):
                ax_rep_wind_x.plot(disps, range(1, len(disps) + 1), marker='o', color=colors_rep[i], label=r['Rep_ID'], alpha=0.8)
        ax_rep_wind_x.set_title('Wind Lateral Displacement (X-dir) for Rep. Solutions'); ax_rep_wind_x.set_xlabel('Displacement (m)'); ax_rep_wind_x.set_ylabel('Story'); ax_rep_wind_x.grid(True); ax_rep_wind_x.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "representative_displacement_check_wind_disp_x.png"))
        # plt.show()

        # 6.4: 대표해 풍하중 층별 변위 (Y-dir)
        fig_rep_wind_y, ax_rep_wind_y = plt.subplots(figsize=(7, 6))
        ax_rep_wind_y.axvline(x=(floors * H) / 400.0, color='r', linestyle='--', label='Limit')
        for i, r in enumerate(representative_results):
            if disps := r.get('wind_displacements_y'):
                ax_rep_wind_y.plot(disps, range(1, len(disps) + 1), marker='o', color=colors_rep[i], label=r['Rep_ID'], alpha=0.8)
        ax_rep_wind_y.set_title('Wind Lateral Displacement (Y-dir) for Rep. Solutions'); ax_rep_wind_y.set_xlabel('Displacement (m)'); ax_rep_wind_y.set_ylabel('Story'); ax_rep_wind_y.grid(True); ax_rep_wind_y.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "representative_displacement_check_wind_disp_y.png"))
        # plt.show()

        # --- 그래프 7: 대표해 다각도 성능 비교 (개별 저장) ---
        print("- 대표해 다각도 성능 비교 그래프 생성 중...")
        colors_rep = plt.cm.jet(np.linspace(0, 1, len(representative_results)))

        # 7.1: 대표해 Radar Chart
        fig_rep_radar, ax_rep_radar = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
        radar_metrics = ['cost', 'co2', 'mean_strength_ratio', 'max_strength_ratio']
        radar_labels = ['Cost Demand', 'Co2 Demand', 'Average Strength Ratio', 'Max Strength Ratio']
        radar_data_rep = pd.DataFrame({m: [r[m] for r in representative_results] for m in radar_metrics})
        radar_data_rep_normalized = pd.DataFrame(index=radar_data_rep.index)
        cost_co2_metrics = ['cost', 'co2']
        range_cost_co2_rep = radar_data_rep[cost_co2_metrics].max() - radar_data_rep[cost_co2_metrics].min()
        range_cost_co2_rep[range_cost_co2_rep == 0] = 1.0
        radar_data_rep_normalized[cost_co2_metrics] = (radar_data_rep[cost_co2_metrics].max() - radar_data_rep[cost_co2_metrics]) / range_cost_co2_rep
        dcr_metrics = ['mean_strength_ratio', 'max_strength_ratio']
        range_dcr_rep = radar_data_rep[dcr_metrics].max() - radar_data_rep[dcr_metrics].min()
        range_dcr_rep[range_dcr_rep == 0] = 1.0
        radar_data_rep_normalized[dcr_metrics] = (radar_data_rep[dcr_metrics].max() - radar_data_rep[dcr_metrics]) / range_dcr_rep
        angles_rep = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles_rep += angles_rep[:1]
        ax_rep_radar.set_thetagrids(np.degrees(angles_rep[:-1]), radar_labels)
        for i, result in enumerate(representative_results):
            values = radar_data_rep_normalized.iloc[i].values.flatten().tolist()
            values += values[:1]
            ax_rep_radar.plot(angles_rep, values, 'o-', linewidth=2, color=colors_rep[i], label=result['Rep_ID'])
            ax_rep_radar.fill(angles_rep, values, color=colors_rep[i], alpha=0.15)
        ax_rep_radar.set_title('Representative Solutions Performance Radar Chart', fontsize=14, y=1.15)
        ax_rep_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_rep_radar.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "representative_solution_comparison_radar.png"))
        # plt.show()

        # 7.2: 대표해 Box Plot
        fig_rep_box, ax_rep_box = plt.subplots(figsize=(12, 9))
        col_indices_per_rep = [r['individual'][:chromosome_structure['col_sec']] for r in representative_results]
        beam_indices_per_rep = [r['individual'][chromosome_structure['col_sec']+chromosome_structure['col_rot']:] for r in representative_results]
        positions_col_rep = np.array(range(len(representative_results))) * 2.0 - 0.4
        positions_beam_rep = np.array(range(len(representative_results))) * 2.0 + 0.4
        bp_col_rep = ax_rep_box.boxplot(col_indices_per_rep, positions=positions_col_rep, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightblue')); bp_beam_rep = ax_rep_box.boxplot(beam_indices_per_rep, positions=positions_beam_rep, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax_rep_box.legend([bp_col_rep["boxes"][0], bp_beam_rep["boxes"][0]], ['Column Sections', 'Beam Sections'])
        ax_rep_box.set_xticks(range(0, len(representative_results) * 2, 2))
        ax_rep_box.set_xticklabels([r['Rep_ID'] for r in representative_results], rotation=45, ha='right')
        ax_rep_box.set_title('Design Variable Distribution for Rep. Solutions', fontsize=14)
        ax_rep_box.set_xlabel('Representative Solution ID')
        ax_rep_box.set_ylabel('Section Index ID'); ax_rep_box.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "representative_solution_comparison_boxplot.png"))
        # plt.show()

        # --- 그래프 8: 대표해 상세 응력비 분석 (솔루션별 개별 저장) ---
        print("- 대표해 상세 응력비 분석 그래프 생성 중...")
        if len(representative_results) > 0:
            for i, result in enumerate(representative_results):
                fig_rep_stress, axs_rep_stress = plt.subplots(1, 3, figsize=(20, 5), squeeze=False)
                title_prefix = f"[{result['Rep_ID']}: {result['ID']}]"
                fig_rep_stress.suptitle(f'Detailed Stress Ratio (DCR) Analysis for {title_prefix}', fontsize=16)
                
                ax_dist, ax_bar_col, ax_bar_beam = axs_rep_stress[0, 0], axs_rep_stress[0, 1], axs_rep_stress[0, 2]
                ratios = result['strength_ratios']
                
                # --- 1열: 정렬된 DCR 분포 ---
                sorted_ratios = sorted(ratios, reverse=True)
                mean_ratio = result['mean_strength_ratio']
                ax_dist.plot(sorted_ratios, marker='.', linestyle='-', color='gray')
                ax_dist.axhline(y=1.0, color='r', linestyle='--', label='Allowable Ratio (1.0)')
                ax_dist.axhline(y=mean_ratio, color='b', linestyle='--', label=f'Avg DCR: {mean_ratio:.3f}')
                if sorted_ratios:
                    ax_dist.plot(0, sorted_ratios[0], 'o', color='red', markersize=8, label=f'Max DCR: {sorted_ratios[0]:.3f}')
                
                ax_dist.set_title(f"Sorted DCR Distribution", fontsize=12)
                ax_dist.set_xlabel('Members (Sorted)', fontsize=10)
                ax_dist.set_ylabel('DCR', fontsize=10)
                ax_dist.set_ylim(bottom=0, top=1.2)
                ax_dist.legend(loc='upper right', fontsize='small')
                ax_dist.grid(axis='y', linestyle=':', alpha=0.7)
                
                # --- 2열 & 3열: 그룹별 최대 DCR 막대그래프 ---
                max_dcr_by_col_group = {i: 0.0 for i in range(num_col_groups)}
                max_dcr_by_beam_group = {i: 0.0 for i in range(num_beam_groups)}
                for c_idx in range(num_columns):
                    group_id = col_map[c_idx + 1]
                    max_dcr_by_col_group[group_id] = max(max_dcr_by_col_group[group_id], ratios[c_idx])
                for b_idx in range(num_beams):
                    group_id = beam_map[num_columns + b_idx + 1]
                    max_dcr_by_beam_group[group_id] = max(max_dcr_by_beam_group[group_id], ratios[num_columns + b_idx])
                if max_dcr_by_col_group:
                    col_groups = sorted(max_dcr_by_col_group.keys()); col_values = [max_dcr_by_col_group[g] for g in col_groups]
                    ax_bar_col.bar(col_groups, col_values, color='royalblue', edgecolor='black')
                    ax_bar_col.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
                    ax_bar_col.set_title(f"Max DCR per Column Group", fontsize=12); ax_bar_col.set_xlabel('Column Group ID', fontsize=10); ax_bar_col.set_ylabel('Max DCR', fontsize=10)
                    ax_bar_col.set_xticks(col_groups); ax_bar_col.set_ylim(bottom=0, top=1.2); ax_bar_col.grid(axis='y', linestyle=':', alpha=0.7)
                if max_dcr_by_beam_group:
                    beam_groups = sorted(max_dcr_by_beam_group.keys()); beam_values = [max_dcr_by_beam_group[g] for g in beam_groups]
                    ax_bar_beam.bar(beam_groups, beam_values, color='seagreen', edgecolor='black')
                    ax_bar_beam.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
                    ax_bar_beam.set_title(f"Max DCR per Beam Group", fontsize=12); ax_bar_beam.set_xlabel('Beam Group ID', fontsize=10)
                    ax_bar_beam.set_xticks(beam_groups); ax_bar_beam.set_ylim(bottom=0, top=1.2); ax_bar_beam.grid(axis='y', linestyle=':', alpha=0.7)
            
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(output_folder, f"representative_stress_ratios_{result['Rep_ID']}.png"))
                # plt.show()

    h5_file.close()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    
    # 실행 시간을 파일로 저장 (R1-5 대응)
    with open(os.path.join(output_folder, "execution_time.txt"), "w") as f:
        f.write(f"Total Execution Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total Execution Time: {elapsed_time/60:.2f} minutes\n")
        f.write(f"Total Execution Time: {elapsed_time/3600:.2f} hours\n")

if __name__ == '__main__':
    main()