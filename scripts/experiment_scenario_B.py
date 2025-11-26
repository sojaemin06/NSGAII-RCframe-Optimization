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


def main():
    # =================================================================
    # ===                  라이브러리 임포트                          ===
    # =================================================================
    import openseespy.opensees as ops
    from deap import base, creator, tools, algorithms

    
    # --- ✅ 1. Matplotlib 전역 글꼴 설정: Times New Roman ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12  # 기본 글꼴 크기 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    # --- ✅ 스크립트의 절대 경로를 기준으로 파일 경로 설정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    RANDOM_SEED = 42 
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # =================================================================
    # ===              1. 최적화 및 모델링 주요 설정                  ===
    # =================================================================

    # --- 1.0. 실험 시나리오 제어 플래그 ---
    # 시나리오 B: 물리적 확장
    USE_EXPANDED_DATA = True  # 확장된 DB 사용 (시나리오 B의 핵심)
    ENABLE_ROTATION = False   # 기둥 회전 비활성화

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
    # --- 실험을 위해 Y방향 하중 제거 ---
    Wx_rand=35
    Wy_rand=0
    Ex_rand=40
    Ey_rand=0

    # --- 1.5. 유전 알고리즘 파라미터 (실험을 위해 축소) ---
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 200
    CXPB, MUTPB = 0.8, 0.2

    # --- 1.6. 부재 단면 정보 로드 (실험 시나리오에 따라 분기) ---
    if USE_EXPANDED_DATA:
        print("[Experiment] Using EXPANDED (Rotated) column dataset.")
        beam_sections_df = pd.read_csv(os.path.join(script_dir, "../data/beam_sections_simple02.csv"))
        column_sections_df = pd.read_csv(os.path.join(script_dir, "../data/column_sections_expanded_rotated.csv"))
        h5_file = h5py.File(os.path.join(script_dir, '../data/pm_dataset_expanded_rotated.mat'), 'r')
    else:
        print("[Experiment] Using ORIGINAL column dataset.")
        beam_sections_df = pd.read_csv(os.path.join(script_dir, "../data/beam_sections_simple02.csv"))
        column_sections_df = pd.read_csv(os.path.join(script_dir, "../data/column_sections_simple02.csv"))
        h5_file = h5py.File(os.path.join(script_dir, '../data/pm_dataset_simple02.mat'), 'r')

    # --- ✅ 하중 조합 축소 (해석 시간 단축용) ---
    load_combinations = [
        # --- 강도 설계용 (Strength Design) ---
        ("ACI-2", {"DL": 1.2, "LL": 1.6, "Wx": 0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-3", {"DL": 1.2, "LL": 1.0, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-7", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": 0.3}),
        ("ACI-15", {"DL": 0.9, "LL": 0, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ACI-19", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": 0.3}),
        # --- 사용성 검토용 (Serviceability) ---
        ("ASCE-S-W1", {"DL": 1.0, "LL": 0.5, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-E1", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.7, "Ey": 0.21}),
    ]

    # --- 2.1. 그룹핑 전략 실행 ---
    col_map, beam_map, num_col_groups, num_beam_groups = get_grouping_maps(GROUPING_STRATEGY, floors, column_locations, beam_connections)
    num_columns = len(column_locations) * floors
    
    # --- 2.2. 유전 정보(Chromosome) 구조 정의 ---
    chromosome_structure = {'col_sec': num_col_groups, 'beam_sec': num_beam_groups}
    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    chromosome_len = sum(chromosome_structure.values())
    beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
    column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
    beam_lengths = [math.sqrt((column_locations[p2][0] - column_locations[p1][0])**2 + (column_locations[p2][1] - column_locations[p1][1])**2) for p1, p2 in beam_connections]

    # --- 데이터 기반 고정 스케일 자동 계산 (최종안) ---
    total_column_length = (len(column_locations) * floors) * H
    total_beam_length = sum(beam_lengths) * floors
    min_col_cost_per_m = column_sections_df['Cost'].min()
    max_col_cost_per_m = column_sections_df['Cost'].max()
    min_beam_cost_per_m = beam_sections_df['Cost'].min()
    max_beam_cost_per_m = beam_sections_df['Cost'].max()
    min_col_co2_per_m = column_sections_df['CO2'].min()
    max_col_co2_per_m = column_sections_df['CO2'].max()
    min_beam_co2_per_m = beam_sections_df['CO2'].min()
    max_beam_co2_per_m = beam_sections_df['CO2'].max()
    FIXED_MIN_COST = (min_col_cost_per_m * total_column_length) + (min_beam_cost_per_m * total_beam_length)
    FIXED_MAX_COST = (max_col_cost_per_m * total_column_length) + (max_beam_cost_per_m * total_beam_length)
    FIXED_RANGE_COST = FIXED_MAX_COST - FIXED_MIN_COST
    if FIXED_RANGE_COST == 0: FIXED_RANGE_COST = 1.0
    FIXED_MIN_CO2 = (min_col_co2_per_m * total_column_length) + (min_beam_co2_per_m * total_beam_length)
    FIXED_MAX_CO2 = (max_col_co2_per_m * total_column_length) + (max_beam_co2_per_m * total_beam_length)
    FIXED_RANGE_CO2 = FIXED_MAX_CO2 - FIXED_MIN_CO2
    if FIXED_RANGE_CO2 == 0: FIXED_RANGE_CO2 = 1.0
    print("\n[Data-driven Fixed Scale for Normalization]")
    print(f"- Estimated Cost Range: {FIXED_MIN_COST:,.0f} ~ {FIXED_MAX_COST:,.0f}")
    print(f"- Estimated CO2 Range : {FIXED_MIN_CO2:,.0f} ~ {FIXED_MAX_CO2:,.0f}\n")

    # =================================================================
    # ===                      메인 실행 블록                         ===
    # =================================================================
    # --- ✅ 2. 하중 재하 패턴 정의 (슬래브 하중 분배 방식) ---
    # 각 층별로 하중이 재하될 보의 인덱스(beam_connections 기준)를 지정합니다.
    # 예: 1층의 0, 1, 2번 보에 하중을 가하려면 {1: {0, 1, 2}}
    patterns_by_floor = {
        1: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
        2: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
        3: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
        4: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
    }

    output_folder = "../results/Results_Scenario_B_02"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n[Scenario B] Optimization results will be saved in '{output_folder}' folder.")

    # 최적화 실행
    pop, logbook, hof, hof_stats_history = run_ga_optimization(
        chromosome_structure, DL_rand, LL_rand, Wx_rand, Wy_rand, Ex_rand, Ey_rand,
        CROSSOVER_STRATEGY, patterns_by_floor, h5_file,
        NUM_GENERATIONS, POPULATION_SIZE, CXPB, MUTPB,
        beam_lengths, col_map, beam_map, num_columns, floors, H, column_locations, beam_connections,
        column_sections_df, beam_sections_df, FIXED_MIN_COST, FIXED_RANGE_COST, FIXED_MIN_CO2, FIXED_RANGE_CO2,
        load_combinations, column_sections, beam_sections
    )

    # --- 결과 처리 및 저장 ---
    log_df = pd.DataFrame(logbook)
    log_df.to_csv(os.path.join(output_folder, "optimization_log_B.csv"), index=False)
    if hof_stats_history:
        hof_df = pd.DataFrame(hof_stats_history)
        hof_df.to_csv(os.path.join(output_folder, "hof_convergence_B.csv"), index=False)

    # --- 파레토 최적해 저장 ---
    valid_solutions = [ind for ind in hof if ind.detailed_results.get('violation') == 0.0]
    if valid_solutions:
        pareto_data = []
        for ind in valid_solutions:
            # 'forces_df'와 같이 큰 데이터는 제외하고 저장
            res = {k: v for k, v in ind.detailed_results.items() if not isinstance(v, pd.DataFrame)}
            res['fitness_obj1'] = ind.fitness.values[0]
            res['fitness_obj2'] = ind.fitness.values[1]
            pareto_data.append(res)
        
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(os.path.join(output_folder, "pareto_solutions_B.csv"), index=False)
        print(f"Saved {len(pareto_df)} valid Pareto solutions to 'pareto_solutions_B.csv'")

    print("\n" + "="*80)
    print("### Scenario B Optimization Complete ###")
    print(f"Found {len(hof)} solutions in the Pareto front.")
    print("="*80)

if __name__ == "__main__":
    # 메인 함수를 호출하여 전체 최적화 프로세스를 시작합니다.
    main()