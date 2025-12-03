def main():
    # =================================================================
    # ===                  라이브러리 임포트               ㅗ           ===
    # =================================================================
    import openseespy.opensees as ops
    import pandas as pd
    import numpy as np
    import h5py
    import math
    import os
    import random
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import opsvis as opsv
    from scipy.spatial import ConvexHull
    from deap import base, creator, tools, algorithms

    # --- ✅ 1. Matplotlib 전역 글꼴 설정: Times New Roman ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12  # 기본 글꼴 크기 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    # RANDOM_SEED = 42 
    # random.seed(RANDOM_SEED)
    # np.random.seed(RANDOM_SEED)

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
    Wx_rand=35
    Wy_rand=38
    Ex_rand=40
    Ey_rand=40

    # --- 1.5. 유전 알고리즘 파라미터 ---
    POPULATION_SIZE = 50
    NUM_GENERATIONS = 200
    CXPB, MUTPB = 0.8, 0.2

    # --- 1.6. 부재 단면 정보 로드 ---
    beam_sections_df = pd.read_csv("beam_sections_simple02.csv")
    column_sections_df = pd.read_csv("column_sections_simple02.csv")
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r') # 데이터로드용

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
        ("ACI-22", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": -0.3}),
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

    # =================================================================
    # ===                  2. 그룹핑 및 유틸리티 함수                   ===
    # =================================================================

    # --- 2.1. 그룹핑 전략 실행 ---
    num_locations = len(column_locations)
    num_columns = num_locations * floors
    num_beams = len(beam_connections) * floors
    col_map, beam_map = {}, {}

    if GROUPING_STRATEGY == "Individual":
        num_col_groups, num_beam_groups = num_columns, num_beams
        for i in range(num_columns): col_map[i + 1] = i
        for i in range(num_beams): beam_map[num_columns + i + 1] = i
    elif GROUPING_STRATEGY == "Uniform":
        num_col_groups, num_beam_groups = 1, 1
        for i in range(num_columns): col_map[i + 1] = 0
        for i in range(num_beams): beam_map[num_columns + i + 1] = 0
    elif GROUPING_STRATEGY == "ByFloor":
        num_col_groups, num_beam_groups = floors, floors
        cols_per_floor = num_locations
        beams_per_floor = len(beam_connections)
        for k in range(floors):
            for i in range(cols_per_floor): col_map[k * cols_per_floor + i + 1] = k
            for i in range(beams_per_floor): beam_map[num_columns + k * beams_per_floor + i + 1] = k
    elif GROUPING_STRATEGY == "Hybrid":
        # --- 개선된 Hybrid 그룹핑 전략 (오목 코너 지원) ---
        print("\n[Hybrid Grouping] Analyzing column connectivity and local geometry for grouping...")

        # 1. 보 연결성 계산
        node_connectivity = {i: 0 for i in range(len(column_locations))}
        for p1_idx, p2_idx in beam_connections:
            node_connectivity[p1_idx] += 1
            node_connectivity[p2_idx] += 1

        # 2. 기둥 위치 유형 판별을 위한 Helper 함수 및 Memoization 캐시
        memo_col_type = {}
        
        def is_point_inside_hull(point, hull):
            """점이 ConvexHull 내부에 있는지 확인하는 Helper 함수."""
            return np.all(np.add(np.dot(hull.equations[:, :-1], point), hull.equations[:, -1]) < 1e-9)

        def get_col_loc_type(loc_idx):
            """보 연결 개수와 주변 기둥의 기하학적 배치를 이용해 기둥 유형을 결정합니다."""
            if loc_idx in memo_col_type:
                return memo_col_type[loc_idx]

            connections = node_connectivity.get(loc_idx, 0)
            loc_type = -1 

            if connections <= 2:
                loc_type = 0  # 코너 (볼록 코너 또는 라인 끝)
            elif connections == 3:
                loc_type = 1  # 엣지
            else: # connections >= 4, '내부'와 '오목 코너' 구분 필요
                neighbor_indices = {p[1] if p[0] == loc_idx else p[0] for p in beam_connections if loc_idx in p}
                
                if len(neighbor_indices) < 3:
                    loc_type = 2 # 이웃이 3개 미만이면 Hull 생성 불가, 내부로 간주
                else:
                    neighbor_points = np.array([column_locations[i] for i in neighbor_indices])
                    current_point = np.array(column_locations[loc_idx])
                    
                    try:
                        neighbors_hull = ConvexHull(neighbor_points)
                        if is_point_inside_hull(current_point, neighbors_hull):
                            loc_type = 2  # 점이 이웃들의 Hull 내부에 있으면 '내부' 기둥
                        else:
                            loc_type = 0  # 경계에 있거나 외부에 있으면 '오목 코너' -> 코너 그룹
                    except Exception: # QhullError (e.g., collinear points)
                        loc_type = 2 # 예외 발생 시 안전하게 '내부'로 분류

            memo_col_type[loc_idx] = loc_type
            return loc_type

        # 3. 보 그룹핑: 건물 평면의 전체 Convex Hull을 이용하여 외부보/내부보 구분
        print("[Hybrid Grouping] Analyzing beam location using global Convex Hull...")
        points = np.array(column_locations)
        hull = ConvexHull(points)
        
        # --- 수정된 외부 보 판별 로직 ---
        # ConvexHull의 방정식(ax+by+d=0)을 이용하여, 두 끝점이 모두 특정 경계선 위에 놓이는 보를 외부 보로 판별합니다.
        # 이 방법은 보가 Convex Hull의 꼭지점(vertex)을 직접 연결하지 않더라도 경계선 위에 있는 경우를 올바르게 찾아냅니다.
        equations = hull.equations
        perimeter_beam_indices = set()
        for i, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            # 보의 두 끝점이 하나의 hull 경계선 상에 있는지 확인
            for eq in equations:
                # 두 점이 모두 해당 방정식(선)을 만족하는지 체크 (공차 1e-9)
                on_line1 = abs(eq[0] * p1[0] + eq[1] * p1[1] + eq[2]) < 1e-9
                on_line2 = abs(eq[0] * p2[0] + eq[1] * p2[1] + eq[2]) < 1e-9
                if on_line1 and on_line2:
                    perimeter_beam_indices.add(i)
                    break # 외부 보로 판별되었으면 다음 보로 넘어감

        # 4. 최종 그룹 ID 할당
        floor_step = 2
        num_floor_groups = math.ceil(floors / floor_step)
        num_col_groups = num_floor_groups * 3  # 3가지 타입 (코너, 엣지, 내부)
        num_beam_groups = num_floor_groups * 2  # 2가지 타입 (외부, 내부)
        for abs_col_idx in range(num_columns):
            floor_idx, loc_idx = divmod(abs_col_idx, num_locations)
            floor_group_idx = floor_idx // floor_step
            loc_type = get_col_loc_type(loc_idx)
            group_id = floor_group_idx * 3 + loc_type
            col_map[abs_col_idx + 1] = group_id
        for abs_beam_idx in range(num_beams):
            floor_idx, conn_idx = divmod(abs_beam_idx, len(beam_connections))
            floor_group_idx = floor_idx // floor_step
            loc_type = 0 if conn_idx in perimeter_beam_indices else 1  # 0: Perimeter, 1: Interior
            group_id = floor_group_idx * 2 + loc_type
            beam_map[num_columns + abs_beam_idx + 1] = group_id

    # --- 2.2. 유전 정보(Chromosome) 구조 정의 ---
    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    chromosome_len = sum(chromosome_structure.values())
    beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
    column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
    beam_lengths = [math.sqrt((column_locations[p2][0] - column_locations[p1][0])**2 + (column_locations[p2][1] - column_locations[p1][1])**2) for p1, p2 in beam_connections]

    # --- 데이터 기반 고정 스케일 자동 계산 (최종안) ---
    # 1. 구조물의 총 부재 길이 계산
    total_column_length = (len(column_locations) * floors) * H
    total_beam_length = sum(beam_lengths) * floors
    # 2. 데이터베이스에서 단위 길이당 min/max 값 추출 (오타 방지를 위해 전체 제공)
    min_col_cost_per_m = column_sections_df['Cost'].min()
    max_col_cost_per_m = column_sections_df['Cost'].max()
    min_beam_cost_per_m = beam_sections_df['Cost'].min()
    max_beam_cost_per_m = beam_sections_df['Cost'].max()
    
    min_col_co2_per_m = column_sections_df['CO2'].min()
    max_col_co2_per_m = column_sections_df['CO2'].max()
    min_beam_co2_per_m = beam_sections_df['CO2'].min()
    max_beam_co2_per_m = beam_sections_df['CO2'].max()
    # 3. 구조물 전체의 이론적 min/max 값 계산
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

    # --- 2.3. 유틸리티 함수 정의 ---
    def plot_Structure(title='Structure Shape', view='3D', ax=None):
        """건물 구조의 형상을 2D 또는 3D로 시각화하는 함수."""
        if view == '2D_plan':
            if ax is None: fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(title); ax.clear()
            for (idx1, idx2) in beam_connections:
                p1, p2 = column_locations[idx1], column_locations[idx2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=3, zorder=1)
            xs, ys = [loc[0] for loc in column_locations], [loc[1] for loc in column_locations]
            ax.scatter(xs, ys, c='b', s=100, zorder=2, label='Columns')
            ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.legend(); ax.grid(True)
        elif view == '3D':
            plt.figure(figsize=(12, 10))
            opsv.plot_model(node_labels=0, element_labels=0, az_el=(-60, 30))
            plt.title(title)

    def load_pm_data_for_column(h5_file, column_index):
        """h5py(.mat) 파일에서 특정 기둥 단면의 P-M 상호작용 데이터를 로드하는 함수."""
        try:
            ref = h5_file['Column_Mdata'][0, column_index]; pm_data = h5_file[ref][()]
            return pd.DataFrame(pm_data, columns=['Mnb_z','Pnb_z','Mnb_y','Pnb_y','Alpha_PI_Mnb_z','Alpha_PI_Pnb_z','Alpha_PI_Mnb_y','Alpha_PI_Pnb_y'])
        except Exception: return pd.DataFrame()

    def get_pm_capacity_from_df(slope, pm_df, axis='z'):
        """P-M 상호작용 데이터와 현재 하중경로(slope)를 이용해 공칭강도(Pn, Mn)를 계산하는 함수."""
        if pm_df.empty: return 0.0, 0.0
        moment_col, axial_col = (f'Alpha_PI_Mnb_{axis}', f'Alpha_PI_Pnb_{axis}')
        design_m, design_p = pm_df[moment_col].values, pm_df[axial_col].values
        curve_slopes = abs(design_p) / (abs(design_m) + 1e-9)
        for i in range(len(curve_slopes) - 1):
            if curve_slopes[i] >= slope >= curve_slopes[i+1]:
                m1, p1, m2, p2 = design_m[i], design_p[i], design_m[i+1], design_p[i+1]
                if abs(m2 - m1) < 1e-9: continue

                # 두 점을 잇는 직선의 방정식: P = a*M + b
                a1 = (p2 - p1) / (m2 - m1); b1 = p1 - a1 * m1

                # 하중 경로선(P = slope*M)과의 교점 찾기
                if abs(a1 - slope) < 1e-9: continue
                Mn = -b1 / (a1 - slope)
                Pn = slope * Mn
                return Pn, Mn # 축력(Pn)과 모멘트(Mn)를 함께 반환
        return 0.0, 0.0

    def get_precalculated_strength(element_type, index, col_df, beam_df):
        """단면 데이터프레임에서 사전 계산된 공칭강도(Pn, Vn 등)를 가져오는 함수."""
        strengths = {}
        try:
            if element_type == 'Column':
                row = col_df.iloc[index]; strengths.update({'Pn': row['PI_Pn_max'], 'Vn_y': row['PI_Vn_y'], 'Vn_z': row['PI_Vn_z']})
            elif element_type == 'Beam':
                row = beam_df.iloc[index]; strengths.update({'Pn': float('inf'), 'Vn_y': float('inf'), 'Vn_z': row['PiVn'], 'Mn_y': float('inf'), 'Mn_z': row['PiM']})
        except IndexError: return {'Pn': 0, 'Vn_y': 0, 'Vn_z': 0, 'Mn_y': 0, 'Mn_z': 0}
        return strengths

    def extract_local_element_forces(column_elem_ids, beam_elem_ids):
        """OpenSees 해석 후, 모든 부재의 로컬 좌표계 부재력을 추출하여 DataFrame으로 반환하는 함수."""
        all_forces = []
        def append_column_forces(eid):
            try:
                f=ops.eleResponse(eid,'localForce')
                all_forces.append({'ElementType':'Column','ElementID':eid,'Node':'i','Axial (kN)':-f[0],'Shear-y (kN)':f[2],'Shear-z (kN)':f[1],'Torsion (kNm)':f[3],'Moment-y (kNm)':f[5],'Moment-z (kNm)':-f[4]})
                all_forces.append({'ElementType':'Column','ElementID':eid,'Node':'j','Axial (kN)':f[6],'Shear-y (kN)':-f[8],'Shear-z (kN)':-f[7],'Torsion (kNm)':-f[9],'Moment-y (kNm)':-f[11],'Moment-z (kNm)':f[10]})
            except: pass
        def append_beam_forces(eid):
            try:
                f=ops.eleResponse(eid,'localForce')
                all_forces.append({'ElementType':'Beam','ElementID':eid,'Node':'i','Axial (kN)':f[0],'Shear-y (kN)':-f[1],'Shear-z (kN)':-f[2],'Torsion (kNm)':f[3],'Moment-y (kNm)':f[4],'Moment-z (kNm)':-f[5]})
                all_forces.append({'ElementType':'Beam','ElementID':eid,'Node':'j','Axial (kN)':f[6],'Shear-y (kN)':f[7],'Shear-z (kN)':f[8],'Torsion (kNm)':f[9],'Moment-y (kNm)':-f[10],'Moment-z (kNm)':f[11]})
            except: pass
        for eid in column_elem_ids:append_column_forces(eid)
        for eid in beam_elem_ids:append_beam_forces(eid)
        if not all_forces: return pd.DataFrame()
        df_all = pd.DataFrame(all_forces)
        df_all['ElementType'] = pd.Categorical(df_all['ElementType'],categories=['Column','Beam'],ordered=True)
        df_all = df_all.sort_values(['ElementType','ElementID'])
        df_max = (df_all.groupby(['ElementType','ElementID'], observed=True).agg({col:lambda x: x.iloc[0] if abs(x.iloc[0]) > abs(x.iloc[1]) else x.iloc[1] for col in ['Axial (kN)','Shear-y (kN)','Shear-z (kN)','Torsion (kNm)','Moment-y (kNm)','Moment-z (kNm)']}).reset_index())
        return df_max

    def build_model_for_section(col_indices, col_rotations, beam_indices):
        """주어진 유전자 정보를 바탕으로 OpenSees에서 3D 골조 모델을 생성하는 함수."""
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        E, nu = 2.5791e7, 0.167; G = E / (2 * (1 + nu))
        node_map = {}; node_id_counter = 1
        for k in range(floors + 1):
            for i, (x, y) in enumerate(column_locations):
                ops.node(node_id_counter, x, y, k * H)
                node_map[(k, i)] = node_id_counter
                if k == 0: ops.fix(node_id_counter, 1, 1, 1, 1, 1, 1)
                node_id_counter += 1
        ops.geomTransf('Linear', 1, 1, 0, 0); ops.geomTransf('Linear', 2, 0, 1, 0); ops.geomTransf('Linear', 3, 0, 0, 1)
        column_elem_ids, beam_elem_ids = [], []; elem_id_counter = 1
        for k in range(floors):
            for i in range(num_locations):
                abs_col_idx = k * num_locations + i; group_idx = col_map[abs_col_idx + 1]
                rotation_flag = col_rotations[group_idx]; transf_tag = 2 if rotation_flag == 1 else 1
                sec_idx = col_indices[group_idx]; b_c, h_c = column_sections[sec_idx]
                A_c, Iz_c, Iy_c = b_c*h_c, (b_c*h_c**3)/12, (h_c*b_c**3)/12; J_c = Iy_c + Iz_c
                n1, n2 = node_map[(k, i)], node_map[(k + 1, i)]
                ops.element('elasticBeamColumn', elem_id_counter, n1, n2, A_c, E, G, J_c, Iy_c, Iz_c, transf_tag)
                column_elem_ids.append(elem_id_counter); elem_id_counter += 1
        for k in range(1, floors + 1):
            for i, (loc_idx1, loc_idx2) in enumerate(beam_connections):
                abs_beam_idx = (k - 1) * len(beam_connections) + i; group_idx = beam_map[num_columns + abs_beam_idx + 1]
                sec_idx = beam_indices[group_idx]; b_b, h_b = beam_sections[sec_idx]
                A_b, Iz_b, Iy_b = b_b*h_b, (b_b*h_b**3)/12, (h_b*b_b**3)/12; J_b = Iy_b + Iz_b
                n1, n2 = node_map[(k, loc_idx1)], node_map[(k, loc_idx2)]
                ops.element('elasticBeamColumn', elem_id_counter, n1, n2, A_b, E, G, J_b, Iy_b, Iz_b, 3)
                beam_elem_ids.append(elem_id_counter); elem_id_counter += 1
        return column_elem_ids, beam_elem_ids, node_map

    def visualize_load_patterns(column_locations, beam_connections, patterns_by_floor, output_folder="."):
        """
        주어진 하중 패턴을 각 층의 평면도에 시각화하고 이미지 파일로 저장하는 함수.
        """
        print("\n[Visualization] Generating load pattern plots...")
        
        num_floors = len(patterns_by_floor)
        # 층 수에 맞춰 subplot 개수 동적 조정
        ncols = min(num_floors, 4) 
        nrows = (num_floors - 1) // ncols + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        fig.suptitle('Visualization of Applied Load Patterns by Floor', fontsize=16)

        for i, (floor_num, loaded_indices) in enumerate(patterns_by_floor.items()):
            ax = axes[i // ncols, i % ncols]
            ax.set_title(f"Floor {floor_num} Load Pattern")
            
            # 모든 기둥 위치 그리기
            xs = [loc[0] for loc in column_locations]
            ys = [loc[1] for loc in column_locations]
            ax.scatter(xs, ys, c='black', s=50, zorder=2)

            # 모든 보와 인덱스 번호 그리기 (기본 색상)
            for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
                p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=2, zorder=1)
                center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                ax.text(center_x, center_y, str(conn_idx), color='gray', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

            # 하중이 재하된 보 그리기 (강조 색상)
            for conn_idx in loaded_indices:
                p1_idx, p2_idx = beam_connections[conn_idx]
                p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=4, zorder=3, label='Loaded Beams' if 'Loaded Beams' not in [l.get_label() for l in ax.get_lines()] else "")

            ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.grid(True, linestyle='--', alpha=0.6)
            if i == 0: ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(output_folder, "load_pattern_visualization.png")
        plt.savefig(save_path); print(f"Load pattern visualization saved to '{save_path}'")
        # plt.show()

    # =================================================================
    # ===               3. DEAP 평가 함수 (Fitness Function)          ===
    # =================================================================

    def evaluate(individual, DL, LL, Wx, Wy, Ex, Ey, h5_file, patterns_by_floor):
        """
        유전 알고리즘의 핵심 평가 함수. 하나의 유전자를 입력받아 구조 해석을 수행하고,
        설계 제약조건 위반 여부와 목표 함수 값을 포함한 상세 결과 딕셔너리를 반환한다.
        """
        failure_results_dict = {
            "cost": float('inf'), "co2": float('inf'),
            "mean_strength_ratio": float('inf'), "max_strength_ratio": float('inf'),
            "strength_ratios": [], "story_drifts_x": [], "story_drifts_y": [],
            "wind_displacements_x": [], "wind_displacements_y": [],
            "violation_deflection": float('inf'), "violation_drift": float('inf'),
            "violation_hierarchy": float('inf'), "violation_wind_disp": float('inf'),
            "forces_df": pd.DataFrame(), "violation": float('inf'), "margins": {}
        }
        try:
            len_col_sec, len_col_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
            col_indices = individual[:len_col_sec]
            col_rotations = individual[len_col_sec : len_col_sec + len_col_rot]
            beam_indices = individual[len_col_sec + len_col_rot :]
            column_elem_ids, beam_elem_ids, node_map = build_model_for_section(col_indices, col_rotations, beam_indices)
        except Exception:
            return failure_results_dict # 튜플 대신 실패 딕셔너리를 반환
        
        lateral_force_dist_sum = sum(range(1, floors + 1))
        deflection_ratios = []
        for i, elem_id in enumerate(beam_elem_ids):
            beam_len = beam_lengths[i % len(beam_connections)]; group_idx = beam_map.get(num_columns + i + 1, 0)
            sec_idx = beam_indices[group_idx]; h_b = beam_sections[sec_idx][1]
            min_thickness = beam_len / 21.0
            deflection_ratios.append(min_thickness / h_b)
        actual_deflection_ratio = max(deflection_ratios) if deflection_ratios else 0.0

        all_max_combo_forces, analysis_ok = [], True
        ops.timeSeries('Linear',1)
        ops.system('ProfileSPD')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.integrator('LoadControl',1.0)
        ops.algorithm('Linear')
        ops.analysis('Static')
        for i, (combo_name, factors) in enumerate(load_combinations):
            pattern_tag = i + 1
            ops.reset()
            ops.pattern('Plain', pattern_tag, 1)
            superimposed_beam_load = DL * factors["DL"] + LL * factors["LL"]
            for beam_idx, eid in enumerate(beam_elem_ids):
                group_idx = beam_map[num_columns + beam_idx + 1]
                sec_idx = beam_indices[group_idx]
                b, h = beam_sections[sec_idx]
                unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']
                beam_self_weight = b * h * unit_weight
                total_beam_load = beam_self_weight * factors["DL"]
                beam_floor = (beam_idx // len(beam_connections)) + 1; conn_idx = beam_idx % len(beam_connections)
                loaded_beams_for_this_floor = patterns_by_floor.get(beam_floor, set())
                if conn_idx in loaded_beams_for_this_floor: total_beam_load += superimposed_beam_load
                if abs(total_beam_load) > 1e-6:
                    ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0, -total_beam_load)
            for col_idx, eid in enumerate(column_elem_ids):
                group_idx = col_map[col_idx + 1]; sec_idx = col_indices[group_idx]
                b, h = column_sections[sec_idx]; unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']
                col_self_weight = b * h * H * unit_weight; node1_tag, node2_tag = ops.eleNodes(eid)
                ops.load(node1_tag, 0,0, -col_self_weight/2 * factors["DL"], 0,0,0)
                ops.load(node2_tag, 0,0, -col_self_weight/2 * factors["DL"], 0,0,0)
            base_force_x = Wx * factors["Wx"] + Ex * factors["Ex"]
            base_force_y = Wy * factors["Wy"] + Ey * factors["Ey"]
            if abs(base_force_x) > 1e-9 or abs(base_force_y) > 1e-9:
                for k in range(1, floors + 1):
                    floor_multiplier = k / lateral_force_dist_sum; story_force_x = base_force_x * floor_multiplier; story_force_y = base_force_y * floor_multiplier
                    nodal_load_x = story_force_x / num_locations; nodal_load_y = story_force_y / num_locations
                    for loc_idx in range(num_locations):
                        if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], nodal_load_x, nodal_load_y, 0, 0, 0, 0)
            if ops.analyze(1)!=0: analysis_ok = False; ops.remove('loadPattern', pattern_tag); break
            df_max_curr = extract_local_element_forces(column_elem_ids, beam_elem_ids)
            if df_max_curr.empty: analysis_ok = False; ops.remove('loadPattern', pattern_tag); break
            df_max_curr['Combo'] = combo_name # <<< 하중조합 이름 추가
            all_max_combo_forces.append(df_max_curr)
            ops.remove('loadPattern', pattern_tag)

        if not analysis_ok or not all_max_combo_forces:
            return failure_results_dict # 튜플 대신 실패 딕셔너리를 반환
        df_all_combos = pd.concat(all_max_combo_forces, ignore_index=True)
        
        force_cols = ['Axial (kN)', 'Shear-y (kN)', 'Shear-z (kN)', 'Torsion (kNm)', 'Moment-y (kNm)', 'Moment-z (kNm)']
        idx_cols = ['ElementType', 'ElementID']
        max_rows = []
        for _, group in df_all_combos.groupby(idx_cols, observed=True):
            row = {k: group.iloc[0][k] for k in idx_cols}
            for force in force_cols:
                max_idx = group[force].abs().idxmax()
                row[force] = group.loc[max_idx, force]
                # 해당 부재력을 발생시킨 하중조합의 이름을 함께 저장
                row[f'{force}_Combo'] = group.loc[max_idx, 'Combo'] 
            max_rows.append(row)
        final_max_forces = pd.DataFrame(max_rows)

        strength_ratios = []
        final_max_forces_sorted = final_max_forces.sort_values('ElementID').set_index('ElementID')

        all_elements_ids = sorted(list(set(column_elem_ids) | set(beam_elem_ids)))
        for elem_id in all_elements_ids:
            try:
                row, elem_type = final_max_forces_sorted.loc[elem_id], final_max_forces_sorted.loc[elem_id]['ElementType']
                p, vy, vz = abs(row['Axial (kN)']), abs(row['Shear-y (kN)']), abs(row['Shear-z (kN)']);
                my, mz = abs(row['Moment-z (kNm)']), abs(row['Moment-y (kNm)'])
                if elem_type == 'Column':
                    abs_col_idx = column_elem_ids.index(elem_id); group_idx = col_map[abs_col_idx + 1]; sec_idx = col_indices[group_idx]
                    strengths, pm_df = get_precalculated_strength(elem_type, sec_idx, column_sections_df, beam_sections_df), load_pm_data_for_column(h5_file, sec_idx)
                    pn_z, mn_z = get_pm_capacity_from_df(p/(mz+1e-9), pm_df, axis='z')
                    pn_y, mn_y = get_pm_capacity_from_df(p/(my+1e-9), pm_df, axis='y')
                    ratios = [p/(pn_z+1e-9),                 # <-- P-M 상호작용이 고려된 축력 응력비 (강축 기준)
                            p/(pn_y+1e-9),                 # <-- P-M 상호작용이 고려된 축력 응력비 (약축 기준)
                            vy/(strengths['Vn_y']+1e-9),
                            vz/(strengths['Vn_z']+1e-9),
                            my/(mn_y+1e-9),
                            mz/(mn_z+1e-9)]
                else:
                    abs_beam_idx = beam_elem_ids.index(elem_id); group_idx = beam_map[num_columns + abs_beam_idx + 1]; sec_idx = beam_indices[group_idx]
                    strengths = get_precalculated_strength(elem_type, sec_idx, column_sections_df, beam_sections_df)
                    ratios = [vz/(strengths['Vn_z']+1e-9),
                                mz/(strengths['Mn_z']+1e-9)]
                strength_ratios.append(max(r for r in ratios if r is not None and not math.isinf(r) and r >= 0))
            except (KeyError, IndexError): strength_ratios.append(float('inf'))
        max_strength_ratio = max(strength_ratios) if strength_ratios else 1.0
        mean_strength_ratio = np.mean([r for r in strength_ratios if not math.isinf(r)]) if strength_ratios else 0.0

        story_drifts_x, story_drifts_y = [], []; actual_drift_ratio = 0.0
        if analysis_ok:
            allowable_drift_ratio = 0.015
            # --- X방향 층간변위 검토 (최대값 기준) ---
            ops.reset(); ops.pattern('Plain', 101, 1)
            drift_factors_x = next((f for name, f in load_combinations if name == "ASCE-S-E1"), None)
            NODALLOADx, NODALLOADy = Ex*drift_factors_x["Ex"], Ey*drift_factors_x["Ey"]
            for k in range(1,floors+1):
                for loc_idx in range(num_locations):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
            if ops.analyze(1) == 0:
                for k in range(1, floors + 1):
                    max_story_drift_x = 0
                    for loc_idx in range(num_locations):
                        node_upper, node_lower = node_map.get((k, loc_idx)), node_map.get((k - 1, loc_idx))
                        if node_upper and node_lower:
                            drift = abs(ops.nodeDisp(node_upper, 1) - ops.nodeDisp(node_lower, 1)) / H
                            if drift > max_story_drift_x: max_story_drift_x = drift
                    story_drifts_x.append(max_story_drift_x)
            
            # --- Y방향 층간변위 검토 (최대값 기준) ---
            ops.reset(); ops.pattern('Plain', 102, 1)
            drift_factors_y = next((f for name, f in load_combinations if name == "ASCE-S-E5"), None)
            NODALLOADx, NODALLOADy = Ex*drift_factors_y["Ex"], Ey*drift_factors_y["Ey"]
            for k in range(1,floors+1):
                for loc_idx in range(num_locations):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
            if ops.analyze(1) == 0:
                for k in range(1, floors + 1):
                    max_story_drift_y = 0
                    for loc_idx in range(num_locations):
                        node_upper, node_lower = node_map.get((k, loc_idx)), node_map.get((k - 1, loc_idx))
                        if node_upper and node_lower:
                            drift = abs(ops.nodeDisp(node_upper, 2) - ops.nodeDisp(node_lower, 2)) / H
                            if drift > max_story_drift_y: max_story_drift_y = drift
                    story_drifts_y.append(max_story_drift_y)

            max_drift_x = max(story_drifts_x) if story_drifts_x else 0
            max_drift_y = max(story_drifts_y) if story_drifts_y else 0
            actual_drift_ratio = max(max_drift_x, max_drift_y) / allowable_drift_ratio
        else: actual_drift_ratio = float('inf')

        wind_disps_x, wind_disps_y = [], []; actual_wind_disp_ratio = 0.0
        if analysis_ok:
            lateral_force_dist_sum = sum(range(1, floors + 1))
            # --- X 방향 풍하중 변위 검토 (최대값 기준) ---
            actual_wind_disp_ratio_x = float('inf')
            ops.reset(); ops.pattern('Plain', 201, 1)
            wind_factors_x = next((f for name, f in load_combinations if name == "ASCE-S-W1"), None)
            if wind_factors_x:
                base_force_x = Wx * wind_factors_x["Wx"]
                for k in range(1, floors + 1):
                    story_force_x = base_force_x * (k / lateral_force_dist_sum)
                    nodal_load_x = story_force_x / num_locations
                    for loc_idx in range(num_locations):
                        if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], nodal_load_x, 0, 0, 0, 0, 0)
                if ops.analyze(1) == 0:
                    # 각 층의 모든 절점 중 최대 변위(절대값)를 계산
                    disps = [max([abs(ops.nodeDisp(nid, 1)) for (fl, _), nid in node_map.items() if fl == k]) for k in range(1, floors + 1)]
                    wind_disps_x = disps
                    if wind_disps_x:
                        actual_wind_disp_ratio_x = wind_disps_x[-1] / ((floors * H) / 400.0)
            
            # --- Y 방향 풍하중 변위 검토 (최대값 기준) ---
            actual_wind_disp_ratio_y = float('inf')
            ops.reset(); ops.pattern('Plain', 202, 1)
            wind_factors_y = next((f for name, f in load_combinations if name == "ASCE-S-W3"), None)
            if wind_factors_y:
                base_force_y = Wy * wind_factors_y["Wy"]
                for k in range(1, floors + 1):
                    story_force_y = base_force_y * (k / lateral_force_dist_sum)
                    nodal_load_y = story_force_y / num_locations
                    for loc_idx in range(num_locations):
                        if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], 0, nodal_load_y, 0, 0, 0, 0)
                if ops.analyze(1) == 0:
                    # 각 층의 모든 절점 중 최대 변위(절대값)를 계산
                    disps = [max([abs(ops.nodeDisp(nid, 2)) for (fl, _), nid in node_map.items() if fl == k]) for k in range(1, floors + 1)]
                    wind_disps_y = disps
                    if wind_disps_y:
                        actual_wind_disp_ratio_y = wind_disps_y[-1] / ((floors * H) / 400.0)
            actual_wind_disp_ratio = max(actual_wind_disp_ratio_x, actual_wind_disp_ratio_y)

        hierarchy_ratios = [1.0]
        if GROUPING_STRATEGY != "Uniform":
            for k in range(floors - 1): 
                for i in range(num_locations):
                    abs_col_idx_lower = k * num_locations + i; group_idx_lower = col_map[abs_col_idx_lower + 1]; sec_idx_lower = col_indices[group_idx_lower]; rot_lower = col_rotations[group_idx_lower]
                    b_lower, h_lower = column_sections[sec_idx_lower]; dim_x_lower, dim_y_lower = (b_lower, h_lower) if rot_lower == 0 else (h_lower, b_lower)
                    abs_col_idx_upper = (k + 1) * num_locations + i; group_idx_upper = col_map[abs_col_idx_upper + 1]; sec_idx_upper = col_indices[group_idx_upper]; rot_upper = col_rotations[group_idx_upper]
                    b_upper, h_upper = column_sections[sec_idx_upper]; dim_x_upper, dim_y_upper = (b_upper, h_upper) if rot_upper == 0 else (h_upper, b_upper)
                    ratio_x = dim_x_upper / dim_x_lower if dim_x_lower > 0 else 0.0; ratio_y = dim_y_upper / dim_y_lower if dim_y_lower > 0 else 0.0
                    hierarchy_ratios.append(max(ratio_x, ratio_y))
        actual_hierarchy_ratio = max(hierarchy_ratios)

        total_cost, total_co2 = 0, 0
        for i in range(num_columns):
            group_idx = col_map[i + 1]; sec_idx = col_indices[group_idx]
            total_cost += column_sections_df.iloc[sec_idx]['Cost'] * H
            total_co2 += column_sections_df.iloc[sec_idx]['CO2'] * H
        for k in range(floors):
            for i in range(len(beam_connections)):
                abs_beam_idx = k * len(beam_connections) + i; group_idx = beam_map[num_columns + abs_beam_idx + 1]
                sec_idx = beam_indices[group_idx]
                total_cost += beam_sections_df.iloc[sec_idx]['Cost'] * beam_lengths[i]
                total_co2 += beam_sections_df.iloc[sec_idx]['CO2'] * beam_lengths[i]

        # 1. 각 제약조건의 '최대 허용 위반 비율'을 사전에 정의 (위반의 '스케일'을 정의하는 단계)
        max_allowable_ratios = {
            'strength': 2.0,    # DCR은 최대 2.0까지를 위반 범위로 간주 (위반 마진 스케일: 0 ~ 1.0)
            'drift': 2.5,       # 층간변위비는 최대 2.5배까지를 범위로 간주 (위반 마진 스케일: 0 ~ 1.5)
            'wind_disp': 3.0,   # 풍하중 변위는 최대 3.0배까지를 범위로 간주 (위반 마진 스케일: 0 ~ 2.0)
            'deflection': 2.0,  # 보 처짐은 최대 2.0배까지를 범위로 간주 (위반 마진 스케일: 0 ~ 1.0)
            'hierarchy': 1.2    # 기둥 위계는 최대 1.2배까지만을 범위로 간주 (위반 마진 스케일: 0 ~ 0.2)
        }
        # 2. 가중치 설정 (모두 동일하게 1.0으로 설정)
        weights = {
            'strength': 1.0, 'drift': 1.0, 'wind_disp': 1.0, 'deflection': 1.0, 'hierarchy': 1.0
        }
        # 3. 각 위반 마진을 '최대 허용 마진'으로 정규화하여 가중 합산
        margins = {
            'strength': max(0, max_strength_ratio - 1.0),
            'drift': max(0, actual_drift_ratio - 1.0),
            'wind_disp': max(0, actual_wind_disp_ratio - 1.0),
            'deflection': max(0, actual_deflection_ratio - 1.0),
            'hierarchy': max(0, actual_hierarchy_ratio - 1.0)
        }
        total_normalized_violation = 0
        normalized_margins = {}  # 각 제약조건별 정규화 마진을 저장할 딕셔너리 생성
        for key, margin in margins.items():
            # 최대 허용 마진 = (최대 허용 비율 - 1.0)
            max_allowed_margin = max_allowable_ratios[key] - 1.0
            # [핵심] 위반 마진을 최대 허용 마진으로 나누어 0~1 사이의 '진짜 정규화된 위반 값'으로 변환
            normalized_margin = min(1.0, margin / (max_allowed_margin + 1e-9))
            # 가중치 적용하여 합산
            total_normalized_violation += weights[key] * normalized_margin
            normalized_margins[key] = normalized_margin  # 계산된 마진 값을 딕셔너리에 저장

        # 최종 Fitness가 아닌, 계산에 필요한 Raw 값들을 딕셔너리에 담아 반환
        detailed_results_dict = {
            "cost": total_cost, "co2": total_co2,
            "mean_strength_ratio": mean_strength_ratio,
            "violation": total_normalized_violation, # 페널티 계산을 위한 위반량
            "normalized_margins": normalized_margins, # <<< 딕셔너리 전체를 저장하고 키 이름을 복수형으로 변경
            "absolute_margins": margins, # <<< 절대 위반량(실제 마진값) 추가
            # 이하 상세 결과는 그대로 유지
            "max_strength_ratio": max_strength_ratio, "strength_ratios": strength_ratios,
            "story_drifts_x": story_drifts_x, "story_drifts_y": story_drifts_y,
            "wind_displacements_x": wind_disps_x, "wind_displacements_y": wind_disps_y,
            "violation_deflection": actual_deflection_ratio, "violation_drift": actual_drift_ratio,
            "violation_hierarchy": actual_hierarchy_ratio, "violation_wind_disp": actual_wind_disp_ratio,
            "forces_df": final_max_forces
        }
        
        return detailed_results_dict

    # =================================================================
    # ===              4. DEAP 최적화 실행 함수                      ===
    # =================================================================

    def run_ga_optimization(DL, LL, Wx, Wy, Ex, Ey, crossover_method, patterns_by_floor, h5_file,
                            num_generations, population_size,
                            initial_pop=None, start_gen=0, logbook=None, hof=None, hof_stats_history=None):
        """
        DEAP 라이브러리를 사용하여 NSGA-II 다중목표 유전 알고리즘을 설정하고 실행하는 함수.
        (제약조건 우선 원칙 적용 최종 버전)
        """
        # --- 1. 제약조건 우선 선택 함수 정의 ---
        def constrained_dominance_selection(individuals, k):
            """제약조건 우선 원칙(Constraint-Dominance Principle)을 적용하는 선택 함수."""
            feasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] == 0.0]
            infeasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] > 0.0]
            
            # 유효해 그룹 내에서 NSGA-II 선택을 먼저 수행
            selected_from_feasible = tools.selNSGA2(feasible_inds, len(feasible_inds))
            
            # 다음 세대를 구성할 리스트
            next_generation = selected_from_feasible
            
            # 유효해만으로 k개를 채우지 못했다면, 위반해로 나머지를 채움
            if len(next_generation) < k:
                num_needed = k - len(next_generation)
                # 위반량이 적은 순으로 위반해들을 정렬
                infeasible_inds.sort(key=lambda ind: ind.detailed_results['violation'])
                # 가장 덜 위반한 해들로 나머지 자리를 채움
                next_generation.extend(infeasible_inds[:num_needed])
                
            return next_generation[:k] # 최종적으로 k개만 반환

        # --- 2. Fitness 계산 헬퍼 함수 정의 ---
        def _assign_fitness(population):
            """주어진 인구집단에 대해 페널티 없는 Fitness를 계산하고 할당"""
            for ind in population:
                res = ind.detailed_results
                if res['cost'] == float('inf'):
                    ind.fitness.values = (float('inf'), float('inf'))
                    continue
                
                # 고정 스케일로 정규화
                norm_cost = max(0.0, min(1.0, (res['cost'] - FIXED_MIN_COST) / FIXED_RANGE_COST))
                norm_co2 = max(0.0, min(1.0, (res['co2'] - FIXED_MIN_CO2) / FIXED_RANGE_CO2))
                
                # 페널티 없이 목표함수 값 할당
                obj1 = norm_cost + norm_co2
                obj2 = res['mean_strength_ratio']

                ind.fitness.values = (obj1, obj2 if obj2 > 0 else float('inf'))

        # --- 3. DEAP Toolbox 설정 ---
        # 2목표: (정규화된 Cost+CO2) 최소화, (평균 응력비) 최대화
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        num_col_opts, num_beam_opts = len(column_sections), len(beam_sections)
        gene_pool = [lambda: random.randint(0, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
        gene_pool.extend(lambda: random.randint(0, 1) for _ in range(chromosome_structure['col_rot']))
        gene_pool.extend(lambda: random.randint(0, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec']))

        toolbox.register("individual", tools.initCycle, creator.Individual, tuple(gene_pool))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        h5_file = h5_file
        toolbox.register("evaluate", evaluate, DL=DL, LL=LL, Wx=Wx, Wy=Wy, Ex=Ex, Ey=Ey, h5_file=h5_file, patterns_by_floor=patterns_by_floor)
        toolbox.register("select", constrained_dominance_selection)
        
        if crossover_method == 'Uniform':
            toolbox.register("mate", tools.cxUniform, indpb=0.5)
        elif crossover_method == 'OnePoint':
            toolbox.register("mate", tools.cxOnePoint)
        else:
            toolbox.register("mate", tools.cxTwoPoint)

        def custom_mutate(individual, indpb):
            for i in range(len(individual)):
                if random.random() < indpb:
                    if i < chromosome_structure['col_sec']:
                        individual[i] = random.randint(0, num_col_opts - 1)
                    elif i < chromosome_structure['col_sec'] + chromosome_structure['col_rot']:
                        individual[i] = random.randint(0, 1)
                    else:
                        individual[i] = random.randint(0, num_beam_opts - 1)
            return individual,
        toolbox.register("mutate", custom_mutate, indpb=0.1)
        toolbox.register("select_offspring", tools.selTournament, tournsize=7)
        
        # 통계 및 로그북 설정
        def get_valid_ratio(population):
            """제약조건을 모두 만족하는 해의 비율을 계산하는 함수"""
            valid_count = sum(1 for ind in population if (
                hasattr(ind, 'detailed_results') and
                # 'violation' 값이 0이면 모든 제약조건을 만족한 유효해
                ind.detailed_results.get('violation', float('inf')) == 0.0
            ))
            return valid_count / len(population) * 100 if population else 0.0
        def get_analysis_success_ratio(population):
            """해석에 성공한 해의 비율을 계산하는 함수"""
            if not population:
                return 0.0
            success_count = sum(1 for ind in population if 
                                hasattr(ind, 'detailed_results') and 
                                ind.detailed_results.get('cost') != float('inf'))
            return success_count / len(population) * 100
        def get_best_invalid_margins(population):
            """
            위반량이 가장 적은 개체를 찾아 해당 개체의 상세 위반 내역을 문자열로 반환.
            (S: Strength, D: Drift, W: Wind, F: deFlection, H: Hierarchy)
            """
            if not population:
                return "N/A"

            # detailed_results가 있고 'violation' 키가 있는 개체들만 필터링
            valid_for_check = [ind for ind in population if hasattr(ind, 'detailed_results') and 'violation' in ind.detailed_results]

            if not valid_for_check:
                return "No detailed results"

            # 위반량이 가장 적은 개체를 찾음
            best_ind = min(valid_for_check, key=lambda ind: ind.detailed_results['violation'])

            margins = best_ind.detailed_results.get('absolute_margins')
            if not margins:
                return "Margins N/A"

            # 각 위반 항목을 축약된 문자열로 포매팅
            margin_str = (
                f"S:{margins.get('strength', 0):.2f} "
                f"D:{margins.get('drift', 0):.2f} "
                f"W:{margins.get('wind_disp', 0):.2f} "
                f"F:{margins.get('deflection', 0):.2f} "
                f"H:{margins.get('hierarchy', 0):.2f}"
            )
            return margin_str
        # --- 4. 최적화 루프 실행 ---
        # 통계 헬퍼 함수 정의 (유효해만 필터링하여 계산)
        def calculate_valid_stat(pop, key, stat_func, default_val=0.0):
            """
            유효해(violation == 0)만 필터링하여 특정 값(key)에 대한 통계(stat_func)를 계산합니다.
            유효해가 없을 경우 default_val을 반환합니다.
            """
            valid_values = [
                ind.detailed_results[key] for ind in pop
                if hasattr(ind, 'detailed_results') and
                ind.detailed_results.get('violation') == 0.0 and
                key in ind.detailed_results
            ]
            return stat_func(valid_values) if valid_values else default_val
        def calculate_margin_stat(pop, margin_dict_key, margin_key, stat_func, default_val=float('inf')):
            """특정 마진(dict_key/key)에 대한 통계(stat_func)를 계산하는 헬퍼 함수"""
            margin_values = []
            for ind in pop:
                if hasattr(ind, 'detailed_results') and isinstance(ind.detailed_results, dict):
                    margins_dict = ind.detailed_results.get(margin_dict_key, {})
                    val = margins_dict.get(margin_key)
                    if val is not None: margin_values.append(val)
            
            valid_values = [v for v in margin_values if v != float('inf')]
            return stat_func(valid_values) if valid_values else default_val
        fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        # --- 통계 객체 재구성 (전체 항목) ---
        # 그룹 1: Fitness 통계 (전체)
        fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        fitness_stats.register("avg", np.mean, axis=0)
        fitness_stats.register("max", np.max, axis=0)
        fitness_stats.register("min", np.min, axis=0)
        fitness_stats.register("std", np.std, axis=0)

        # 그룹 2: 모집단 상태 통계
        health_stats = tools.Statistics()
        health_stats.register("success_rate", get_analysis_success_ratio)
        health_stats.register("valid_ratio", get_valid_ratio)

        # 그룹 3: 유효해 값 통계 (전체)
        value_stats = tools.Statistics()
        value_stats.register("avg_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.mean))
        value_stats.register("max_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.max))
        value_stats.register("min_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.min))
        value_stats.register("std_cost", lambda pop: calculate_valid_stat(pop, 'cost', np.std))
        value_stats.register("avg_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.mean))
        value_stats.register("max_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.max))
        value_stats.register("min_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.min))
        value_stats.register("std_co2", lambda pop: calculate_valid_stat(pop, 'co2', np.std))
        value_stats.register("avg_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.mean))
        value_stats.register("max_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.max))
        value_stats.register("min_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.min))
        value_stats.register("std_dcr", lambda pop: calculate_valid_stat(pop, 'mean_strength_ratio', np.std))

        # 그룹 4: 위반량 상세 통계 (전체)
        margin_stats = tools.Statistics()
        margin_stats.register("best_margins", get_best_invalid_margins)
        margin_keys = ['strength', 'drift', 'wind_disp', 'deflection', 'hierarchy']
        margin_abbrs = ['str', 'drift', 'wind', 'defl', 'hier']
        for key, abbr in zip(margin_keys, margin_abbrs):
            margin_stats.register(f"min_AM_{abbr}", lambda pop, k=key: calculate_margin_stat(pop, 'absolute_margins', k, np.min, default_val=0.0))
        margin_stats.register("min_NM_str", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'strength', np.min))
        margin_stats.register("min_NM_drift", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'drift', np.min))
        margin_stats.register("min_NM_wind", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'wind_disp', np.min))
        margin_stats.register("min_NM_defl", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'deflection', np.min))
        margin_stats.register("min_NM_hier", lambda pop: calculate_margin_stat(pop, 'normalized_margins', 'hierarchy', np.min))

        # 그룹 5: Hall of Fame 통계 (유효해 대상)
        hof_value_stats = tools.Statistics()
        hof_value_stats.register("hof_avg_cost", lambda h: calculate_valid_stat(h, 'cost', np.mean))
        hof_value_stats.register("hof_max_cost", lambda h: calculate_valid_stat(h, 'cost', np.max))
        hof_value_stats.register("hof_min_cost", lambda h: calculate_valid_stat(h, 'cost', np.min))
        hof_value_stats.register("hof_std_cost", lambda h: calculate_valid_stat(h, 'cost', np.std))
        hof_value_stats.register("hof_avg_co2", lambda h: calculate_valid_stat(h, 'co2', np.mean))
        hof_value_stats.register("hof_max_co2", lambda h: calculate_valid_stat(h, 'co2', np.max))
        hof_value_stats.register("hof_min_co2", lambda h: calculate_valid_stat(h, 'co2', np.min))
        hof_value_stats.register("hof_std_co2", lambda h: calculate_valid_stat(h, 'co2', np.std))
        hof_value_stats.register("hof_avg_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.mean))
        hof_value_stats.register("hof_max_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.max))
        hof_value_stats.register("hof_min_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.min))
        hof_value_stats.register("hof_std_dcr", lambda h: calculate_valid_stat(h, 'mean_strength_ratio', np.std))

        # --- 로그북 헤더 재구성 ---
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + fitness_stats.fields + \
                         ['sep1'] + health_stats.fields + \
                         ['sep2'] + value_stats.fields + \
                         ['sep3'] + margin_stats.fields + \
                         ['sep4', 'hof_size'] + \
                         ['sep5'] + hof_value_stats.fields

        if initial_pop is None:
            # --- 최초 실행 시 초기화 ---
            pop = toolbox.population(n=population_size)
            hof = tools.ParetoFront()
            hof_stats_history = []
            
            print("\n초기 집단 평가 중...")
            eval_results = []
            for ind in tqdm(pop, desc="Initial Population Evaluation", unit="individual"):
                eval_results.append(toolbox.evaluate(ind))
            for ind, res in zip(pop, eval_results):
                ind.detailed_results = res
            _assign_fitness(pop)
            
            feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
            hof.update(feasible_pop)
            if hof:
                best_obj1 = min(ind.fitness.values[0] for ind in hof)
                best_obj2 = min(ind.fitness.values[1] for ind in hof)
                hof_stats_history.append({'gen': 0, 'best_obj1': best_obj1, 'best_obj2': best_obj2})

            record = fitness_stats.compile(pop)
            record.update(health_stats.compile(pop))
            record.update(value_stats.compile(pop))
            record.update(margin_stats.compile(pop))
            record.update(hof_value_stats.compile(hof))
            record['hof_size'] = len(hof)
            record['sep1'], record['sep2'], record['sep3'], record['sep4'], record['sep5'] = "|", "|", "|", "|", "|"
            logbook.record(gen=0, nevals=len(pop), **record)
            
            print("최적화 시작...")
            print(logbook.stream)
        else:
            # --- 이어하기 시 상태 복원 ---
            pop = initial_pop
            print(f"\n이전 {start_gen} 세대에서 최적화를 계속합니다...")

        # 메인 진화 루프
        for gen in tqdm(range(start_gen + 1, start_gen + num_generations + 1), desc="세대 진화"):
            offspring = toolbox.select_offspring(pop, len(pop))
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            
            eval_results = []
            if invalid_ind:
                for ind in tqdm(invalid_ind, desc=f"Gen {gen} Evaluation", unit="ind", leave=False):
                    eval_results.append(toolbox.evaluate(ind))
            for ind, res in zip(invalid_ind, eval_results):
                ind.detailed_results = res
                
            _assign_fitness(offspring)
            
            pop = toolbox.select(pop + offspring, k=population_size)
            
            feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
            hof.update(feasible_pop)
            if hof:
                best_obj1 = min(ind.fitness.values[0] for ind in hof)
                best_obj2 = min(ind.fitness.values[1] for ind in hof)
                hof_stats_history.append({'gen': gen, 'best_obj1': best_obj1, 'best_obj2': best_obj2})
            record = fitness_stats.compile(pop)
            record.update(health_stats.compile(pop))
            record.update(value_stats.compile(pop))
            record.update(margin_stats.compile(pop))
            record.update(hof_value_stats.compile(hof))
            record['hof_size'] = len(hof)
            record['sep1'], record['sep2'], record['sep3'], record['sep4'], record['sep5'] = "|", "|", "|", "|", "|"
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            tqdm.write(logbook.stream.splitlines()[-1])

        
        return pop, logbook, hof, hof_stats_history






















# =================================================================
# ===                    5. 메인 실행 블록                        ===
# =================================================================
# if __name__ == '__main__':
    
    # output_folder = f"03_Optimization_Result_{GROUPING_STRATEGY}_{CROSSOVER_STRATEGY}"
    output_folder = f"Results_main_하중증가,모집단{POPULATION_SIZE}_06_200세대"
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
        DL=DL_rand, LL=LL_rand, Wx=Wx_rand, Wy=Wy_rand, Ex=Ex_rand, Ey=Ey_rand, 
        crossover_method=CROSSOVER_STRATEGY, patterns_by_floor=patterns_by_floor, h5_file=h5_file,
        num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
        # 최초 실행이므로 아래 인자들은 기본값(None)을 사용
        initial_pop=None, start_gen=0, logbook=None, hof=None, hof_stats_history=None
    )
    
    # --- 추가 세대 진행 여부 확인 루프 ---
    while True:
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
                DL=DL_rand, LL=LL_rand, Wx=Wx_rand, Wy=Wy_rand, Ex=Ex_rand, Ey=Ey_rand, 
                crossover_method=CROSSOVER_STRATEGY, 
                patterns_by_floor=patterns_by_floor, 
                h5_file=h5_file,
                num_generations=additional_gens,      # 추가 세대 수
                population_size=POPULATION_SIZE,      # 인구 크기는 동일
                initial_pop=population,               # 이전 결과 전달
                start_gen=last_gen_num,               # 마지막 세대부터 시작
                logbook=logbook,                      # 로그북 이어쓰기
                hof=pareto_front,                     # HOF 이어쓰기
                hof_stats_history=hof_stats_history   # HOF 통계 이력 이어쓰기
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

    if not valid_solutions:
        print("모든 제약조건을 만족하는 해를 찾지 못했습니다.")
    else:
        # 정렬 기준을 첫 번째 목표값(피트니스)으로 변경
        sorted_pareto = sorted(valid_solutions, key=lambda ind: ind.fitness.values[0])
        all_results = []
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
        
        # =================================================================
        # ===             추가: 최적화 결과 데이터 CSV 저장             ===
        # =================================================================
        print("\n\n" + "="*90)
        print(f"### 최적화 결과 데이터를 '{output_folder}' 폴더에 CSV 파일로 저장합니다. ###")

        # 1. 파레토 최적해 요약 (pareto_summary.csv)
        summary_df = pd.DataFrame(summary_data_list)
        summary_df.to_csv(os.path.join(output_folder, "pareto_summary.csv"), index=False)
        print("- 'pareto_summary.csv' 저장 완료.")

        # 2. 최적화 과정 로그 (optimization_log.csv)
        log_df = pd.DataFrame(logbook)
        log_df = log_df.drop(columns=[col for col in log_df.columns if 'sep' in str(col)], errors='ignore')
        log_df.to_csv(os.path.join(output_folder, "optimization_log.csv"), index=False)
        print("- 'optimization_log.csv' 저장 완료.")

        # Hall of Fame 수렴도 데이터 저장 (hof_convergence.csv)
        if hof_stats_history:
            hof_df = pd.DataFrame(hof_stats_history)
            hof_df.to_csv(os.path.join(output_folder, "hof_convergence.csv"), index=False)
            print("- 'hof_convergence.csv' 저장 완료.")

        # 3. 최적해별 설계 변수 (design_variables.csv)
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

        # 4. 최적해별 변위/변형 검토 결과 (displacement_checks_all.csv)
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

        # 5. 최적해별 부재 DCR (dcr_by_element.csv)
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
            print(f"{'Floor':<10} {'Drift-X (m)':<15} {'Drift-Y (m)':<15} {'Drift Ratio-X':<15} {'Drift Ratio-Y':<15} {'Result'}")
            allowable_drift_ratio = 0.015
            drifts_x_ratio = result.get('story_drifts_x', []); drifts_y_ratio = result.get('story_drifts_y', [])
            for i in range(floors):
                drift_x_r = drifts_x_ratio[i] if i < len(drifts_x_ratio) else 0.0
                drift_y_r = drifts_y_ratio[i] if i < len(drifts_y_ratio) else 0.0
                drift_x_m = drift_x_r * H; drift_y_m = drift_y_r * H
                status = "OK" if drift_x_r <= allowable_drift_ratio and drift_y_r <= allowable_drift_ratio else "NG"
                print(f"{'Story ' + str(i+1):<10} {drift_x_m:<15.5f} {drift_y_m:<15.5f} {drift_x_r:<15.5f} {drift_y_r:<15.5f} {status}")

            print("\n[풍하중 층별 변위]")
            print(f"{'Floor':<10} {'Disp-X (m)':<15} {'Disp-Y (m)':<15} {'Allowable (m)':<15} {'Result'}")
            disps_x = result.get('wind_displacements_x', []); disps_y = result.get('wind_displacements_y', [])
            for i in range(floors):
                disp_x = disps_x[i] if i < len(disps_x) else 0.0; disp_y = disps_y[i] if i < len(disps_y) else 0.0
                allowable_disp = ((i + 1) * H) / 400.0
                status = "OK" if disp_x <= allowable_disp and disp_y <= allowable_disp else "NG"
                print(f"{'Story ' + str(i+1):<10} {disp_x:<15.5f} {disp_y:<15.5f} {allowable_disp:<15.5f} {status}")
        
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
            
            column_elem_ids, beam_elem_ids, _ = build_model_for_section(col_indices, col_rotations, beam_indices)
            
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
        build_model_for_section(all_results[0]['individual'][:chromosome_structure['col_sec']], all_results[0]['individual'][chromosome_structure['col_sec']:chromosome_structure['col_sec']+chromosome_structure['col_rot']], all_results[0]['individual'][chromosome_structure['col_sec']+chromosome_structure['col_rot']:])
        
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

if __name__ == '__main__':
    main()
