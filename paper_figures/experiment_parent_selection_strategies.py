def main():
    # =================================================================
    # ===                  라이브러리 임포트                          ===
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
    plt.rcParams['font.family'] = 'Times New Roman' # 글꼴 설정
    from deap import base, creator, tools, algorithms
    from scipy.spatial import ConvexHull

    # =================================================================
    # ===              1. 최적화 및 모델링 주요 설정                  ===
    # =================================================================

    # --- 1.1. 실험 및 그룹핑 전략 설정 ---
    GROUPING_STRATEGY = "Hybrid" 
    CROSSOVER_STRATEGY = "OnePoint" # 교배 전략은 TwoPoint으로 고정
    # <<< 실험할 부모 선택 전략 리스트 >>>
    # parent_selection_to_test = [
    #     {'name': 'Random_Selection',    'func': tools.selRandom,       'params': {}},
    #     {'name': 'Tournament_Size_2',   'func': tools.selTournament,   'params': {'tournsize': 2}},
    #     {'name': 'Tournament_Size_3',   'func': tools.selTournament,   'params': {'tournsize': 3}},
    #     {'name': 'Tournament_Size_5',   'func': tools.selTournament,   'params': {'tournsize': 5}},
    #     {'name': 'Tournament_Size_7',   'func': tools.selTournament,   'params': {'tournsize': 7}},
    #     {'name': 'Best_Selection',      'func': tools.selBest,         'params': {}}
    # ]
    parent_selection_to_test = [
        {'name': 'Tournament_Size_9',   'func': tools.selTournament,   'params': {'tournsize': 9}},
        {'name': 'Tournament_Size_11',   'func': tools.selTournament,   'params': {'tournsize': 11}}
    ]

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

    # --- 1.4. 유전 알고리즘 파라미터 ---
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 100
    CXPB, MUTPB = 0.8, 0.2

    # --- 1.5. 부재 단면 정보 로드 ---
    data_dir = "C:\\Users\\82105\\OneDrive\\Desktop\\ERS\[논문작성]\\RL_RCframe_비용최적설계\\Python\\05 Frame해석"
    beam_sections_df = pd.read_csv(os.path.join(data_dir, "beam_sections_simple02.csv") )
    column_sections_df = pd.read_csv(os.path.join(data_dir, "column_sections_simple02.csv") )
    h5_file = h5py.File(os.path.join(data_dir, 'pm_dataset_simple02.mat'), 'r') # 데이터로드용

    # --- 1.6. DEAP Fitness 및 Individual 생성 (전역) ---
    # 여러번 호출될 경우 에러가 발생하므로 main 함수 시작점에 한번만 정의
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except Exception:
        pass

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
        # --- ASCE 7 사용성 검토용 하중조합 ---
        ("ASCE-S-W1", {"DL": 1.0, "LL": 0.5, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-W2", {"DL": 1.0, "LL": 0.5, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-W3", {"DL": 1.0, "LL": 0.5, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
        ("ASCE-S-W4", {"DL": 1.0, "LL": 0.5, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
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
    num_locations = len(column_locations)
    num_columns = num_locations * floors
    num_beams = len(beam_connections) * floors
    col_map, beam_map = {}, {}

    if GROUPING_STRATEGY == "Hybrid":
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
        hull_edges = set(tuple(sorted(s)) for s in hull.simplices)
        perimeter_beam_indices = {i for i, (p1, p2) in enumerate(beam_connections) if tuple(sorted((p1, p2))) in hull_edges}

        # 4. 최종 그룹 ID 할당
        floor_step = 2
        num_floor_groups = math.ceil(floors / floor_step)
        num_col_groups = num_floor_groups * 3
        num_beam_groups = num_floor_groups * 2
        for abs_col_idx in range(num_columns):
            floor_idx, loc_idx = divmod(abs_col_idx, num_locations)
            floor_group_idx = floor_idx // floor_step
            loc_type = get_col_loc_type(loc_idx)
            group_id = floor_group_idx * 3 + loc_type
            col_map[abs_col_idx + 1] = group_id
        for abs_beam_idx in range(num_beams):
            floor_idx, conn_idx = divmod(abs_beam_idx, len(beam_connections))
            floor_group_idx = floor_idx // floor_step
            loc_type = 0 if conn_idx in perimeter_beam_indices else 1
            group_id = floor_group_idx * 2 + loc_type
            beam_map[num_columns + abs_beam_idx + 1] = group_id

    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
    column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
    beam_lengths = [math.sqrt((column_locations[p2][0] - column_locations[p1][0])**2 + (column_locations[p2][1] - column_locations[p1][1])**2) for p1, p2 in beam_connections]

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

    def load_pm_data_for_column(h5_file, column_index):
        try:
            ref = h5_file['Column_Mdata'][0, column_index]; pm_data = h5_file[ref][()]
            return pd.DataFrame(pm_data, columns=['Mnb_z','Pnb_z','Mnb_y','Pnb_y','Alpha_PI_Mnb_z','Alpha_PI_Pnb_z','Alpha_PI_Mnb_y','Alpha_PI_Pnb_y'])
        except Exception: return pd.DataFrame()

    def get_pm_capacity_from_df(slope, pm_df, axis='z'):
        if pm_df.empty: return 0.0, 0.0
        moment_col, axial_col = (f'Alpha_PI_Mnb_{axis}', f'Alpha_PI_Pnb_{axis}')
        design_m, design_p = pm_df[moment_col].values, pm_df[axial_col].values
        curve_slopes = abs(design_p) / (abs(design_m) + 1e-9)
        for i in range(len(curve_slopes) - 1):
            if curve_slopes[i] >= slope >= curve_slopes[i+1]:
                m1, p1, m2, p2 = design_m[i], design_p[i], design_m[i+1], design_p[i+1]
                if abs(m2 - m1) < 1e-9: continue
                a1 = (p2 - p1) / (m2 - m1); b1 = p1 - a1 * m1
                if abs(a1 - slope) < 1e-9: continue
                Mn = -b1 / (a1 - slope)
                Pn = slope * Mn
                return Pn, Mn
        return 0.0, 0.0

    def get_precalculated_strength(element_type, index, col_df, beam_df):
        strengths = {}
        try:
            if element_type == 'Column':
                row = col_df.iloc[index]; strengths.update({'Pn': row['PI_Pn_max'], 'Vn_y': row['PI_Vn_y'], 'Vn_z': row['PI_Vn_z']})
            elif element_type == 'Beam':
                row = beam_df.iloc[index]; strengths.update({'Pn': float('inf'), 'Vn_y': float('inf'), 'Vn_z': row['PiVn'], 'Mn_y': float('inf'), 'Mn_z': row['PiM']})
        except IndexError: return {'Pn': 0, 'Vn_y': 0, 'Vn_z': 0, 'Mn_y': 0, 'Mn_z': 0}
        return strengths

    def extract_local_element_forces(column_elem_ids, beam_elem_ids):
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

    # =================================================================
    # ===               3. DEAP 평가 함수 (Fitness Function)          ===
    # =================================================================
    def evaluate(individual, DL, LL, Wx, Wy, Ex, Ey, h5_file, patterns_by_floor):
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
            return failure_results_dict
        
        lateral_force_dist_sum = sum(range(1, floors + 1))
        deflection_ratios = []
        for i, eid in enumerate(beam_elem_ids):
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
            df_max_curr['Combo'] = combo_name
            all_max_combo_forces.append(df_max_curr)
            ops.remove('loadPattern', pattern_tag)

        if not analysis_ok or not all_max_combo_forces:
            return failure_results_dict
        df_all_combos = pd.concat(all_max_combo_forces, ignore_index=True)
        
        force_cols = ['Axial (kN)', 'Shear-y (kN)', 'Shear-z (kN)', 'Torsion (kNm)', 'Moment-y (kNm)', 'Moment-z (kNm)']
        idx_cols = ['ElementType', 'ElementID']
        max_rows = []
        for _, group in df_all_combos.groupby(idx_cols, observed=True):
            row = {k: group.iloc[0][k] for k in idx_cols}
            for force in force_cols:
                max_idx = group[force].abs().idxmax()
                row[force] = group.loc[max_idx, force]
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
                    ratios = [p/(pn_z+1e-9), p/(pn_y+1e-9), vy/(strengths['Vn_y']+1e-9), vz/(strengths['Vn_z']+1e-9), my/(mn_y+1e-9), mz/(mn_z+1e-9)]
                else:
                    abs_beam_idx = beam_elem_ids.index(elem_id); group_idx = beam_map[num_columns + abs_beam_idx + 1]; sec_idx = beam_indices[group_idx]
                    strengths = get_precalculated_strength(elem_type, sec_idx, column_sections_df, beam_sections_df)
                    ratios = [vz/(strengths['Vn_z']+1e-9), mz/(strengths['Mn_z']+1e-9)]
                strength_ratios.append(max(r for r in ratios if r is not None and not math.isinf(r) and r >= 0))
            except (KeyError, IndexError): strength_ratios.append(float('inf'))
        max_strength_ratio = max(strength_ratios) if strength_ratios else 1.0
        mean_strength_ratio = np.mean([r for r in strength_ratios if not math.isinf(r)]) if strength_ratios else 0.0

        story_drifts_x, story_drifts_y = [], []; actual_drift_ratio = 0.0
        if analysis_ok:
            allowable_drift_ratio = 0.015
            ops.reset(); ops.pattern('Plain', 101, 1); drift_factors_x = next((f for name, f in load_combinations if name == "ASCE-S-E1"), None); NODALLOADx, NODALLOADy = Ex*drift_factors_x["Ex"], Ey*drift_factors_x["Ey"]
            for k in range(1,floors+1):
                for loc_idx in range(num_locations):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
            if ops.analyze(1) == 0:
                floor_disps_x = [0.0]*(floors+1)
                for k in range(1,floors+1): floor_disps_x[k] = np.mean([ops.nodeDisp(nid,1) for(fl,li),nid in node_map.items() if fl==k])
                for k in range(1,floors+1): story_drifts_x.append(abs(floor_disps_x[k]-floor_disps_x[k-1])/H)
            ops.reset(); ops.pattern('Plain', 102, 1); drift_factors_y = next((f for name, f in load_combinations if name == "ASCE-S-E5"), None); NODALLOADx, NODALLOADy = Ex*drift_factors_y["Ex"], Ey*drift_factors_y["Ey"]
            for k in range(1,floors+1):
                for loc_idx in range(num_locations):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
            if ops.analyze(1) == 0:
                floor_disps_y = [0.0]*(floors+1);
                for k in range(1,floors+1): floor_disps_y[k] = np.mean([ops.nodeDisp(nid,2) for(fl,li),nid in node_map.items() if fl==k])
                for k in range(1,floors+1): story_drifts_y.append(abs(floor_disps_y[k]-floor_disps_y[k-1])/H)
            max_drift_x = max(story_drifts_x) if story_drifts_x else 0
            max_drift_y = max(story_drifts_y) if story_drifts_y else 0
            actual_drift_ratio = max(max_drift_x, max_drift_y) / allowable_drift_ratio
        else: actual_drift_ratio = float('inf')

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

        wind_disps_x, wind_disps_y = [], []; actual_wind_disp_ratio = 0.0
        if analysis_ok:
            lateral_force_dist_sum = sum(range(1, floors + 1))
            actual_wind_disp_ratio_x = float('inf')
            ops.reset(); ops.pattern('Plain', 201, 1)
            wind_factors_x = next((f for name, f in load_combinations if name == "ASCE-S-W1"), None)
            if wind_factors_x:
                base_force_x = Wx * wind_factors_x["Wx"]
                for k in range(1, floors + 1):
                    floor_multiplier = k / lateral_force_dist_sum
                    story_force_x = base_force_x * floor_multiplier
                    nodal_load_x = story_force_x / num_locations
                    for loc_idx in range(num_locations):
                        if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], nodal_load_x, 0, 0, 0, 0, 0)
                if ops.analyze(1) == 0:
                    disps = [abs(np.mean([ops.nodeDisp(nid, 1) for (fl, _), nid in node_map.items() if fl==k])) for k in range(1, floors + 1)]
                    wind_disps_x = disps
                    if wind_disps_x:
                        actual_wind_disp_ratio_x = wind_disps_x[-1] / ((floors * H) / 400.0)
            actual_wind_disp_ratio_y = float('inf')
            ops.reset(); ops.pattern('Plain', 202, 1)
            wind_factors_y = next((f for name, f in load_combinations if name == "ASCE-S-W3"), None)
            if wind_factors_y:
                base_force_y = Wy * wind_factors_y["Wy"]
                for k in range(1, floors + 1):
                    floor_multiplier = k / lateral_force_dist_sum
                    story_force_y = base_force_y * floor_multiplier
                    nodal_load_y = story_force_y / num_locations
                    for loc_idx in range(num_locations):
                        if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], 0, nodal_load_y, 0, 0, 0, 0)
                if ops.analyze(1) == 0:
                    disps = [abs(np.mean([ops.nodeDisp(nid, 2) for (fl, _), nid in node_map.items() if fl==k])) for k in range(1, floors + 1)]
                    wind_disps_y = disps
                    if wind_disps_y:
                        actual_wind_disp_ratio_y = wind_disps_y[-1] / ((floors * H) / 400.0)
            actual_wind_disp_ratio = max(actual_wind_disp_ratio_x, actual_wind_disp_ratio_y)

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

        max_allowable_ratios = {
            'strength': 2.0, 'drift': 2.5, 'wind_disp': 3.0, 'deflection': 2.0, 'hierarchy': 1.2
        }
        weights = {
            'strength': 1.0, 'drift': 1.0, 'wind_disp': 1.0, 'deflection': 1.0, 'hierarchy': 1.0
        }
        margins = {
            'strength': max(0, max_strength_ratio - 1.0),
            'drift': max(0, actual_drift_ratio - 1.0),
            'wind_disp': max(0, actual_wind_disp_ratio - 1.0),
            'deflection': max(0, actual_deflection_ratio - 1.0),
            'hierarchy': max(0, actual_hierarchy_ratio - 1.0)
        }
        total_normalized_violation = 0
        normalized_margins = {}
        for key, margin in margins.items():
            max_allowed_margin = max_allowable_ratios[key] - 1.0
            normalized_margin = min(1.0, margin / (max_allowed_margin + 1e-9))
            total_normalized_violation += weights[key] * normalized_margin
            normalized_margins[key] = normalized_margin

        detailed_results_dict = {
            "cost": total_cost, "co2": total_co2,
            "mean_strength_ratio": mean_strength_ratio,
            "violation": total_normalized_violation,
            "normalized_margins": normalized_margins,
            "absolute_margins": margins,
            "max_strength_ratio": max_strength_ratio, "strength_ratios": strength_ratios,
            "story_drifts_x": story_drifts_x, "story_drifts_y": story_drifts_y,
            "wind_displacements_x": wind_disps_x, "wind_displacements_y": wind_disps_y,
            "violation_deflection": actual_deflection_ratio, "violation_drift": actual_drift_ratio,
            "violation_hierarchy": actual_hierarchy_ratio, "violation_wind_disp": actual_wind_disp_ratio,
            "forces_df": final_max_forces
        }
        
        return detailed_results_dict

    # =================================================================
    # ===               4. DEAP 최적화 실행 함수                      ===
    # =================================================================

    def run_ga_optimization(POPULATION_SIZE, NUM_GENERATIONS, DL, LL, Wx, Wy, Ex, Ey, crossover_method, parent_selection_config, patterns_by_floor, h5_file):
        def constrained_dominance_selection(individuals, k):
            feasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] == 0.0]
            infeasible_inds = [ind for ind in individuals if ind.detailed_results['violation'] > 0.0]
            selected_from_feasible = tools.selNSGA2(feasible_inds, len(feasible_inds))
            next_generation = selected_from_feasible
            if len(next_generation) < k:
                num_needed = k - len(next_generation)
                infeasible_inds.sort(key=lambda ind: ind.detailed_results['violation'])
                next_generation.extend(infeasible_inds[:num_needed])
            return next_generation[:k]

        def _assign_fitness(population):
            for ind in population:
                res = ind.detailed_results
                if res['cost'] == float('inf'):
                    ind.fitness.values = (float('inf'), float('inf'))
                    continue
                norm_cost = max(0.0, min(1.0, (res['cost'] - FIXED_MIN_COST) / FIXED_RANGE_COST))
                norm_co2 = max(0.0, min(1.0, (res['co2'] - FIXED_MIN_CO2) / FIXED_RANGE_CO2))
                obj1 = norm_cost + norm_co2
                obj2 = res['mean_strength_ratio']
                ind.fitness.values = (obj1, obj2 if obj2 > 0 else float('inf'))

        toolbox = base.Toolbox()
        num_col_opts, num_beam_opts = len(column_sections), len(beam_sections)
        gene_pool = [lambda: random.randint(0, num_col_opts - 1) for _ in range(chromosome_structure['col_sec'])]
        gene_pool.extend(lambda: random.randint(0, 1) for _ in range(chromosome_structure['col_rot']))
        gene_pool.extend(lambda: random.randint(0, num_beam_opts - 1) for _ in range(chromosome_structure['beam_sec']))

        toolbox.register("individual", tools.initCycle, creator.Individual, tuple(gene_pool))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
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
        toolbox.register("select_offspring", parent_selection_config['func'], **parent_selection_config.get('params', {}))
        
        def get_valid_ratio(population):
            valid_count = sum(1 for ind in population if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation', float('inf')) == 0.0)
            return valid_count / len(population) * 100 if population else 0.0
        
        def get_valid_stats(pop, key):
            valid_values = [ind.detailed_results[key] for ind in pop if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0 and key in ind.detailed_results]
            return np.mean(valid_values) if valid_values else 0.0
        
        fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        fitness_stats.register("avg", np.mean, axis=0)
        fitness_stats.register("std", np.std, axis=0)
        fitness_stats.register("min", np.min, axis=0)
        
        population_stats = tools.Statistics()
        population_stats.register("valid_ratio", get_valid_ratio)
        population_stats.register("avg_cost", lambda pop: get_valid_stats(pop, 'cost'))
        population_stats.register("avg_co2", lambda pop: get_valid_stats(pop, 'co2'))
        population_stats.register("avg_dcr", lambda pop: get_valid_stats(pop, 'mean_strength_ratio'))
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + fitness_stats.fields + population_stats.fields + ['hof_size']
        
        pop = toolbox.population(n=POPULATION_SIZE)
        
        print("\n초기 집단 평가 중...")
        eval_results = []
        for ind in tqdm(pop, desc="Initial Population Evaluation", unit="individual", leave=False):
            eval_results.append(toolbox.evaluate(ind))
        for ind, res in zip(pop, eval_results):
            ind.detailed_results = res
        _assign_fitness(pop)
        
        hof = tools.ParetoFront()
        feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
        hof.update(feasible_pop)
        hof_stats_history = []
        if hof:
                best_obj1 = min(ind.fitness.values[0] for ind in hof)
                best_obj2 = min(ind.fitness.values[1] for ind in hof)
                hof_stats_history.append({'gen': 0, 'best_obj1': best_obj1, 'best_obj2': best_obj2})
        
        record = fitness_stats.compile(pop)
        record.update(population_stats.compile(pop))
        record['hof_size'] = len(hof)
        logbook.record(gen=0, nevals=len(pop), **record)
        
        print("최적화 시작...")
        header_str = (
            f"{'Gen':>4s} | {'Nevals':>6s} | {'HOF':>4s} | {'Valid':>7s} | "
            f"{'Avg Cost':>12s} | {'Avg CO2':>12s} | {'Avg DCR':>8s} | "
            f"{'Min Fitness':>20s} | {'Avg Fitness':>20s}"
        )
        print(header_str)
        print("-" * len(header_str))

        log_entry = logbook[0]
        log_str_0 = (
            f"{log_entry['gen']:4d} | {log_entry['nevals']:6d} | {log_entry['hof_size']:4d} | {log_entry['valid_ratio']:6.2f}% | "
            f"{log_entry['avg_cost']:12.2f} | {log_entry['avg_co2']:12.2f} | {log_entry['avg_dcr']:8.4f} | "
            f"[{log_entry['min'][0]:8.4f}, {log_entry['min'][1]:8.4f}]"
        )
        if 'avg' in log_entry and hasattr(log_entry['avg'], '__len__') and len(log_entry['avg']) >= 2:
            log_str_0 += f" | [{log_entry['avg'][0]:8.4f}, {log_entry['avg'][1]:8.4f}]"
        print(log_str_0)

        for gen in tqdm(range(1, NUM_GENERATIONS + 1), desc=f"Evolving - {parent_selection_config['name']}"):
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
            pop = toolbox.select(pop + offspring, k=POPULATION_SIZE)
            
            feasible_pop = [ind for ind in pop if ind.detailed_results.get('violation') == 0.0]
            hof.update(feasible_pop)
            if hof:
                best_obj1 = min(ind.fitness.values[0] for ind in hof)
                best_obj2 = min(ind.fitness.values[1] for ind in hof)
                hof_stats_history.append({'gen': gen, 'best_obj1': best_obj1, 'best_obj2': best_obj2})
            
            record = fitness_stats.compile(pop)
            record.update(population_stats.compile(pop))
            record['hof_size'] = len(hof)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            log_entry = logbook[-1]
            log_str = (
                f"{log_entry['gen']:4d} | {log_entry['nevals']:6d} | {log_entry['hof_size']:4d} | {log_entry['valid_ratio']:6.2f}% | "
                f"{log_entry['avg_cost']:12.2f} | {log_entry['avg_co2']:12.2f} | {log_entry['avg_dcr']:8.4f} | "
                f"[{log_entry['min'][0]:8.4f}, {log_entry['min'][1]:8.4f}]"
            )
            if 'avg' in log_entry and hasattr(log_entry['avg'], '__len__') and len(log_entry['avg']) >= 2:
                log_str += f" | [{log_entry['avg'][0]:8.4f}, {log_entry['avg'][1]:8.4f}]"
            tqdm.write(log_str)
        
        return pop, logbook, hof, hof_stats_history

    # =================================================================
    # ===                    5. 메인 실행 블록                        ===
    # =================================================================
    
    results_by_ps = {}
    for exp_config in tqdm(parent_selection_to_test, desc="Parent Selection Strategy Experiments"):
        
        RANDOM_SEED = 42 
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        DL_rand=int(np.random.uniform(15, 25))
        LL_rand=int(np.random.uniform(10, 20))
        Wx_rand=int(np.random.uniform(50, 75))
        Wy_rand=int(np.random.uniform(50, 75))
        Ex_rand=int(np.random.uniform(50, 75))
        Ey_rand=int(np.random.uniform(50, 75))

        # patterns_by_floor = {}
        # unique_x = sorted(list(set(loc[0] for loc in column_locations))); unique_y = sorted(list(set(loc[1] for loc in column_locations)))
        # for floor_num in range(1, floors + 1):
        #     loaded_beams_indices = set()
        #     if len(unique_x) > 1 and len(unique_y) > 1:
        #         x_indices = sorted(random.sample(range(len(unique_x)), 2)); y_indices = sorted(random.sample(range(len(unique_y)), 2))
        #         x_min, x_max = unique_x[x_indices[0]], unique_x[x_indices[1]]; y_min, y_max = unique_y[y_indices[0]], unique_y[y_indices[1]]
        #         locs_in_rect = {i for i, (x, y) in enumerate(column_locations) if x_min <= x <= x_max and y_min <= y <= y_max}
        #         for i, (p1_idx, p2_idx) in enumerate(beam_connections):
        #             if p1_idx in locs_in_rect and p2_idx in locs_in_rect: loaded_beams_indices.add(i)
        #     patterns_by_floor[floor_num] = sorted(list(loaded_beams_indices))
        
        patterns_by_floor = {
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20],
            2: [1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            3: [3, 4, 6, 7, 9, 12, 14, 15, 17, 18],
            4: [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 19, 20]
        }
        
        output_folder = f"paper_figures/parent_selection_{exp_config['name']}_02"
        os.makedirs(output_folder, exist_ok=True)
        print(f"\n결과는 '{output_folder}' 폴더에 저장됩니다.")
        print("\n" + "="*90)
        print(f"### 최적화 실행 ({exp_config['name']}) ###")
        print(f"- 그룹핑: {GROUPING_STRATEGY}, 교배: {CROSSOVER_STRATEGY}, 부모선택: {exp_config['name']}")
        print(f"- 인구: {POPULATION_SIZE}, 세대: {NUM_GENERATIONS}, 교배/변이율: {CXPB}/{MUTPB}")
        print(f"- 하중: DL={DL_rand}, LL={LL_rand}, Wx={Wx_rand}, Wy={Wy_rand}, Ex={Ex_rand}, Ey={Ey_rand}")
        print("="*90)

        population, logbook, pareto_front, hof_stats_history = run_ga_optimization(
            POPULATION_SIZE=POPULATION_SIZE,
            NUM_GENERATIONS=NUM_GENERATIONS,
            DL=DL_rand, LL=LL_rand, Wx=Wx_rand, Wy=Wy_rand, Ex=Ex_rand, Ey=Ey_rand, 
            crossover_method=CROSSOVER_STRATEGY,
            parent_selection_config=exp_config,
            patterns_by_floor=patterns_by_floor,
            h5_file=h5_file
        )
        
        print("\n" + "="*80)
        print(f"### 최적화 완료 ({exp_config['name']}) ###")
        
        valid_solutions = [ind for ind in pareto_front if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0]
        print(f"총 {len(valid_solutions)}개의 유효한 파레토 최적해를 찾았습니다.")
        print("="*80)

        if not valid_solutions:
            print("모든 제약조건을 만족하는 해를 찾지 못했습니다.")
        else:
            sorted_pareto = sorted(valid_solutions, key=lambda ind: ind.fitness.values[0])
            all_results = []
            for i, ind in enumerate(sorted_pareto):
                result_dict = ind.detailed_results.copy()
                result_dict['ID'] = f"Sol #{i+1}"
                result_dict['ind_object'] = ind
                all_results.append(result_dict)

            if 'all_results' in locals() and all_results:
                results_by_ps[exp_config['name']] = {
                    'logbook': logbook,
                    'pareto_front_results': all_results,
                    'hof_stats': hof_stats_history
                }

            # --- Save plot data to CSV files ---
            print(f"\n### {exp_config['name']} 전략 결과 데이터 CSV 저장 ###")

            # 1. Pareto Front Data
            pareto_data_for_csv = []
            for r in all_results:
                # Create a flat dictionary for CSV, excluding non-serializable objects
                flat_res = {
                    'ID': r['ID'],
                    'cost': r['cost'],
                    'co2': r['co2'],
                    'fitness1_obj': r['ind_object'].fitness.values[0],
                    'fitness2_obj': r['ind_object'].fitness.values[1],
                    'mean_strength_ratio': r['mean_strength_ratio'],
                    'max_strength_ratio': r['max_strength_ratio'],
                    'violation': r['violation'],
                    'violation_deflection': r['violation_deflection'],
                    'violation_drift': r['violation_drift'],
                    'violation_hierarchy': r['violation_hierarchy'],
                    'violation_wind_disp': r['violation_wind_disp'],
                    'individual_genes': str(list(r['ind_object']))
                }
                pareto_data_for_csv.append(flat_res)

            if pareto_data_for_csv:
                df_pareto = pd.DataFrame(pareto_data_for_csv)
                pareto_csv_path = os.path.join(output_folder, "pareto_front_data.csv")
                df_pareto.to_csv(pareto_csv_path, index=False, encoding='utf-8-sig')
                print(f"Pareto front 데이터가 '{pareto_csv_path}'에 저장되었습니다.")

            # 2. Convergence Data (HOF)
            if hof_stats_history:
                df_hof = pd.DataFrame(hof_stats_history)
                hof_csv_path = os.path.join(output_folder, "convergence_hof_data.csv")
                df_hof.to_csv(hof_csv_path, index=False, encoding='utf-8-sig')
                print(f"수렴 과정(HOF) 데이터가 '{hof_csv_path}'에 저장되었습니다.")

            # 3. Logbook Data
            if logbook:
                df_logbook = pd.DataFrame(logbook)
                logbook_csv_path = os.path.join(output_folder, "logbook_data.csv")
                df_logbook.to_csv(logbook_csv_path, index=False, encoding='utf-8-sig')
                print(f"세대별 통계(Logbook) 데이터가 '{logbook_csv_path}'에 저장되었습니다.")

            print(f"\n### {exp_config['name']} 전략 결과 시각화 ###")
            
            fig_analysis, axs = plt.subplots(1, 4, figsize=(35, 7))
            fig_analysis.suptitle(f'Optimization Process Analysis ({exp_config["name"]})', fontsize=20)
            colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
            fitness1_vals = [r['ind_object'].fitness.values[0] for r in all_results]
            fitness2_vals = [r['ind_object'].fitness.values[1] for r in all_results]
            axs[0].scatter(fitness2_vals, fitness1_vals, c=colors, s=80, edgecolors='k', alpha=0.8)
            axs[0].set_title('Pareto Front in Objective Space'); axs[0].set_xlabel('Fitness2 (Mean DCR)'); axs[0].set_ylabel('Fitness1 (Norm Cost+CO2)'); axs[0].grid(True)
            real_costs = [r['cost'] for r in all_results]; real_co2s = [r['co2'] for r in all_results]
            sc = axs[1].scatter(real_co2s, real_costs, c=fitness2_vals, cmap='viridis', s=80, edgecolors='k', alpha=0.8)
            fig_analysis.colorbar(sc, ax=axs[1]).set_label('Mean DCR')
            axs[1].set_title('Pareto Solutions in Real Space'); axs[1].set_xlabel('Total CO2'); axs[1].set_ylabel('Total Cost'); axs[1].grid(True)
            ax_conv = axs[2]
            gen_hof = [s['gen'] for s in hof_stats_history]; hof_best_obj1 = [s['best_obj1'] for s in hof_stats_history]; hof_best_obj2 = [s['best_obj2'] for s in hof_stats_history]
            ax_conv.plot(gen_hof, hof_best_obj1, color='tab:blue', marker='o', linestyle='-', label="Best Fitness1")
            ax_conv.set_xlabel("Generation"); ax_conv.set_ylabel("Fitness1 (Norm Cost+CO2)", color='tab:blue'); ax_conv.tick_params(axis='y', labelcolor='tab:blue'); ax_conv.grid(True)
            ax_conv_twin = ax_conv.twinx()
            ax_conv_twin.plot(gen_hof, hof_best_obj2, color='tab:red', marker='s', linestyle='-', label="Best Fitness2")
            ax_conv_twin.set_ylabel("Fitness2 (Mean DCR)", color='tab:red'); ax_conv.set_title('Convergence of Hall of Fame')
            gen_log = logbook.select("gen"); valid_ratios_log = logbook.select("valid_ratio")
            axs[3].plot(gen_log, valid_ratios_log, marker='o', linestyle='-', color='g')
            axs[3].set_title("Valid Solutions Ratio"); axs[3].set_xlabel("Generation"); axs[3].set_ylabel("Valid Solutions (%)"); axs[3].set_ylim(0, 105); axs[3].grid(True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_folder, "analysis_summary.png")); plt.close()

        print(f"\nCOMPLETED RUN for {exp_config['name']}. Results saved in '{output_folder}'")

    # =================================================================
    # ===                 7. 최종 비교 분석 및 시각화                 ===
    # =================================================================
    if results_by_ps:
        print("\n" + "="*80)
        print("### Generating Final Comparison Plots ###")
        output_folder2 = "paper_figures"
        os.makedirs(output_folder2, exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(36, 7))
        fig.suptitle('Parent Selection Method Performance Comparison', fontsize=20)
        
        ax_conv1, ax_conv2, ax_pareto_obj, ax_pareto_real = axes

        style_map = {
            'Random_Selection':    {'color': 'C0', 'marker': 'o', 'linestyle': '-'}, 
            'Tournament_Size_2':   {'color': 'C1', 'marker': 's', 'linestyle': '--'}, 
            'Tournament_Size_3':   {'color': 'C2', 'marker': 'P', 'linestyle': ':'}, 
            'Tournament_Size_5':   {'color': 'C3', 'marker': 'D', 'linestyle': '-.'}, 
            'Tournament_Size_7':   {'color': 'C4', 'marker': 'v', 'linestyle': '-'}, 
            'Best_Selection':      {'color': 'C5', 'marker': '*', 'linestyle': '--'}
        }

        for method, results in results_by_ps.items():
            style = style_map.get(method, {'color': 'gray', 'marker': 'x', 'linestyle': '-.'})

            hof_stats = results.get('hof_stats')
            if hof_stats:
                gen = [s['gen'] for s in hof_stats]
                best_obj1 = [s['best_obj1'] for s in hof_stats]
                best_obj2 = [s['best_obj2'] for s in hof_stats]
                ax_conv1.plot(gen, best_obj1, label=method, alpha=0.9, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], markersize=4)
                ax_conv2.plot(gen, best_obj2, label=method, alpha=0.9, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], markersize=4)

            pareto_results = results.get('pareto_front_results')
            if pareto_results:
                fitness1_vals = [r['ind_object'].fitness.values[0] for r in pareto_results]
                fitness2_vals = [r['ind_object'].fitness.values[1] for r in pareto_results]
                ax_pareto_obj.scatter(fitness2_vals, fitness1_vals, s=80, edgecolors='k', alpha=0.7, label=method, color=style['color'], marker=style['marker'])
                costs = [r['cost'] for r in pareto_results]
                co2s = [r['co2'] for r in pareto_results]
                ax_pareto_real.scatter(costs, co2s, s=80, edgecolors='k', alpha=0.7, label=method, color=style['color'], marker=style['marker'])

        ax_conv1.set_title('Hof Conv. - Objective 1 (Cost+CO2)')
        ax_conv1.set_xlabel('Generation'); ax_conv1.set_ylabel('Best Fitness1 in Hof'); ax_conv1.legend(); ax_conv1.grid(True)

        ax_conv2.set_title('Hof Conv. - Objective 2 (Mean DCR)')
        ax_conv2.set_xlabel('Generation'); ax_conv2.set_ylabel('Best Fitness2 in Hof'); ax_conv2.legend(); ax_conv2.grid(True)

        ax_pareto_obj.set_title('Final Pareto Front (Objective Space)')
        ax_pareto_obj.set_xlabel('Fitness2 (Mean DCR)'); ax_pareto_obj.set_ylabel('Fitness1 (Normalized Cost + CO2)'); ax_pareto_obj.legend(); ax_pareto_obj.grid(True)

        ax_pareto_real.set_title('Final Pareto Front (Real Space: Cost vs CO2)')
        ax_pareto_real.set_xlabel('Total Cost'); ax_pareto_real.set_ylabel('Total CO2'); ax_pareto_real.legend(); ax_pareto_real.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(output_folder2, "comparison_summary_parent_selection.png"))
        print("Comparison summary plot saved.")
        plt.show()

    h5_file.close()

if __name__ == '__main__':
    main()
