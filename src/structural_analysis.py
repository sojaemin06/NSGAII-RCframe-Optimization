
import openseespy.opensees as ops
import pandas as pd
import numpy as np
import math
from src.config import *
from src.utils import *

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

def build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections):
    """주어진 유전자 정보를 바탕으로 OpenSees에서 3D 골조 모델을 생성하는 함수."""
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    E, nu = 2.5791e7, 0.167; G = E / (2 * (1 + nu))
    node_map = {}; node_id_counter = 1
    for k in range(FLOORS + 1):
        for i, (x, y) in enumerate(COLUMN_LOCATIONS):
            ops.node(node_id_counter, x, y, k * H)
            node_map[(k, i)] = node_id_counter
            if k == 0: ops.fix(node_id_counter, 1, 1, 1, 1, 1, 1)
            node_id_counter += 1
    ops.geomTransf('PDelta', 1, 1, 0, 0); ops.geomTransf('PDelta', 2, 0, 1, 0); ops.geomTransf('PDelta', 3, 0, 0, 1)
    column_elem_ids, beam_elem_ids = [], []; elem_id_counter = 1
    num_locations = len(COLUMN_LOCATIONS)
    num_columns = num_locations * FLOORS
    
    for k in range(FLOORS):
        for i in range(num_locations):
            abs_col_idx = k * num_locations + i; group_idx = col_map[abs_col_idx + 1]
            rotation_flag = col_rotations[group_idx]; transf_tag = 2 if rotation_flag == 1 else 1
            sec_idx = col_indices[group_idx]; b_c, h_c = column_sections[sec_idx]
            
            # [수정] 유효 강성 적용 (Effective Stiffness) - ACI 318
            # 기둥: 0.7 Ig
            A_c = b_c * h_c
            Iz_c = 0.7 * (b_c * h_c**3) / 12 
            Iy_c = 0.7 * (h_c * b_c**3) / 12
            J_c = (Iy_c + Iz_c) # 비틀림 상수는 저감 여부 논란 있으나 보통 유지하거나 0.5적용. 여기선 일단 유지하되 I기반이라 자동 저감됨. (엄밀히는 J는 형상계수라 별도이나 근사적으로)
            # J_c를 단순히 Iy+Iz로 근사하는 건 원형이나 정사각형에서만 유효하나 여기선 약산으로 유지. 
            # 단, 균열 비틀림 강성은 매우 작아질 수 있으므로 보수적으로 유지.
            
            n1, n2 = node_map[(k, i)], node_map[(k + 1, i)]
            ops.element('elasticBeamColumn', elem_id_counter, n1, n2, A_c, E, G, J_c, Iy_c, Iz_c, transf_tag)
            column_elem_ids.append(elem_id_counter); elem_id_counter += 1
    for k in range(1, FLOORS + 1):
        for i, (loc_idx1, loc_idx2) in enumerate(BEAM_CONNECTIONS):
            abs_beam_idx = (k - 1) * len(BEAM_CONNECTIONS) + i; group_idx = beam_map[num_columns + abs_beam_idx + 1]
            sec_idx = beam_indices[group_idx]; b_b, h_b = beam_sections[sec_idx]
            
            # [수정] 유효 강성 적용 (Effective Stiffness) - ACI 318
            # 보: 0.35 Ig
            A_b = b_b * h_b
            Iz_b = 0.35 * (b_b * h_b**3) / 12
            Iy_b = 0.35 * (h_b * b_b**3) / 12
            J_b = (Iy_b + Iz_b)

            n1, n2 = node_map[(k, loc_idx1)], node_map[(k, loc_idx2)]
            ops.element('elasticBeamColumn', elem_id_counter, n1, n2, A_b, E, G, J_b, Iy_b, Iz_b, 3)
            beam_elem_ids.append(elem_id_counter); elem_id_counter += 1
    return column_elem_ids, beam_elem_ids, node_map

def evaluate(individual, DL, LL, Wx, Wy, Ex, Ey, h5_file, patterns_by_floor, 
             col_map, beam_map, beam_sections, column_sections, 
             beam_sections_df, column_sections_df, beam_lengths, 
             chromosome_structure, num_columns, num_beams):
    """
    유전 알고리즘의 핵심 평가 함수.
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
        column_elem_ids, beam_elem_ids, node_map = build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections)
    except Exception:
        return failure_results_dict 
    
    lateral_force_dist_sum = sum(range(1, FLOORS + 1))
    num_locations = len(COLUMN_LOCATIONS)
    deflection_ratios = []
    for i, elem_id in enumerate(beam_elem_ids):
        beam_len = beam_lengths[i % len(BEAM_CONNECTIONS)]; group_idx = beam_map.get(num_columns + i + 1, 0)
        sec_idx = beam_indices[group_idx]; h_b = beam_sections[sec_idx][1]
        min_thickness = beam_len / 21.0
        deflection_ratios.append(min_thickness / h_b)
    actual_deflection_ratio = max(deflection_ratios) if deflection_ratios else 0.0

    # 1. 자중(Self-weight) 및 고정 하중(Dead Load) 계산
    total_structure_weight = 0.0 # 총 지진 중량 W
    story_weights = [0.0] * FLOORS # 각 층의 무게 (지진력 분배용 Wx)
    story_heights_from_base = [(k + 1) * H for k in range(FLOORS)] # 각 층 높이 (최하층부터)

    # 기둥 자중
    # 기둥은 절반씩 위아래 층에 분배된다고 가정
    for col_idx in range(num_columns):
        floor_idx = col_idx // num_locations # 0-indexed floor for column
        group_idx = col_map[col_idx + 1]; sec_idx = col_indices[group_idx]
        b, h = column_sections[sec_idx]; unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']
        col_weight_per_story = b * h * H * unit_weight # 한 층 기둥의 무게
        
        total_structure_weight += col_weight_per_story
        
        # 기둥 무게를 해당 층 (상부 절반)과 그 아래 층 (하부 절반)에 분배
        story_weights[floor_idx] += col_weight_per_story # 일단 해당 층에 전체 할당. 나중에 층 중량 결정 시 재조정.

    # 보 자중 및 슬래브 고정 하중 (지진력 산정용 W에 포함)
    for beam_idx in range(num_beams):
        floor_idx = beam_idx // len(BEAM_CONNECTIONS) # 0-indexed floor for beam
        group_idx = beam_map[num_columns + beam_idx + 1]; sec_idx = beam_indices[group_idx]
        b, h = beam_sections[sec_idx]; unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']
        beam_len = beam_lengths[beam_idx % len(BEAM_CONNECTIONS)]
        beam_weight = b * h * beam_len * unit_weight # 보 자중
        
        total_structure_weight += beam_weight
        story_weights[floor_idx] += beam_weight

    # 슬래브 자중 (층 면적당 하중)
    # 각 층의 슬래브 무게를 총 중량과 층 중량에 추가
    slab_total_weight_per_floor = FLOOR_AREA * SLAB_DL_KN_M2
    total_structure_weight += slab_total_weight_per_floor * FLOORS
    for floor_idx in range(FLOORS):
        story_weights[floor_idx] += slab_total_weight_per_floor
    
    # DL_RAND (추가 고정 하중)은 보 선하중이므로 지진력 산정용 W에 포함
    # 기존 DL은 보 선하중 형태로 이미 evaluate 함수의 인자로 들어와서 재하되므로,
    # 지진 중량 W에는 층별로 해당 DL이 재하되는 보의 길이를 곱하여 추가.
    total_dl_rand_on_beams = 0
    for beam_len in beam_lengths:
        total_dl_rand_on_beams += DL * beam_len * FLOORS # 모든 보에 DL이 재하된다고 가정
    total_structure_weight += total_dl_rand_on_beams


    # 2. 건물 총 높이 H_n (최상층 높이)
    H_total_structure = FLOORS * H

    # 3. 고유 주기 Ta 계산 (ASCE 7-16 Eq. 12.8-7)
    Ta = PERIOD_CT * (H_total_structure ** PERIOD_X)
    
    # 4. 지진 응답 계수 Cs 계산 및 제한 (ASCE 7-16 12.8.1)
    Cs_denom = R_COEFF / I_FACTOR
    Cs_initial = SDS / Cs_denom # 초기 Cs = SDS / (R/Ie) (ASCE 7-16 Eq. 12.8-2)

    # Cs 상한 (ASCE 7-16 Eq. 12.8-3)
    # Cs_upper = SD1 / (Ta * (R_COEFF / I_FACTOR))
    # T > TL (Long-period transition period) 일 경우 다른 상한 적용되나, 여기선 Ta < TL 가정
    Cs_upper = SD1 / (Ta * Cs_denom) if Ta != 0 else float('inf') # Ta가 0이 아닐 때만 계산

    Cs = min(Cs_initial, Cs_upper) # Cs 상한 적용

    # Cs 하한 (ASCE 7-16 Eq. 12.8-5)
    Cs_lower_1 = 0.01
    Cs = max(Cs, Cs_lower_1)

    # 추가 하한 (ASCE 7-16 Eq. 12.8-6) - SDS >= 0.1g 일 때
    if SDS >= 0.1:
        Cs_lower_2 = 0.044 * SDS * I_FACTOR
        Cs = max(Cs, Cs_lower_2)
    
    # 5. 베이스 전단력 V 계산 (ASCE 7-16 Eq. 12.8-1)
    base_shear_force_seismic = Cs * total_structure_weight
    
    # 6. 층별 지진력 분배 (ASCE 7-16 12.8.3)
    # 층별 분포 지수 k 결정 (ASCE 7-16 12.8.3.2)
    k_exponent = 1.0
    if Ta <= 0.5:
        k_exponent = 1.0
    elif Ta >= 2.5:
        k_exponent = 2.0
    else: # 0.5 < Ta < 2.5, 선형 보간
        k_exponent = 1.0 + (Ta - 0.5) / 2.0

    sum_w_h_k = sum(story_weights[f] * (story_heights_from_base[f] ** k_exponent) for f in range(FLOORS))
    
    story_seismic_forces = [0.0] * FLOORS
    if sum_w_h_k > 1e-9: # 0으로 나누는 것 방지
        for f in range(FLOORS):
            Cvx = (story_weights[f] * (story_heights_from_base[f] ** k_exponent)) / sum_w_h_k
            story_seismic_forces[f] = Cvx * base_shear_force_seismic
            
    # OpenSees 해석 부분 (기존 코드 유지)
    all_max_combo_forces, analysis_ok = [], True
    ops.timeSeries('Linear',1)
    ops.system('ProfileSPD')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('LoadControl',1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    for i, (combo_name, factors) in enumerate(LOAD_COMBINATIONS):
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
            beam_floor = (beam_idx // len(BEAM_CONNECTIONS)) + 1; conn_idx = beam_idx % len(BEAM_CONNECTIONS)
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
        
        # [수정] 풍하중 상세 산정 (ASCE 7-16)
        # 각 층별 높이에 따른 풍압(qz) 계산 및 하중 분배
        story_wind_forces_x = [0.0] * FLOORS
        story_wind_forces_y = [0.0] * FLOORS
        
        # 기준 높이 H에서의 풍압 q_h (풍하측용)
        def get_Kz(z):
            z_eff = max(z, 4.57) # 최소 높이 15ft (4.57m)
            return 2.01 * ((z_eff / ZG) ** (2 / ALPHA))
            
        Kz_top = get_Kz(FLOORS * H)
        qh = 0.000613 * Kz_top * KZT * KD * (BASIC_WIND_SPEED ** 2) # kN/m2
        
        for k in range(FLOORS):
            z_story = (k + 1) * H
            Kz = get_Kz(z_story)
            qz = 0.000613 * Kz * KZT * KD * (BASIC_WIND_SPEED ** 2) # kN/m2
            
            # 풍압력 p = q * G * Cp
            # Total Force = (p_windward + p_leeward) * Area
            p_windward = qz * G_FACTOR * CP_WINDWARD
            p_leeward = qh * G_FACTOR * abs(CP_LEEWARD) # Leeward는 qh 기준, 흡입력이므로 절대값 더함
            p_total = p_windward + p_leeward
            
            # 수압 면적 (층고 H * 폭)
            A_x = BUILDING_WIDTH_Y * H # X방향 풍하중을 받는 면적 (Y축 폭)
            A_y = BUILDING_WIDTH_X * H # Y방향 풍하중을 받는 면적 (X축 폭)
            
            story_wind_forces_x[k] = p_total * A_x
            story_wind_forces_y[k] = p_total * A_y

        # 지진 하중은 층별 분배 로직 적용
        # factors["Ex"]가 1.0이면 X방향 지진력, factors["Ey"]가 1.0이면 Y방향 지진력 적용
        # Ex, Ey는 1.0 또는 0.3 (직교 효과) 또는 0
        
        # 층별 지진력 및 풍하중 적용
        if abs(factors["Ex"]) > 1e-9 or abs(factors["Ey"]) > 1e-9 or abs(factors["Wx"]) > 1e-9 or abs(factors["Wy"]) > 1e-9:
            for k in range(FLOORS): # 0-indexed floor
                # 지진력
                Fx_seismic = story_seismic_forces[k] * factors["Ex"]
                Fy_seismic = story_seismic_forces[k] * factors["Ey"]
                
                # 풍하중
                Fx_wind = story_wind_forces_x[k] * factors["Wx"]
                Fy_wind = story_wind_forces_y[k] * factors["Wy"]
                
                # 합산 (지진력 + 풍하중) - 둘 중 하나만 factor가 1이고 나머지는 0일 것임
                Fx_total = Fx_seismic + Fx_wind
                Fy_total = Fy_seismic + Fy_wind
                
                # 각 층의 기둥 수로 나눠 절점당 하중으로 변환
                nodal_load_x = Fx_total / num_locations
                nodal_load_y = Fy_total / num_locations

                # 해당 층의 모든 기둥 상단 절점에 하중 재하
                for loc_idx in range(num_locations):
                    node_tag = node_map.get((k + 1, loc_idx)) # 0-indexed floor -> (k+1)-th floor node
                    if node_tag:
                        ops.load(node_tag, nodal_load_x, nodal_load_y, 0, 0, 0, 0)

        # 기존 base_force_x, base_force_y 계산 로직 대체 (이미 위에서 처리됨)
        # pass
        
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
        # --- X방향 층간변위 ---
        ops.reset(); ops.pattern('Plain', 101, 1)
        drift_factors_x = next((f for name, f in LOAD_COMBINATIONS if name == "ASCE-S-E1"), None)
        NODALLOADx, NODALLOADy = Ex*drift_factors_x["Ex"], Ey*drift_factors_x["Ey"]
        for k in range(1,FLOORS+1):
            for loc_idx in range(num_locations):
                if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
        if ops.analyze(1) == 0:
            for k in range(1, FLOORS + 1):
                max_story_drift_x = 0
                for loc_idx in range(num_locations):
                    node_upper, node_lower = node_map.get((k, loc_idx)), node_map.get((k - 1, loc_idx))
                    if node_upper and node_lower:
                        drift = abs(ops.nodeDisp(node_upper, 1) - ops.nodeDisp(node_lower, 1)) / H
                        if drift > max_story_drift_x: max_story_drift_x = drift
                story_drifts_x.append(max_story_drift_x)
        
        # --- Y방향 층간변위 ---
        ops.reset(); ops.pattern('Plain', 102, 1)
        drift_factors_y = next((f for name, f in LOAD_COMBINATIONS if name == "ASCE-S-E5"), None)
        NODALLOADx, NODALLOADy = Ex*drift_factors_y["Ex"], Ey*drift_factors_y["Ey"]
        for k in range(1,FLOORS+1):
            for loc_idx in range(num_locations):
                if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
        if ops.analyze(1) == 0:
            for k in range(1, FLOORS + 1):
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
        # --- X 방향 풍하중 변위 ---
        actual_wind_disp_ratio_x = float('inf')
        ops.reset(); ops.pattern('Plain', 201, 1)
        wind_factors_x = next((f for name, f in LOAD_COMBINATIONS if name == "ASCE-S-W1"), None)
        if wind_factors_x:
            base_force_x = Wx * wind_factors_x["Wx"]
            for k in range(1, FLOORS + 1):
                story_force_x = base_force_x * (k / lateral_force_dist_sum)
                nodal_load_x = story_force_x / num_locations
                for loc_idx in range(num_locations):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], nodal_load_x, 0, 0, 0, 0, 0)
            if ops.analyze(1) == 0:
                disps = [max([abs(ops.nodeDisp(nid, 1)) for (fl, _), nid in node_map.items() if fl == k]) for k in range(1, FLOORS + 1)]
                wind_disps_x = disps
                if wind_disps_x:
                    actual_wind_disp_ratio_x = wind_disps_x[-1] / ((FLOORS * H) / 400.0)
        
        # --- Y 방향 풍하중 변위 ---
        actual_wind_disp_ratio_y = float('inf')
        ops.reset(); ops.pattern('Plain', 202, 1)
        wind_factors_y = next((f for name, f in LOAD_COMBINATIONS if name == "ASCE-S-W3"), None)
        if wind_factors_y:
            base_force_y = Wy * wind_factors_y["Wy"]
            for k in range(1, FLOORS + 1):
                story_force_y = base_force_y * (k / lateral_force_dist_sum)
                nodal_load_y = story_force_y / num_locations
                for loc_idx in range(num_locations):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], 0, nodal_load_y, 0, 0, 0, 0)
            if ops.analyze(1) == 0:
                disps = [max([abs(ops.nodeDisp(nid, 2)) for (fl, _), nid in node_map.items() if fl == k]) for k in range(1, FLOORS + 1)]
                wind_disps_y = disps
                if wind_disps_y:
                    actual_wind_disp_ratio_y = wind_disps_y[-1] / ((FLOORS * H) / 400.0)
        actual_wind_disp_ratio = max(actual_wind_disp_ratio_x, actual_wind_disp_ratio_y)

    hierarchy_ratios = [1.0]
    if GROUPING_STRATEGY != "Uniform":
        for k in range(FLOORS - 1): 
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
    for k in range(FLOORS):
        for i in range(len(BEAM_CONNECTIONS)):
            abs_beam_idx = k * len(BEAM_CONNECTIONS) + i; group_idx = beam_map[num_columns + abs_beam_idx + 1]
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
