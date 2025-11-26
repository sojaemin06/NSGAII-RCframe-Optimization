
import math
import numpy as np
import pandas as pd
import openseespy.opensees as ops
from .modeling import build_model_for_section
from .utils import get_precalculated_strength, load_pm_data_for_column, get_pm_capacity_from_df, extract_local_element_forces

def evaluate(individual, chromosome_structure, DL, LL, Wx, Wy, Ex, Ey, h5_file, patterns_by_floor, beam_lengths, col_map, beam_map, num_columns, floors, H, column_locations, beam_connections, column_sections_df, beam_sections_df, FIXED_MIN_COST, FIXED_RANGE_COST, FIXED_MIN_CO2, FIXED_RANGE_CO2, load_combinations, column_sections, beam_sections):
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
        column_elem_ids, beam_elem_ids, node_map = build_model_for_section(floors, H, column_locations, beam_connections, col_indices, col_rotations, beam_indices, col_map, beam_map, column_sections, beam_sections, num_columns)
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
                nodal_load_x = story_force_x / len(column_locations); nodal_load_y = story_force_y / len(column_locations)
                for loc_idx in range(len(column_locations)):
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
            for loc_idx in range(len(column_locations)):
                if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
        if ops.analyze(1) == 0:
            for k in range(1, floors + 1):
                max_story_drift_x = 0
                for loc_idx in range(len(column_locations)):
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
            for loc_idx in range(len(column_locations)):
                if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)],NODALLOADx,NODALLOADy,0,0,0,0)
        if ops.analyze(1) == 0:
            for k in range(1, floors + 1):
                max_story_drift_y = 0
                for loc_idx in range(len(column_locations)):
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
                nodal_load_x = story_force_x / len(column_locations)
                for loc_idx in range(len(column_locations)):
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
                nodal_load_y = story_force_y / len(column_locations)
                for loc_idx in range(len(column_locations)):
                    if node_map.get((k, loc_idx)): ops.load(node_map[(k, loc_idx)], 0, nodal_load_y, 0, 0, 0, 0)
            if ops.analyze(1) == 0:
                # 각 층의 모든 절점 중 최대 변위(절대값)를 계산
                disps = [max([abs(ops.nodeDisp(nid, 2)) for (fl, _), nid in node_map.items() if fl == k]) for k in range(1, floors + 1)]
                wind_disps_y = disps
                if wind_disps_y:
                    actual_wind_disp_ratio_y = wind_disps_y[-1] / ((floors * H) / 400.0)
        actual_wind_disp_ratio = max(actual_wind_disp_ratio_x, actual_wind_disp_ratio_y)

    hierarchy_ratios = [1.0]
    GROUPING_STRATEGY = 'Hybrid'
    if GROUPING_STRATEGY != "Uniform":
        for k in range(floors - 1): 
            for i in range(len(column_locations)):
                abs_col_idx_lower = k * len(column_locations) + i; group_idx_lower = col_map[abs_col_idx_lower + 1]; sec_idx_lower = col_indices[group_idx_lower]; rot_lower = col_rotations[group_idx_lower]
                b_lower, h_lower = column_sections[sec_idx_lower]; dim_x_lower, dim_y_lower = (b_lower, h_lower) if rot_lower == 0 else (h_lower, b_lower)
                abs_col_idx_upper = (k + 1) * len(column_locations) + i; group_idx_upper = col_map[abs_col_idx_upper + 1]; sec_idx_upper = col_indices[group_idx_upper]; rot_upper = col_rotations[group_idx_upper]
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
