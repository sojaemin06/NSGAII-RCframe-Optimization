import openseespy.opensees as ops
import pandas as pd
import numpy as np
import math
import sys
import os
import src.config as cfg 
from src.utils import *

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

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
    
    num_locations = len(cfg.COLUMN_LOCATIONS)
    
    node_map = {}; node_id_counter = 1
    for k in range(cfg.FLOORS + 1):
        for i, (x, y) in enumerate(cfg.COLUMN_LOCATIONS):
            ops.node(node_id_counter, x, y, k * cfg.H)
            node_map[(k, i)] = node_id_counter
            if k == 0: ops.fix(node_id_counter, 1, 1, 1, 1, 1, 1)
            node_id_counter += 1
    ops.geomTransf('PDelta', 1, 1, 0, 0); ops.geomTransf('PDelta', 2, 0, 1, 0); ops.geomTransf('PDelta', 3, 0, 0, 1)

    # --- 강체 횡격막 (Rigid Diaphragm) 구현 ---
    for k in range(1, cfg.FLOORS + 1): 
        master_node_id = node_map[(k, 0)]
        for i in range(1, num_locations):
            slave_node_id = node_map[(k, i)]
            ops.equalDOF(master_node_id, slave_node_id, 1)
            ops.equalDOF(master_node_id, slave_node_id, 2)
            ops.equalDOF(master_node_id, slave_node_id, 6)
        
    column_elem_ids, beam_elem_ids = [], []; elem_id_counter = 1
    num_columns = num_locations * cfg.FLOORS
    
    for k in range(cfg.FLOORS):
        for i in range(num_locations):
            abs_col_idx = k * num_locations + i; group_idx = col_map[abs_col_idx + 1]
            
            if len(col_rotations) > 0:
                rotation_flag = col_rotations[group_idx]
                transf_tag = 2 if rotation_flag == 1 else 1
            else:
                transf_tag = 1
            
            sec_idx = col_indices[group_idx]; b_c, h_c = column_sections[sec_idx]
            
            # 유효 강성 적용 (ACI 318)
            A_c = b_c * h_c
            Iz_c = 0.7 * (b_c * h_c**3) / 12 
            Iy_c = 0.7 * (h_c * b_c**3) / 12
            J_c = (Iy_c + Iz_c)
            
            n1, n2 = node_map[(k, i)], node_map[(k + 1, i)]
            ops.element('elasticBeamColumn', elem_id_counter, n1, n2, A_c, E, G, J_c, Iy_c, Iz_c, transf_tag)
            column_elem_ids.append(elem_id_counter); elem_id_counter += 1
    for k in range(1, cfg.FLOORS + 1):
        for i, (loc_idx1, loc_idx2) in enumerate(cfg.BEAM_CONNECTIONS):
            abs_beam_idx = (k - 1) * len(cfg.BEAM_CONNECTIONS) + i
            group_idx = beam_map[num_columns + abs_beam_idx + 1]
            sec_idx = beam_indices[group_idx]; b_b, h_b = beam_sections[sec_idx]
            
            # 유효 강성 적용 (ACI 318)
            A_b = b_b * h_b
            Iz_b = 0.35 * (b_b * h_b**3) / 12
            Iy_b = 0.35 * (h_b * b_b**3) / 12
            J_b = (Iy_b + Iz_b)

            n1, n2 = node_map[(k, loc_idx1)], node_map[(k, loc_idx2)]
            ops.element('elasticBeamColumn', elem_id_counter, n1, n2, A_b, E, G, J_b, Iy_b, Iz_b, 3)
            beam_elem_ids.append(elem_id_counter); elem_id_counter += 1
    return column_elem_ids, beam_elem_ids, node_map

def evaluate(individual, DL, LL, h5_file, patterns_by_floor, 
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
        "violation_scwb": float('inf'), "violation_wind_disp": float('inf'),
        "max_drift_ratio": float('inf'),
        "forces_df": pd.DataFrame(), "violation": float('inf'), "absolute_margins": {}
    }

    num_locations = len(cfg.COLUMN_LOCATIONS)
    
    len_col_sec, len_col_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
    col_indices = individual[:len_col_sec]
    col_rotations = individual[len_col_sec : len_col_sec + len_col_rot]
    beam_indices = individual[len_col_sec + len_col_rot :]

    total_structure_weight = 0.0 
    story_weights = [0.0] * cfg.FLOORS 

    # 기둥 자중
    for col_idx in range(num_columns):
        floor_idx = col_idx // num_locations 
        group_idx = col_map[col_idx + 1]; sec_idx = col_indices[group_idx]
        b, h = column_sections[sec_idx]; unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']
        col_weight_per_story = b * h * cfg.H * unit_weight
        total_structure_weight += col_weight_per_story
        story_weights[floor_idx] += col_weight_per_story

    # 보 자중
    for beam_idx in range(num_beams):
        floor_idx = beam_idx // len(cfg.BEAM_CONNECTIONS)
        group_idx = beam_map[num_columns + beam_idx + 1]; sec_idx = beam_indices[group_idx]
        b, h = beam_sections[sec_idx]; unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']
        beam_len = beam_lengths[beam_idx % len(cfg.BEAM_CONNECTIONS)]
        beam_weight = b * h * beam_len * unit_weight
        total_structure_weight += beam_weight
        story_weights[floor_idx] += beam_weight

    # 슬래브 및 기타 고정하중
    slab_and_superimposed_DL_total_per_floor = cfg.FLOOR_AREA * cfg.DL_AREA_LOAD
    total_structure_weight += slab_and_superimposed_DL_total_per_floor * cfg.FLOORS
    for floor_idx in range(cfg.FLOORS):
        story_weights[floor_idx] += slab_and_superimposed_DL_total_per_floor

    H_total_structure = cfg.FLOORS * cfg.H
    Ta = cfg.PERIOD_CT * (H_total_structure ** cfg.PERIOD_X)
    
    Cs_denom = cfg.R_COEFF / cfg.I_FACTOR
    Cs_initial = cfg.SDS / Cs_denom 
    Cs_upper = cfg.SD1 / (Ta * Cs_denom) if Ta != 0 else float('inf') 
    Cs = min(Cs_initial, Cs_upper)
    Cs_lower_1 = 0.01
    Cs = max(Cs, Cs_lower_1)
    if cfg.SDS >= 0.1:
        Cs_lower_2 = 0.044 * cfg.SDS * cfg.I_FACTOR
        Cs = max(Cs, Cs_lower_2)
    
    base_shear_force_seismic = Cs * total_structure_weight

    try:
        column_elem_ids, beam_elem_ids, node_map = build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections)
    except Exception as e:
        # print(f"DEBUG: Error building model in evaluate: {str(e)}")
        import traceback
        traceback.print_exc()
        return failure_results_dict 
    
    # --- Deflection Check (Immediate + Long-term) ---
    max_deflection_ratio = 0.0
    deflection_ratios = []
    
    for i, elem_id in enumerate(beam_elem_ids):
        beam_len = beam_lengths[i % len(cfg.BEAM_CONNECTIONS)]
        group_idx = beam_map.get(num_columns + i + 1, 0)
        sec_idx = beam_indices[group_idx]
        b, h = beam_sections[sec_idx]
        
        # [수정] 층별 활하중 적용 (Deflection)
        beam_floor = (i // len(cfg.BEAM_CONNECTIONS)) + 1
        ll_val = LL.get(beam_floor, LL.get('default', 2.0))
        
        rho_prime = 0.0 
        if 'rho_c' in beam_sections_df.columns:
             rho_prime = beam_sections_df.iloc[sec_idx]['rho_c']
        lambda_delta = 2.0 / (1 + 50 * rho_prime)
        
        tributary_width = cfg.BEAM_TRIBUTARY_WIDTHS[i % len(cfg.BEAM_CONNECTIONS)]
        w_DL = (beam_sections_df.iloc[sec_idx]['UnitWeight'] * b * h) + (DL * tributary_width) 
        w_LL = ll_val * tributary_width # [수정] ll_val 사용
        
        E_c = 2.5791e7 
        I_g = (b * h**3) / 12
        I_eff = 0.35 * I_g 
        
        delta_immediate_LL = (5 * w_LL * (beam_len**4)) / (384 * E_c * I_eff)
        w_sustained = w_DL + 0.5 * w_LL
        delta_sustained_immediate = (5 * w_sustained * (beam_len**4)) / (384 * E_c * I_eff)
        delta_LT = lambda_delta * delta_sustained_immediate
        total_deflection = delta_LT + delta_immediate_LL
        allowable_deflection = beam_len / 240.0
        ratio = total_deflection / allowable_deflection
        deflection_ratios.append(ratio)

    actual_deflection_ratio = max(deflection_ratios) if deflection_ratios else 0.0

    # 층별 질량 및 고유치 해석
    mass_nodes = [node_map[(k, 0)] for k in range(1, cfg.FLOORS + 1)] 
    g_accel = 9.81 
    for k in range(cfg.FLOORS):
        node_tag = mass_nodes[k]
        m_val = story_weights[k] / g_accel 
        # 수치적 안정성을 위해 회전 및 수직 자유도에도 아주 작은 질량 할당 (ArpackSolver 오류 방지)
        ops.mass(node_tag, m_val, m_val, 0.01 * m_val, 1e-9, 1e-9, 1e-9) 

    num_eigenvalues = 1 
    try:
        lambda_val = ops.eigen(num_eigenvalues) 
    except:
        lambda_val = None
    
    # 기본 모드 형상 (삼각형 분포) 초기화
    phi_1x = [(k + 1) * cfg.H for k in range(cfg.FLOORS)] 
    phi_1y = [(k + 1) * cfg.H for k in range(cfg.FLOORS)]
    
    if lambda_val and len(lambda_val) > 0 and lambda_val[0] > 1e-9: 
        try:
            # 고유벡터 추출 시도
            extracted_phi_x = [ops.nodeEigenvector(node_tag, 1, 1) for node_tag in mass_nodes] 
            extracted_phi_y = [ops.nodeEigenvector(node_tag, 1, 2) for node_tag in mass_nodes] 
            
            # 모든 값이 0인지 확인 (추가 안전장치)
            if any(abs(v) > 1e-9 for v in extracted_phi_x): phi_1x = extracted_phi_x
            if any(abs(v) > 1e-9 for v in extracted_phi_y): phi_1y = extracted_phi_y
        except Exception: pass

    sum_w_phi_x = sum(story_weights[f] * phi_1x[f] for f in range(cfg.FLOORS))
    story_seismic_forces_x = [0.0] * cfg.FLOORS
    if sum_w_phi_x > 1e-9:
        for f in range(cfg.FLOORS):
            Cvx_mode = (story_weights[f] * phi_1x[f]) / sum_w_phi_x
            story_seismic_forces_x[f] = Cvx_mode * base_shear_force_seismic
            
    sum_w_phi_y = sum(story_weights[f] * phi_1y[f] for f in range(cfg.FLOORS))
    story_seismic_forces_y = [0.0] * cfg.FLOORS
    if sum_w_phi_y > 1e-9:
        for f in range(cfg.FLOORS):
            Cvy_mode = (story_weights[f] * phi_1y[f]) / sum_w_phi_y
            story_seismic_forces_y[f] = Cvy_mode * base_shear_force_seismic
            
    all_max_combo_forces, analysis_ok = [], True
    story_drifts_x, story_drifts_y = [], []
    wind_disps_x, wind_disps_y = [], []
    
    # Initialize Analysis Settings
    ops.timeSeries('Linear', 1)
    ops.system('ProfileSPD')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Newton')
    ops.analysis('Static')

    # Robust Analysis Strategy
    for i, (combo_name, factors) in enumerate(cfg.LOAD_COMBINATIONS):
        pattern_tag = i + 1
        converged = False
        
        # Suppress OpenSees output during model manipulation and analysis
        with SuppressOutput():
            try:
                ops.reset()
                ops.pattern('Plain', pattern_tag, 1)
                
                # --- Load Application ---
                for beam_idx, eid in enumerate(beam_elem_ids):
                    group_idx = beam_map[num_columns + beam_idx + 1]; sec_idx = beam_indices[group_idx]
                    b, h = beam_sections[sec_idx]; unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']
                    beam_self_weight = b * h * unit_weight 
                    beam_floor = (beam_idx // len(cfg.BEAM_CONNECTIONS)) + 1
                    conn_idx = beam_idx % len(cfg.BEAM_CONNECTIONS)
                    ll_val = LL.get(beam_floor, LL.get('default', 2.0))
                    tributary_width = cfg.BEAM_TRIBUTARY_WIDTHS[conn_idx] 
                    dl_line_load = DL * tributary_width; ll_line_load = ll_val * tributary_width
                    total_beam_load = beam_self_weight * factors["DL"]
                    loaded_beams_for_this_floor = patterns_by_floor.get(beam_floor, set())
                    if conn_idx in loaded_beams_for_this_floor: total_beam_load += dl_line_load * factors["DL"] + ll_line_load * factors["LL"]
                    else: total_beam_load += dl_line_load * factors["DL"]
                    if abs(total_beam_load) > 1e-6: ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0, -total_beam_load)

                for col_idx, eid in enumerate(column_elem_ids):
                    group_idx = col_map[col_idx + 1]; sec_idx = col_indices[group_idx]
                    b, h = column_sections[sec_idx]; unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']
                    col_self_weight = b * h * cfg.H * unit_weight; node1_tag, node2_tag = ops.eleNodes(eid)
                    ops.load(node1_tag, 0,0, -col_self_weight/2 * factors["DL"], 0,0,0)
                    ops.load(node2_tag, 0,0, -col_self_weight/2 * factors["DL"], 0,0,0)
                
                # Wind forces
                story_wind_forces_x = [0.0] * cfg.FLOORS; story_wind_forces_y = [0.0] * cfg.FLOORS
                def get_Kz(z): return 2.01 * ((max(z, 4.57) / cfg.ZG) ** (2 / cfg.ALPHA))
                Kz_top = get_Kz(cfg.FLOORS * cfg.H)
                qh = 0.000613 * Kz_top * cfg.KZT * cfg.KD * (cfg.BASIC_WIND_SPEED ** 2) 
                for k in range(cfg.FLOORS):
                    z_story = (k + 1) * cfg.H; Kz = get_Kz(z_story)
                    qz = 0.000613 * Kz * cfg.KZT * cfg.KD * (cfg.BASIC_WIND_SPEED ** 2) 
                    p_total = (qz * cfg.G_FACTOR * cfg.CP_WINDWARD) + (qh * cfg.G_FACTOR * abs(cfg.CP_LEEWARD))
                    story_wind_forces_x[k] = p_total * (cfg.BUILDING_WIDTH_Y * cfg.H)
                    story_wind_forces_y[k] = p_total * (cfg.BUILDING_WIDTH_X * cfg.H)

                if abs(factors["Ex"]) > 1e-9 or abs(factors["Ey"]) > 1e-9 or abs(factors["Wx"]) > 1e-9 or abs(factors["Wy"]) > 1e-9:
                    for k in range(cfg.FLOORS): 
                        Fx_total = (story_seismic_forces_x[k] * factors["Ex"]) + (story_wind_forces_x[k] * factors["Wx"])
                        Fy_total = (story_seismic_forces_y[k] * factors["Ey"]) + (story_wind_forces_y[k] * factors["Wy"])
                        nodal_load_x = Fx_total / num_locations; nodal_load_y = Fy_total / num_locations
                        for loc_idx in range(num_locations):
                            node_tag = node_map.get((k + 1, loc_idx)) 
                            if node_tag: ops.load(node_tag, nodal_load_x, nodal_load_y, 0, 0, 0, 0)

                # --- Adaptive Analysis Execution ---
                converged = False
                solvers = ['BandGeneral', 'UmfPack', 'FullGeneral']
                algorithms_list = ['Newton', 'NewtonLineSearch', 'ModifiedNewton', 'KrylovNewton']
                
                for solver in solvers:
                    if converged: break
                    for algo in algorithms_list:
                        try:
                            ops.system(solver)
                            ops.numberer('RCM')
                            ops.constraints('Transformation')
                            ops.integrator('LoadControl', 1.0)
                            ops.algorithm(algo)
                            ops.analysis('Static')
                            
                            if ops.analyze(1) == 0:
                                converged = True
                                break
                        except:
                            pass
                
                if converged:
                    # Capture Drifts for Seismic (ASCE-S-E1 and ASCE-S-E5)
                    if combo_name == "ASCE-S-E1":
                        for k in range(1, cfg.FLOORS + 1):
                            m_up, m_low = node_map.get((k, 0)), node_map.get((k - 1, 0))
                            d_up = ops.nodeDisp(m_up, 1) if m_up else 0.0
                            d_low = ops.nodeDisp(m_low, 1) if m_low else 0.0
                            scaling = cfg.CD_FACTOR / (0.7 * cfg.I_FACTOR)
                            story_drifts_x.append((abs(d_up - d_low) * scaling) / cfg.H)
                    elif combo_name == "ASCE-S-E5":
                        for k in range(1, cfg.FLOORS + 1):
                            m_up, m_low = node_map.get((k, 0)), node_map.get((k - 1, 0))
                            d_up = ops.nodeDisp(m_up, 2) if m_up else 0.0
                            d_low = ops.nodeDisp(m_low, 2) if m_low else 0.0
                            scaling = cfg.CD_FACTOR / (0.7 * cfg.I_FACTOR)
                            story_drifts_y.append((abs(d_up - d_low) * scaling) / cfg.H)
                    
                    # Capture Displacements for Wind (ASCE-S-W1 and ASCE-S-W3)
                    elif combo_name == "ASCE-S-W1":
                        for k in range(1, cfg.FLOORS + 1):
                            m_id = node_map.get((k, 0))
                            if m_id: wind_disps_x.append(abs(ops.nodeDisp(m_id, 1)))
                    elif combo_name == "ASCE-S-W3":
                        for k in range(1, cfg.FLOORS + 1):
                            m_id = node_map.get((k, 0))
                            if m_id: wind_disps_y.append(abs(ops.nodeDisp(m_id, 2)))

            except Exception:
                pass 
        # End of SuppressOutput
        
        if not converged:
            analysis_ok = False
            try:
                with SuppressOutput(): ops.remove('loadPattern', pattern_tag)
            except: pass
            break
            
        df_max_curr = extract_local_element_forces(column_elem_ids, beam_elem_ids)
        if df_max_curr.empty: 
            analysis_ok = False
            try:
                with SuppressOutput(): ops.remove('loadPattern', pattern_tag)
            except: pass
            break
            
        df_max_curr['Combo'] = combo_name
        all_max_combo_forces.append(df_max_curr)
        try:
            with SuppressOutput(): ops.remove('loadPattern', pattern_tag)
        except: pass

    # Restore Output
    try:
        ops.stopLog()
    except:
        pass

    if not analysis_ok or not all_max_combo_forces: return failure_results_dict
    df_all_combos = pd.concat(all_max_combo_forces, ignore_index=True)
    
    force_cols = ['Axial (kN)', 'Shear-y (kN)', 'Shear-z (kN)', 'Torsion (kNm)', 'Moment-y (kNm)', 'Moment-z (kNm)']
    idx_cols = ['ElementType', 'ElementID']
    max_rows = []
    for _, group in df_all_combos.groupby(idx_cols, observed=True):
        row = {k: group.iloc[0][k] for k in idx_cols}
        for force in force_cols:
            max_idx = group[force].abs().idxmax(); row[force] = group.loc[max_idx, force]
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
                dcr_z = mz / (mn_z + 1e-9); dcr_y = my / (mn_y + 1e-9)
                ratios = [p/(pn_z+1e-9), p/(pn_y+1e-9), vy/(strengths['Vn_y']+1e-9), vz/(strengths['Vn_z']+1e-9), dcr_z, dcr_y, (dcr_z**1.5)+(dcr_y**1.5)]
            else:
                abs_beam_idx = beam_elem_ids.index(elem_id); group_idx = beam_map[num_columns + abs_beam_idx + 1]; sec_idx = beam_indices[group_idx]
                strengths = get_precalculated_strength(elem_type, sec_idx, column_sections_df, beam_sections_df)
                ratios = [vz/(strengths['Vn_z']+1e-9), mz/(strengths['Mn_z']+1e-9)]
            strength_ratios.append(max(r for r in ratios if r is not None and not math.isinf(r) and r >= 0))
        except (KeyError, IndexError): strength_ratios.append(float('inf'))
    max_strength_ratio = max(strength_ratios) if strength_ratios else 1.0
    mean_strength_ratio = np.mean([r for r in strength_ratios if not math.isinf(r)]) if strength_ratios else 0.0

    # Process Drift results
    actual_drift_ratio = max(max(story_drifts_x) if story_drifts_x else [0.0], max(story_drifts_y) if story_drifts_y else [0.0])
    
    # Process Wind Displacement results
    actual_wind_disp_ratio_x = wind_disps_x[-1] / ((cfg.FLOORS * cfg.H) / 400.0) if wind_disps_x else 0.0
    actual_wind_disp_ratio_y = wind_disps_y[-1] / ((cfg.FLOORS * cfg.H) / 400.0) if wind_disps_y else 0.0
    actual_wind_disp_ratio = max(actual_wind_disp_ratio_x, actual_wind_disp_ratio_y)

    scwb_ratios = []
    node_beams_x = {}; node_beams_y = {}
    for beam_idx, (u, v) in enumerate(cfg.BEAM_CONNECTIONS):
        ux, uy = cfg.COLUMN_LOCATIONS[u]; vx, vy = cfg.COLUMN_LOCATIONS[v]
        is_x_beam = abs(uy - vy) < 1e-4
        for k in range(1, cfg.FLOORS + 1):
            abs_beam_idx = (k - 1) * len(cfg.BEAM_CONNECTIONS) + beam_idx
            group_idx = beam_map[num_columns + abs_beam_idx + 1]
            sec_idx = beam_indices[group_idx]; mn_beam = beam_sections_df.iloc[sec_idx]['PiM']
            for node_idx in [u, v]:
                node_key = (k, node_idx)
                if is_x_beam:
                    if node_key not in node_beams_x: node_beams_x[node_key] = []
                    node_beams_x[node_key].append(mn_beam)
                else:
                    if node_key not in node_beams_y: node_beams_y[node_key] = []
                    node_beams_y[node_key].append(mn_beam)
    for k in range(1, cfg.FLOORS + 1):
        for i in range(num_locations):
            sum_mb_x = sum(node_beams_x.get((k, i), [])); sum_mb_y = sum(node_beams_y.get((k, i), []))
            if sum_mb_x == 0 and sum_mb_y == 0: continue
            cols_to_check = []
            if k >= 1: cols_to_check.append( (k-1) * num_locations + i )
            if k < cfg.FLOORS: cols_to_check.append( k * num_locations + i )
            sum_mc_for_x_beams, sum_mc_for_y_beams = 0.0, 0.0
            for c_idx in cols_to_check:
                group_idx = col_map[c_idx + 1]; sec_idx = col_indices[group_idx]
                
                # Determine rotation based on scenario
                if len(col_rotations) > 0:
                    rot = col_rotations[group_idx]
                else:
                    # Scenario B: Odd index means rotated section
                    rot = 1 if sec_idx % 2 == 1 else 0
                
                pm_df = load_pm_data_for_column(h5_file, sec_idx)
                _, mn0_z = get_pm_capacity_from_df(0, pm_df, axis='z'); _, mn0_y = get_pm_capacity_from_df(0, pm_df, axis='y')
                
                # Apply rotation to moment capacity summation
                if rot == 0: sum_mc_for_x_beams += mn0_y; sum_mc_for_y_beams += mn0_z
                else: sum_mc_for_x_beams += mn0_z; sum_mc_for_y_beams += mn0_y
            if sum_mb_x > 0: scwb_ratios.append( (1.2 * sum_mb_x) / (sum_mc_for_x_beams + 1e-9) )
            if sum_mb_y > 0: scwb_ratios.append( (1.2 * sum_mb_y) / (sum_mc_for_y_beams + 1e-9) )
    actual_scwb_ratio = max(scwb_ratios) if scwb_ratios else 0.0

    # --- Cost & CO2 Calculation with Strict ACI Detailing ---
    total_cost, total_co2 = 0, 0
    UNIT_COST_CONCRETE = 80000; UNIT_COST_STEEL = 1000000; UNIT_COST_FORMWORK = cfg.FORMWORK_UNIT_COST
    DENSITY_STEEL = 7.85; ECF_CONCRETE = 0.15; ECF_STEEL = 1.99

    # --- Columns Cost & CO2 ---
    for i in range(num_columns):
        group_idx = col_map[i + 1]; sec_idx = col_indices[group_idx]
        b, h = column_sections[sec_idx]
        row = column_sections_df.iloc[sec_idx]
        
        vol_conc_gross = b * h * cfg.H
        rho = row['rho']
        vol_main_steel = vol_conc_gross * rho
        vol_conc_net = vol_conc_gross - vol_main_steel
        
        cost_conc = vol_conc_net * UNIT_COST_CONCRETE
        co2_conc = (vol_conc_net * 2400) * ECF_CONCRETE
        
        mass_main_steel_ton = vol_main_steel * DENSITY_STEEL
        cost_main_steel = mass_main_steel_ton * UNIT_COST_STEEL
        co2_main_steel = (mass_main_steel_ton * 1000) * ECF_STEEL
        
        # Stirrup & Crosstie
        cover = row['Cover_size'] / 1000; tie_size_mm = row['Stirrup_size']; main_bar_size_mm = row['MainRebar_size']
        L_hinge = max(h, b, cfg.H / 6)
        s_end = min(b/4, h/4, 6 * (main_bar_size_mm/1000), 150/1000)
        s_mid = min(6 * (main_bar_size_mm/1000), 150/1000)
        num_stirrups = math.ceil(2 * L_hinge / s_end) + math.ceil(max(0, cfg.H - 2 * L_hinge) / s_mid)
        
        hook_len = max(6 * (tie_size_mm / 1000), 0.075)
        tie_len_outer = 2 * ((b - 2*cover) + (h - 2*cover)) + (2 * hook_len)
        
        crosstie_len_total = 0
        side_rebars_top = int(row.get('side_rebars_top', 2) - 2)
        side_rebars_left = int(row.get('side_rebars_left', 2) - 2)
        
        if side_rebars_left > 0:
            bar_spacing_h = (h - 2*cover - main_bar_size_mm/1000) / (side_rebars_left + 1)
            if bar_spacing_h > 150/1000:
                crosstie_len_total += math.floor((side_rebars_left + 1) / 2) * (b - 2*cover + 2*hook_len)
        if side_rebars_top > 0:
            bar_spacing_b = (b - 2*cover - main_bar_size_mm/1000) / (side_rebars_top + 1)
            if bar_spacing_b > 150/1000:
                crosstie_len_total += math.floor((side_rebars_top + 1) / 2) * (h - 2*cover + 2*hook_len)
                
        total_tie_len_per_set = tie_len_outer + crosstie_len_total
        tie_area = (math.pi * (tie_size_mm/1000)**2) / 4
        vol_tie_steel = num_stirrups * total_tie_len_per_set * tie_area
        mass_tie_steel_ton = vol_tie_steel * DENSITY_STEEL
        cost_tie_steel = mass_tie_steel_ton * UNIT_COST_STEEL
        co2_tie_steel = (mass_tie_steel_ton * 1000) * ECF_STEEL
        
        cost_form = 2 * (b + h) * cfg.H * UNIT_COST_FORMWORK
        total_cost += cost_conc + cost_main_steel + cost_tie_steel + cost_form
        total_co2 += co2_conc + co2_main_steel + co2_tie_steel

    # --- Beams Cost & CO2 ---
    for k in range(cfg.FLOORS):
        for i in range(len(cfg.BEAM_CONNECTIONS)):
            abs_beam_idx = k * len(cfg.BEAM_CONNECTIONS) + i; group_idx = beam_map[num_columns + abs_beam_idx + 1]
            sec_idx = beam_indices[group_idx]
            b, h = beam_sections[sec_idx]; L_beam = beam_lengths[i]
            row = beam_sections_df.iloc[sec_idx]; d = h - 0.06; main_bar_size_mm = row.get('dimension', 22)
            
            rho_t = row.get('rho_t', 0.01); rho_c = row.get('rho_c', 0.005)
            vol_main_steel = b * h * (rho_t + rho_c) * L_beam
            vol_conc_net = (b*h*L_beam) - vol_main_steel
            
            cost_conc = vol_conc_net * UNIT_COST_CONCRETE
            co2_conc = (vol_conc_net * 2400) * ECF_CONCRETE
            mass_main_steel_ton = vol_main_steel * DENSITY_STEEL
            cost_main_steel = mass_main_steel_ton * UNIT_COST_STEEL
            co2_main_steel = (mass_main_steel_ton * 1000) * ECF_STEEL

            L_hinge = 2 * h
            s_end = min(d/4, 6 * (main_bar_size_mm/1000), 150/1000)
            s_mid = d/2
            num_stirrups = math.ceil(2 * L_hinge / s_end) + math.ceil(max(0, L_beam - 2 * L_hinge) / s_mid)
            
            cover = 0.04; tie_size_mm = row.get('stirrup', 10)
            hook_len = max(6 * (tie_size_mm / 1000), 0.075)
            tie_len = 2 * ((b - 2*cover) + (h - 2*cover)) + (2 * hook_len)
            tie_area = (math.pi * (tie_size_mm/1000)**2) / 4
            vol_tie_steel = num_stirrups * tie_len * tie_area
            cost_tie_steel = (vol_tie_steel * DENSITY_STEEL) * UNIT_COST_STEEL
            co2_tie_steel = (vol_tie_steel * DENSITY_STEEL * 1000) * ECF_STEEL

            cost_skin_steel, co2_skin_steel = 0.0, 0.0
            if h > 0.9:
                num_skin_bars_per_side = math.floor((h - 0.1) / 0.3)
                vol_skin_steel = 2 * num_skin_bars_per_side * L_beam * ((math.pi * (10/1000)**2) / 4)
                cost_skin_steel = (vol_skin_steel * DENSITY_STEEL) * UNIT_COST_STEEL
                co2_skin_steel = (vol_skin_steel * DENSITY_STEEL * 1000) * ECF_STEEL

            cost_form = (2*h + b) * L_beam * UNIT_COST_FORMWORK
            total_cost += cost_conc + cost_main_steel + cost_tie_steel + cost_skin_steel + cost_form
            total_co2 += co2_conc + co2_main_steel + co2_tie_steel + co2_skin_steel

    # --- Constraint Normalization (Updated for Raw Drift Ratio) ---
    limits = {
        'strength': 1.0, 'drift': 0.02, 'wind_disp': 1.0, 'deflection': 1.0, 
        'scwb': 1.2
    }
    norm_scales = {
        'strength': 1.0, 'drift': 0.02, 'wind_disp': 1.0, 'deflection': 1.0, 
        'scwb': 0.2
    }
    
    # inf 값 방어 로직 추가
    safe_drift = actual_drift_ratio if not math.isinf(actual_drift_ratio) else 1.0 # inf일 경우 임의의 큰 위반량 유도
    
    margins = {key: max(0, val - limits[key]) for key, val in {
        'strength': max_strength_ratio, 'drift': actual_drift_ratio,
        'wind_disp': actual_wind_disp_ratio, 'deflection': actual_deflection_ratio,
        'scwb': actual_scwb_ratio
    }.items()}

    total_normalized_violation = 0
    normalized_margins = {}
    for key, margin in margins.items():
        if math.isinf(margin):
            normalized_margin = 1.0 # 최대 페널티
        else:
            normalized_margin = min(1.0, margin / (norm_scales[key] + 1e-9))
        
        total_normalized_violation += normalized_margin
        normalized_margins[key] = normalized_margin

    # 해석이 실패했거나 drift가 inf인 경우 violation을 최소 1.0 이상으로 강제
    if not analysis_ok or math.isinf(actual_drift_ratio):
        total_normalized_violation = max(total_normalized_violation, 1.0)


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
        "violation_scwb": actual_scwb_ratio, "violation_wind_disp": actual_wind_disp_ratio,
        "max_drift_ratio": actual_drift_ratio,
        "N_types": len(set(col_indices)) + len(set(beam_indices)),
        "forces_df": final_max_forces
    }
    
    return detailed_results_dict