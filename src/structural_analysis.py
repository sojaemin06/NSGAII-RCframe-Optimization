
import openseespy.opensees as ops
import pandas as pd
import numpy as np
import math
import src.config as cfg # Use cfg prefix for dynamic access
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
    
    num_locations = len(cfg.COLUMN_LOCATIONS) # [Fix] 정의 위치 이동
    
    node_map = {}; node_id_counter = 1
    for k in range(cfg.FLOORS + 1):
        for i, (x, y) in enumerate(cfg.COLUMN_LOCATIONS):
            ops.node(node_id_counter, x, y, k * cfg.H)
            node_map[(k, i)] = node_id_counter
            if k == 0: ops.fix(node_id_counter, 1, 1, 1, 1, 1, 1)
            node_id_counter += 1
    ops.geomTransf('PDelta', 1, 1, 0, 0); ops.geomTransf('PDelta', 2, 0, 1, 0); ops.geomTransf('PDelta', 3, 0, 0, 1)

    # --- 강체 횡격막 (Rigid Diaphragm) 구현 (equalDOF 기반으로 변경) ---
    # 각 층별로 모든 절점의 횡방향 자유도(X, Y) 및 Z축 회전을 마스터 절점에 구속
    for k in range(1, cfg.FLOORS + 1): # 1층부터 최상층까지
        master_node_id = node_map[(k, 0)] # 각 층의 첫 번째 기둥 노드를 마스터로 설정

        # 슬레이브 노드들을 마스터 노드에 구속 (X, Y 방향 변위, Z축 회전)
        for i in range(1, num_locations): # 마스터 노드를 제외한 나머지 기둥 노드들
            slave_node_id = node_map[(k, i)]
            # X방향 변위 구속 (dof=1)
            ops.equalDOF(master_node_id, slave_node_id, 1)
            # Y방향 변위 구속 (dof=2)
            ops.equalDOF(master_node_id, slave_node_id, 2)
            # Z축 회전 구속 (dof=6)
            ops.equalDOF(master_node_id, slave_node_id, 6)
        
    column_elem_ids, beam_elem_ids = [], []; elem_id_counter = 1
    # num_locations = len(COLUMN_LOCATIONS) # [Removed] 기존 위치 삭제
    num_columns = num_locations * cfg.FLOORS
    
    for k in range(cfg.FLOORS):
        for i in range(num_locations):
            abs_col_idx = k * num_locations + i; group_idx = col_map[abs_col_idx + 1]
            
            # [Scenario B Support] Handle case where col_rotations is empty
            if len(col_rotations) > 0:
                rotation_flag = col_rotations[group_idx]
                transf_tag = 2 if rotation_flag == 1 else 1
            else:
                transf_tag = 1 # Default to unrotated if no rotation genes provided
            
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
    for k in range(1, cfg.FLOORS + 1):
        for i, (loc_idx1, loc_idx2) in enumerate(cfg.BEAM_CONNECTIONS):
            abs_beam_idx = (k - 1) * len(cfg.BEAM_CONNECTIONS) + i
            group_idx = beam_map[num_columns + abs_beam_idx + 1]
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

def evaluate(individual, DL, LL, h5_file, patterns_by_floor, 
             col_map, beam_map, beam_sections, column_sections, 
             beam_sections_df, column_sections_df, beam_lengths, 
             chromosome_structure, num_columns, num_beams):
    """
    유전 알고리즘의 핵심 평가 함수.
    """
    num_locations = len(cfg.COLUMN_LOCATIONS) # [Fix] evaluate 함수 내 변수 정의 추가
    
    len_col_sec, len_col_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
    col_indices = individual[:len_col_sec]
    col_rotations = individual[len_col_sec : len_col_sec + len_col_rot]
    beam_indices = individual[len_col_sec + len_col_rot :]

    # 1. 자중(Self-weight) 및 고정 하중(Dead Load) 계산
    total_structure_weight = 0.0 # 총 지진 중량 W
    story_weights = [0.0] * cfg.FLOORS # 각 층의 무게 (지진력 분배용 Wx)
    story_heights_from_base = [(k + 1) * cfg.H for k in range(cfg.FLOORS)] # 각 층 높이 (최하층부터)

    # 기둥 자중
    for col_idx in range(num_columns):
        floor_idx = col_idx // num_locations # 0-indexed floor for column

        group_idx = col_map[col_idx + 1]; sec_idx = col_indices[group_idx]
        b, h = column_sections[sec_idx]; unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']
        col_weight_per_story = b * h * cfg.H * unit_weight # 한 층 기둥의 무게
        
        total_structure_weight += col_weight_per_story
        story_weights[floor_idx] += col_weight_per_story # 일단 해당 층에 전체 할당. 나중에 층 중량 결정 시 재조정.

    # 보 자중
    for beam_idx in range(num_beams):
        floor_idx = beam_idx // len(cfg.BEAM_CONNECTIONS) # 0-indexed floor for beam
        group_idx = beam_map[num_columns + beam_idx + 1]; sec_idx = beam_indices[group_idx]
        b, h = beam_sections[sec_idx]; unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']
        beam_len = beam_lengths[beam_idx % len(cfg.BEAM_CONNECTIONS)]
        beam_weight = b * h * beam_len * unit_weight # 보 자중
        
        total_structure_weight += beam_weight
        story_weights[floor_idx] += beam_weight

    # 슬래브 및 기타 고정하중 (면적 하중)
    # config.py의 DL_AREA_LOAD는 슬래브 자중을 포함한 면적당 고정하중이므로, 이 값을 사용
    slab_and_superimposed_DL_total_per_floor = cfg.FLOOR_AREA * cfg.DL_AREA_LOAD
    total_structure_weight += slab_and_superimposed_DL_total_per_floor * cfg.FLOORS
    for floor_idx in range(cfg.FLOORS):
        story_weights[floor_idx] += slab_and_superimposed_DL_total_per_floor

    # 2. 건물 총 높이 H_n (최상층 높이)
    H_total_structure = cfg.FLOORS * cfg.H

    # 3. 고유 주기 Ta 계산 (ASCE 7-16 Eq. 12.8-7) - 이 값은 Cs 계산에만 사용됨
    Ta = cfg.PERIOD_CT * (H_total_structure ** cfg.PERIOD_X)
    
    # 4. 지진 응답 계수 Cs 계산 및 제한 (ASCE 7-16 12.8.1)
    Cs_denom = cfg.R_COEFF / cfg.I_FACTOR
    Cs_initial = cfg.SDS / Cs_denom # 초기 Cs = SDS / (R/Ie) (ASCE 7-16 Eq. 12.8-2)

    # Cs 상한 (ASCE 7-16 Eq. 12.8-3)
    # Cs_upper = SD1 / (Ta * (R_COEFF / I_FACTOR))
    # T > TL (Long-period transition period) 일 경우 다른 상한 적용되나, 여기선 Ta < TL 가정
    Cs_upper = cfg.SD1 / (Ta * Cs_denom) if Ta != 0 else float('inf') # Ta가 0이 아닐 때만 계산

    Cs = min(Cs_initial, Cs_upper) # Cs 상한 적용

    # Cs 하한 (ASCE 7-16 Eq. 12.8-5)
    Cs_lower_1 = 0.01
    Cs = max(Cs, Cs_lower_1)

    # 추가 하한 (ASCE 7-16 Eq. 12.8-6) - SDS >= 0.1g 일 때
    if cfg.SDS >= 0.1:
        Cs_lower_2 = 0.044 * cfg.SDS * cfg.I_FACTOR
        Cs = max(Cs, Cs_lower_2)
    
    # 5. 베이스 전단력 V 계산 (ASCE 7-16 Eq. 12.8-1)
    base_shear_force_seismic = Cs * total_structure_weight

    failure_results_dict = {
        "cost": float('inf'), "co2": float('inf'),
        "mean_strength_ratio": float('inf'), "max_strength_ratio": float('inf'),
        "strength_ratios": [], "story_drifts_x": [], "story_drifts_y": [],
        "wind_displacements_x": [], "wind_displacements_y": [],
        "violation_deflection": float('inf'), "violation_drift": float('inf'),
        "violation_hierarchy": float('inf'), "violation_wind_disp": float('inf'),
        "forces_df": pd.DataFrame(), "violation": float('inf'), "absolute_margins": {}
    }
    len_col_sec, len_col_rot = chromosome_structure['col_sec'], chromosome_structure['col_rot']
    col_indices = individual[:len_col_sec]
    col_rotations = individual[len_col_sec : len_col_sec + len_col_rot]
    beam_indices = individual[len_col_sec + len_col_rot :]

    try:
        column_elem_ids, beam_elem_ids, node_map = build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections)
        print("DEBUG: Model built successfully.")
    except Exception as e:
        print(f"DEBUG: Error building model: {e}")
        return failure_results_dict 
    
    # [Fix] num_locations 정의 위치 이동 (여기서 정의되어야 다른 코드에서 사용 가능)
    num_locations = len(cfg.COLUMN_LOCATIONS)

    # --- [Modified] Deflection Check (Immediate + Long-term) ---
    # ACI 318-19 24.2.4.1: Total deflection = Delta_immediate + lambda_delta * Delta_sustained
    # lambda_delta = xi / (1 + 50 * rho')
    # xi = 2.0 (duration > 5 years)
    
    # Helper function to calculate max deflection of a simply supported beam under uniform load
    # Delta_max = (5 * w * L^4) / (384 * E * I_eff)
    # Note: This is a simplified approximation assuming simple support and uniform load.
    # For rigorous analysis, nodal displacements from OpenSees could be used, but beam local deflection is not directly output.
    
    max_deflection_ratio = 0.0
    deflection_ratios = []
    
    for i, elem_id in enumerate(beam_elem_ids):
        beam_len = beam_lengths[i % len(cfg.BEAM_CONNECTIONS)]
        group_idx = beam_map.get(num_columns + i + 1, 0)
        sec_idx = beam_indices[group_idx]
        b, h = beam_sections[sec_idx]
        
        # Get Reinforcement Ratios (rho, rho') from DB
        # Assuming column_sections_df has 'Rho_top' (tension) and 'Rho_bot' (compression) for positive moment at midspan?
        # Actually, beam DB usually has 'Rho' or total area. Let's assume rho' (compression) is approx 0.3 * rho or 0 if not specified.
        # For conservative long-term check, we can use rho' = 0 (worst case lambda = 2.0).
        # Or check if DB has 'Rho_compression'. If not, use 0.
        rho_prime = 0.0 
        if 'rho_c' in beam_sections_df.columns: # [Fix] DB 컬럼명 수정 (Rho_comp -> rho_c)
             rho_prime = beam_sections_df.iloc[sec_idx]['rho_c']
        
        lambda_delta = 2.0 / (1 + 50 * rho_prime)
        
        # Loads for deflection
        # Immediate Live Load: LL
        # Sustained Load: DL + 0.5 * LL (Assumption: 50% of LL is sustained)
        
        tributary_width = cfg.BEAM_TRIBUTARY_WIDTHS[i % len(cfg.BEAM_CONNECTIONS)]
        w_DL = (beam_sections_df.iloc[sec_idx]['UnitWeight'] * b * h) + (DL * tributary_width) # Self-weight + Slab DL
        w_LL = LL * tributary_width
        
        # Effective Moment of Inertia (I_e) - Simplified as 0.35 * Ig for cracked section per ACI 318
        # Or use Ig for uncracked if service load is low. Using 0.35 Ig is conservative for deflection.
        # ACI 318-19 Table 6.6.3.1.1(a) for beams -> 0.35 Ig
        E_c = 2.5791e7 # kPa (from build_model)
        I_g = (b * h**3) / 12
        I_eff = 0.35 * I_g 
        
        # Deflection calculation (5wL^4 / 384EI)
        # w in kN/m, L in m, E in kPa (kN/m2), I in m4 -> Delta in m
        
        # 1. Immediate Deflection due to Live Load (Delta_LL)
        delta_immediate_LL = (5 * w_LL * (beam_len**4)) / (384 * E_c * I_eff)
        
        # 2. Sustained Deflection (Delta_sustained) -> DL + 0.5*LL
        w_sustained = w_DL + 0.5 * w_LL
        delta_sustained_immediate = (5 * w_sustained * (beam_len**4)) / (384 * E_c * I_eff)
        
        # 3. Long-term Deflection (Delta_LT) = lambda * Delta_sustained_immediate
        delta_LT = lambda_delta * delta_sustained_immediate
        
        # Total Deflection to check against L/240 (or L/480)
        # ACI Table 24.2.2: Immediate LL -> L/180 or L/360
        # Total (LT + Immediate LL) -> L/240 (roof/floor supporting non-structural elements likely to be damaged)
        
        # We check Total = Delta_LT + Delta_immediate_LL
        # (Note: Total deflection is strictly Delta_immediate_DL + Delta_immediate_LL + Delta_LT, 
        # but code limit usually applies to the part occurring *after* attachment of non-structural elements.
        # Commonly: Delta_LL + Delta_LT is checked against L/240)
        
        total_deflection = delta_LT + delta_immediate_LL
        
        allowable_deflection = beam_len / 240.0
        
        ratio = total_deflection / allowable_deflection
        deflection_ratios.append(ratio)

    actual_deflection_ratio = max(deflection_ratios) if deflection_ratios else 0.0

    # 6. 층별 질량 매핑
    mass_nodes = [node_map[(k, 0)] for k in range(1, cfg.FLOORS + 1)] # 각 층의 마스터 노드
    # story_weights는 이미 계산되어 있으므로, 이를 OpenSees에 질량으로 정의
    
    # OpenSees에 질량 정의 (마스터 노드에 lump mass 집중)
    # 지진하중은 X, Y 방향으로 작용하므로, X(1), Y(2) 방향 질량만 정의
    g_accel = 9.81 # m/s^2
    for k in range(cfg.FLOORS):
        node_tag = mass_nodes[k]
        m_val = story_weights[k] / g_accel # 질량 = 무게 / g
        ops.mass(node_tag, m_val, m_val, 0, 0, 0, 0) # X, Y 방향 질량만 고려 (회전 질량 무시)

    # 고유치 해석 (Eigenvalue Analysis) 수행
    num_eigenvalues = 1 # 1차 모드만 필요
    lambda_val = ops.eigen(num_eigenvalues) # 람다 = 오메가^2 (rad/s)^2
    
    # 1차 모드 주기 및 형상 추출
    phi_1x = [(k + 1) * cfg.H for k in range(cfg.FLOORS)] # 기본값으로 높이에 비례하는 선형 모드 가정
    phi_1y = [(k + 1) * cfg.H for k in range(cfg.FLOORS)]
    
    if lambda_val and lambda_val[0] > 1e-9: # 람다가 유효한 값일 경우
        # omega = math.sqrt(lambda_val[0]) # 고유 진동수 (rad/s)
        # T1 = (2 * math.pi) / omega # 1차 고유 주기 (초) (여기서 계산된 T1은 Cs 계산에는 사용되지 않음)
        
        # 1차 모드 형상 (X, Y 방향 변위 성분) 추출
        try:
            phi_1x = [ops.nodeEigenvector(node_tag, 1, 1) for node_tag in mass_nodes] # mode 1, dof 1 (X)
            phi_1y = [ops.nodeEigenvector(node_tag, 1, 2) for node_tag in mass_nodes] # mode 1, dof 2 (Y)
        except Exception as e:
            print(f"DEBUG: Failed to get eigenvector: {e}. Using linear approximation for mode shapes.")
            # 실패 시 선형 근사 유지 (초기값 그대로)
    else:
        print(f"DEBUG: Eigenvalue analysis failed or invalid. Using linear approximation for mode shapes.")
        # 실패 시 선형 근사 유지 (초기값 그대로)

    # 층별 지진력 분배 (1차 모드 형상 기반)
    # Fx = (Cvx * V)
    # Cvx = (wx * phi_ix) / sum(wj * phi_ij)
    
    # X 방향 지진력 분배
    sum_w_phi_x = sum(story_weights[f] * phi_1x[f] for f in range(cfg.FLOORS))
    story_seismic_forces_x = [0.0] * cfg.FLOORS
    if sum_w_phi_x > 1e-9:
        for f in range(cfg.FLOORS):
            Cvx_mode = (story_weights[f] * phi_1x[f]) / sum_w_phi_x
            story_seismic_forces_x[f] = Cvx_mode * base_shear_force_seismic
            
    # Y 방향 지진력 분배
    sum_w_phi_y = sum(story_weights[f] * phi_1y[f] for f in range(cfg.FLOORS))
    story_seismic_forces_y = [0.0] * cfg.FLOORS
    if sum_w_phi_y > 1e-9:
        for f in range(cfg.FLOORS):
            Cvy_mode = (story_weights[f] * phi_1y[f]) / sum_w_phi_y
            story_seismic_forces_y[f] = Cvy_mode * base_shear_force_seismic
            
    # OpenSees 해석 부분 (기존 코드 유지)
    all_max_combo_forces, analysis_ok = [], True
    ops.timeSeries('Linear',1)
    ops.system('ProfileSPD')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('LoadControl',1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    for i, (combo_name, factors) in enumerate(cfg.LOAD_COMBINATIONS):
        pattern_tag = i + 1
        ops.reset()
        ops.pattern('Plain', pattern_tag, 1)
        
        for beam_idx, eid in enumerate(beam_elem_ids):
            group_idx = beam_map[num_columns + beam_idx + 1]
            sec_idx = beam_indices[group_idx]
            b, h = beam_sections[sec_idx]
            unit_weight = beam_sections_df.iloc[sec_idx]['UnitWeight']
            beam_self_weight = b * h * unit_weight # kN/m (보 자중 선하중)

            beam_floor = (beam_idx // len(cfg.BEAM_CONNECTIONS)) + 1
            conn_idx = beam_idx % len(cfg.BEAM_CONNECTIONS)
            
            tributary_width = cfg.BEAM_TRIBUTARY_WIDTHS[conn_idx] # 해당 보의 분담폭 (m)

            # 면적 하중을 선하중으로 변환 (kN/m2 * m = kN/m)
            dl_line_load = DL * tributary_width
            ll_line_load = LL * tributary_width

            # 하중 조합에 따른 보 선하중 계산
            total_beam_load = beam_self_weight * factors["DL"]
            
            # 패턴 로딩 적용
            loaded_beams_for_this_floor = patterns_by_floor.get(beam_floor, set())
            if conn_idx in loaded_beams_for_this_floor: 
                total_beam_load += dl_line_load * factors["DL"] + ll_line_load * factors["LL"]
            else: # 활하중이 재하되지 않는 보에는 고정하중만 적용
                total_beam_load += dl_line_load * factors["DL"]
            
            if abs(total_beam_load) > 1e-6:
                ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0, -total_beam_load)

        for col_idx, eid in enumerate(column_elem_ids):
            group_idx = col_map[col_idx + 1]; sec_idx = col_indices[group_idx]
            b, h = column_sections[sec_idx]; unit_weight = column_sections_df.iloc[sec_idx]['UnitWeight']
            col_self_weight = b * h * cfg.H * unit_weight; node1_tag, node2_tag = ops.eleNodes(eid)
            ops.load(node1_tag, 0,0, -col_self_weight/2 * factors["DL"], 0,0,0)
            ops.load(node2_tag, 0,0, -col_self_weight/2 * factors["DL"], 0,0,0)
        
        # [수정] 풍하중 상세 산정 (ASCE 7-16)
        # 각 층별 높이에 따른 풍압(qz) 계산 및 하중 분배
        story_wind_forces_x = [0.0] * cfg.FLOORS
        story_wind_forces_y = [0.0] * cfg.FLOORS
        
        # 기준 높이 H에서의 풍압 q_h (풍하측용)
        def get_Kz(z):
            z_eff = max(z, 4.57) # 최소 높이 15ft (4.57m)
            return 2.01 * ((z_eff / cfg.ZG) ** (2 / cfg.ALPHA))
            
        Kz_top = get_Kz(cfg.FLOORS * cfg.H)
        qh = 0.000613 * Kz_top * cfg.KZT * cfg.KD * (cfg.BASIC_WIND_SPEED ** 2) # kN/m2
        
        for k in range(cfg.FLOORS):
            z_story = (k + 1) * cfg.H
            Kz = get_Kz(z_story)
            qz = 0.000613 * Kz * cfg.KZT * cfg.KD * (cfg.BASIC_WIND_SPEED ** 2) # kN/m2
            
            # 풍압력 p = q * G * Cp
            # Total Force = (p_windward + p_leeward) * Area
            p_windward = qz * cfg.G_FACTOR * cfg.CP_WINDWARD
            p_leeward = qh * cfg.G_FACTOR * abs(cfg.CP_LEEWARD) # Leeward는 qh 기준, 흡입력이므로 절대값 더함
            p_total = p_windward + p_leeward
            
            # 수압 면적 (층고 H * 폭)
            A_x = cfg.BUILDING_WIDTH_Y * cfg.H # X방향 풍하중을 받는 면적 (Y축 폭)
            A_y = cfg.BUILDING_WIDTH_X * cfg.H # Y방향 풍하중을 받는 면적 (X축 폭)
            
            story_wind_forces_x[k] = p_total * A_x
            story_wind_forces_y[k] = p_total * A_y

        # 지진 하중은 층별 분배 로직 적용
        # factors["Ex"]가 1.0이면 X방향 지진력, factors["Ey"]가 1.0이면 Y방향 지진력 적용
        # Ex, Ey는 1.0 또는 0.3 (직교 효과) 또는 0
        
        # 층별 지진력 및 풍하중 적용
        if abs(factors["Ex"]) > 1e-9 or abs(factors["Ey"]) > 1e-9 or abs(factors["Wx"]) > 1e-9 or abs(factors["Wy"]) > 1e-9:
            for k in range(cfg.FLOORS): # 0-indexed floor
                # 지진력
                Fx_seismic = story_seismic_forces_x[k] * factors["Ex"]
                Fy_seismic = story_seismic_forces_y[k] * factors["Ey"]
                
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
        
        analysis_result = ops.analyze(1)
        if analysis_result != 0: 
            print(f"DEBUG: Analysis failed for combo {combo_name}, result: {analysis_result}")
            analysis_ok = False; ops.remove('loadPattern', pattern_tag); break
        
        df_max_curr = extract_local_element_forces(column_elem_ids, beam_elem_ids)
        if df_max_curr.empty: 
            print(f"DEBUG: extract_local_element_forces returned empty for combo {combo_name}")
            analysis_ok = False; ops.remove('loadPattern', pattern_tag); break
        
        df_max_curr['Combo'] = combo_name
        all_max_combo_forces.append(df_max_curr)
        ops.remove('loadPattern', pattern_tag)

    if not analysis_ok or not all_max_combo_forces:
        print(f"DEBUG: Final check failed: analysis_ok={analysis_ok}, all_max_combo_forces={len(all_max_combo_forces) if all_max_combo_forces else 0}")
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
        # --- X방향 층간변위 (지진하중) ---
        ops.reset(); ops.pattern('Plain', 101, 1)
        drift_factors_x_seismic = next((f for name, f in LOAD_COMBINATIONS if name == "ASCE-S-E1"), None) # 1.0D + 1.0L + 0.7Ex + 0.21Ey
        
        if drift_factors_x_seismic:
            for k in range(1,cfg.FLOORS+1):
                master_node_id = node_map.get((k, 0))
                if master_node_id:
                    # 해당 층의 총 지진력에 조합계수 Ex를 곱하여 마스터 노드에 재하
                    # ASCE-S-E1은 0.7Ex 이므로, story_seismic_forces[k-1]에 0.7을 곱함.
                    # story_seismic_forces는 이미 각 층에 분배된 지진력의 총합이므로, nodal_load_x로 나눌 필요 없음
                                        Fx_story_load = story_seismic_forces_x[k-1] * drift_factors_x_seismic["Ex"]
                                        Fy_story_load = story_seismic_forces_y[k-1] * drift_factors_x_seismic["Ey"] # 직교효과                    ops.load(master_node_id, Fx_story_load, Fy_story_load, 0, 0, 0, 0) # 마스터 노드에 직접 재하

            if ops.analyze(1) == 0:
                for k in range(1, cfg.FLOORS + 1):
                    master_node_upper = node_map.get((k, 0))
                    master_node_lower = node_map.get((k - 1, 0)) # 기초는 0층이므로 0.0 변위
                    if master_node_upper:
                        disp_upper_x = ops.nodeDisp(master_node_upper, 1)
                        disp_lower_x = ops.nodeDisp(master_node_lower, 1) if master_node_lower else 0.0
                        drift_x = abs(disp_upper_x - disp_lower_x) / cfg.H
                        story_drifts_x.append(drift_x)
                        # print(f"DEBUG Drift X: Floor {k}, Upper Node {master_node_upper} DispX: {disp_upper_x:.6f}, Lower Node {master_node_lower} DispX: {disp_lower_x:.6f}, Drift: {drift_x:.6f}") # DEBUG
                else:
                    story_drifts_x.append(0.0)
        
        # --- Y방향 층간변위 (지진하중) ---
        ops.reset(); ops.pattern('Plain', 102, 1)
        drift_factors_y_seismic = next((f for name, f in cfg.LOAD_COMBINATIONS if name == "ASCE-S-E5"), None) # 1.0D + 1.0L + 0.21Ex + 0.7Ey
        
        if drift_factors_y_seismic:
            for k in range(1,cfg.FLOORS+1):
                master_node_id = node_map.get((k, 0))
                if master_node_id:
                    Fx_story_load = story_seismic_forces_x[k-1] * drift_factors_y_seismic["Ex"] # 직교효과
                    Fy_story_load = story_seismic_forces_y[k-1] * drift_factors_y_seismic["Ey"] # 0-indexed story_seismic_forces
                    ops.load(master_node_id, Fx_story_load, Fy_story_load, 0, 0, 0, 0) # 마스터 노드에 직접 재하

            if ops.analyze(1) == 0:
                for k in range(1, cfg.FLOORS + 1):
                    master_node_upper = node_map.get((k, 0))
                    master_node_lower = node_map.get((k - 1, 0))
                    if master_node_upper:
                        disp_upper_y = ops.nodeDisp(master_node_upper, 2)
                        disp_lower_y = ops.nodeDisp(master_node_lower, 2) if master_node_lower else 0.0
                        drift_y = abs(disp_upper_y - disp_lower_y) / cfg.H
                        story_drifts_y.append(drift_y)
                        # print(f"DEBUG Drift Y: Floor {k}, Upper Node {master_node_upper} DispY: {disp_upper_y:.6f}, Lower Node {master_node_lower} DispY: {disp_lower_y:.6f}, Drift: {drift_y:.6f}") # DEBUG
                else:
                    story_drifts_y.append(0.0)

        max_drift_x = max(story_drifts_x) if story_drifts_x else 0
        max_drift_y = max(story_drifts_y) if story_drifts_y else 0
        actual_drift_ratio = max(max_drift_x, max_drift_y) / allowable_drift_ratio
    else: actual_drift_ratio = float('inf')

    wind_disps_x, wind_disps_y = [], []; actual_wind_disp_ratio = 0.0
    if analysis_ok:
        # --- X 방향 풍하중 변위 ---
        actual_wind_disp_ratio_x = float('inf')
        ops.reset(); ops.pattern('Plain', 201, 1)
        wind_factors_x = next((f for name, f in cfg.LOAD_COMBINATIONS if name == "ASCE-S-W1"), None) # 1.0D + 0.5L + 1.0Wx
        if wind_factors_x:
            # base_force_x = Wx * wind_factors_x["Wx"] # 기존
            for k in range(1, cfg.FLOORS + 1):
                # story_force_x = base_force_x * (k / lateral_force_dist_sum) # 기존
                master_node_id = node_map.get((k, 0))
                if master_node_id:
                    # story_wind_forces_x[k-1]에 조합계수 Wx를 곱하여 마스터 노드에 재하
                    Fx_story_load = story_wind_forces_x[k-1] * wind_factors_x["Wx"]
                    ops.load(master_node_id, Fx_story_load, 0, 0, 0, 0, 0)
            if ops.analyze(1) == 0:
                disps = []
                for k in range(1, cfg.FLOORS + 1):
                    master_node_id = node_map.get((k, 0))
                    if master_node_id: disps.append(abs(ops.nodeDisp(master_node_id, 1)))
                wind_disps_x = disps
                if wind_disps_x:
                    actual_wind_disp_ratio_x = wind_disps_x[-1] / ((cfg.FLOORS * cfg.H) / 400.0)
        
        # --- Y 방향 풍하중 변위 ---
        actual_wind_disp_ratio_y = float('inf')
        ops.reset(); ops.pattern('Plain', 202, 1)
        wind_factors_y = next((f for name, f in cfg.LOAD_COMBINATIONS if name == "ASCE-S-W3"), None) # 1.0D + 0.5L + 1.0Wy
        if wind_factors_y:
            # base_force_y = Wy * wind_factors_y["Wy"] # 기존
            for k in range(1, cfg.FLOORS + 1):
                # story_force_y = base_force_y * (k / lateral_force_dist_sum) # 기존
                master_node_id = node_map.get((k, 0))
                if master_node_id:
                    # story_wind_forces_y[k-1]에 조합계수 Wy를 곱하여 마스터 노드에 재하
                    Fy_story_load = story_wind_forces_y[k-1] * wind_factors_y["Wy"]
                    ops.load(master_node_id, 0, Fy_story_load, 0, 0, 0, 0)
            if ops.analyze(1) == 0:
                disps = []
                for k in range(1, cfg.FLOORS + 1):
                    master_node_id = node_map.get((k, 0))
                    if master_node_id: disps.append(abs(ops.nodeDisp(master_node_id, 2)))
                wind_disps_y = disps
                if wind_disps_y:
                    actual_wind_disp_ratio_y = wind_disps_y[-1] / ((cfg.FLOORS * cfg.H) / 400.0)
        actual_wind_disp_ratio = max(actual_wind_disp_ratio_x, actual_wind_disp_ratio_y)

    # --- SCWB (Strong Column - Weak Beam) Check ---
    # Requirement: sum(M_nc) >= 1.2 * sum(M_nb) at every joint
    scwb_ratios = []
    
    # 1. Map beams to nodes to sum beam capacities at each joint
    # node_beams_x/y keys: (floor_idx (1..cfg.FLOORS), loc_idx (0..num_locations-1))
    node_beams_x = {} 
    node_beams_y = {}
    
    for beam_idx, (u, v) in enumerate(cfg.BEAM_CONNECTIONS):
        # Determine beam orientation
        ux, uy = cfg.COLUMN_LOCATIONS[u]
        vx, vy = cfg.COLUMN_LOCATIONS[v]
        is_x_beam = abs(uy - vy) < 1e-4 # Same Y-coord -> Beam is along X
        
        for k in range(1, cfg.FLOORS + 1):
            abs_beam_idx = (k - 1) * len(cfg.BEAM_CONNECTIONS) + beam_idx
            group_idx = beam_map[num_columns + abs_beam_idx + 1]
            sec_idx = beam_indices[group_idx]
            
            # Beam Nominal Moment Capacity (Mn_z)
            mn_beam = beam_sections_df.iloc[sec_idx]['PiM']
            
            # Add to both end nodes
            for node_idx in [u, v]:
                node_key = (k, node_idx)
                if is_x_beam:
                    if node_key not in node_beams_x: node_beams_x[node_key] = []
                    node_beams_x[node_key].append(mn_beam)
                else:
                    if node_key not in node_beams_y: node_beams_y[node_key] = []
                    node_beams_y[node_key].append(mn_beam)

    # 2. Iterate all joints to check SCWB
    for k in range(1, cfg.FLOORS + 1):
        for i in range(num_locations):
            sum_mb_x = sum(node_beams_x.get((k, i), []))
            sum_mb_y = sum(node_beams_y.get((k, i), []))
            
            if sum_mb_x == 0 and sum_mb_y == 0: continue
            
            # Identify columns framing into this joint
            cols_to_check = []
            if k >= 1: cols_to_check.append( (k-1) * num_locations + i ) # Column Below
            if k < cfg.FLOORS: cols_to_check.append( k * num_locations + i ) # Column Above
            
            sum_mc_for_x_beams = 0.0 # Resisting Moment against X-Beams (needs M about Global Y)
            sum_mc_for_y_beams = 0.0 # Resisting Moment against Y-Beams (needs M about Global X)
            
            for c_idx in cols_to_check:
                group_idx = col_map[c_idx + 1]
                sec_idx = col_indices[group_idx]
                rot = col_rotations[group_idx] if len(col_rotations) > 0 else 0
                
                # Get Column Capacity at P=0 (Pure Bending) for conservatism
                pm_df = load_pm_data_for_column(h5_file, sec_idx)
                pn0_z, mn0_z = get_pm_capacity_from_df(0, pm_df, axis='z') # Strong Axis Capacity
                pn0_y, mn0_y = get_pm_capacity_from_df(0, pm_df, axis='y') # Weak Axis Capacity
                
                # Orientation Mapping:
                # Rot 0: Local z // Global X, Local y // Global Y
                #   -> X-Beam (Global Y moment) resisted by Mn_y (Weak)
                #   -> Y-Beam (Global X moment) resisted by Mn_z (Strong)
                # Rot 1: Local z // Global Y, Local y // Global X
                #   -> X-Beam (Global Y moment) resisted by Mn_z (Strong)
                #   -> Y-Beam (Global X moment) resisted by Mn_y (Weak)
                
                if rot == 0:
                    sum_mc_for_x_beams += mn0_y
                    sum_mc_for_y_beams += mn0_z
                else:
                    sum_mc_for_x_beams += mn0_z
                    sum_mc_for_y_beams += mn0_y
            
            # Calculate DCR: Demand (1.2*Beam) / Capacity (Col)
            if sum_mb_x > 0:
                scwb_ratios.append( (1.2 * sum_mb_x) / (sum_mc_for_x_beams + 1e-9) )
            if sum_mb_y > 0:
                scwb_ratios.append( (1.2 * sum_mb_y) / (sum_mc_for_y_beams + 1e-9) )
    
    actual_hierarchy_ratio = max(scwb_ratios) if scwb_ratios else 0.0

    total_cost, total_co2 = 0, 0
    for i in range(num_columns):
        group_idx = col_map[i + 1]; sec_idx = col_indices[group_idx]
        # 콘크리트 및 철근 비용
        total_cost += column_sections_df.iloc[sec_idx]['Cost'] * cfg.H
        total_co2 += column_sections_df.iloc[sec_idx]['CO2'] * cfg.H
        # 거푸집 비용 (기둥)
        b_c, h_c = column_sections[sec_idx]
        column_formwork_area = 2 * (b_c + h_c) * cfg.H
        total_cost += column_formwork_area * cfg.FORMWORK_UNIT_COST
    for k in range(cfg.FLOORS):
        for i in range(len(cfg.BEAM_CONNECTIONS)):
            abs_beam_idx = k * len(cfg.BEAM_CONNECTIONS) + i; group_idx = beam_map[num_columns + abs_beam_idx + 1]
            sec_idx = beam_indices[group_idx]
            # 콘크리트 및 철근 비용
            total_cost += beam_sections_df.iloc[sec_idx]['Cost'] * beam_lengths[i]
            total_co2 += beam_sections_df.iloc[sec_idx]['CO2'] * beam_lengths[i]
            # 거푸집 비용 (보)
            b_b, h_b = beam_sections[sec_idx]
            beam_formwork_area = 2 * (b_b + h_b) * beam_lengths[i] # 상부 슬래브 접면 제외
            total_cost += beam_formwork_area * cfg.FORMWORK_UNIT_COST

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
