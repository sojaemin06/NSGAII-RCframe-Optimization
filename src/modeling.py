
import openseespy.opensees as ops

def build_model_for_section(floors, H, column_locations, beam_connections, col_indices, col_rotations, beam_indices, col_map, beam_map, column_sections, beam_sections, num_columns):
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
    # P-Delta 효과 고려를 위해 Linear -> PDelta 변경 (R2-7 반영)
    # 1: Column (Standard), 2: Column (Rotated), 3: Beam
    ops.geomTransf('PDelta', 1, 1, 0, 0)
    ops.geomTransf('PDelta', 2, 0, 1, 0)
    ops.geomTransf('PDelta', 3, 0, 0, 1)

    # --- Rigid Diaphragm (강체 격막) 적용 ---
    # 각 층(1~floors)의 질량 중심(CM)을 계산하고 마스터 노드를 생성하여 rigidDiaphragm 구속 적용
    master_nodes = {}
    # CM 계산 (모든 기둥 위치의 평균으로 가정)
    avg_x = sum(loc[0] for loc in column_locations) / len(column_locations)
    avg_y = sum(loc[1] for loc in column_locations) / len(column_locations)

    # 마스터 노드용 ID 시작 번호 (충분히 큰 값 사용)
    master_node_start_id = 10000

    for k in range(1, floors + 1):
        master_id = master_node_start_id + k
        # 마스터 노드 생성 (CM 위치)
        ops.node(master_id, avg_x, avg_y, k * H)
        # 마스터 노드 구속: 횡방향(1,2)과 회전(6)은 자유, 수직(3) 및 기울기(4,5)는 구속?
        # 일반적으로 마스터 노드는 6자유도를 가지며, 하중을 받아서 전달함.
        # 다만, 강체 격막 모델에서는 바닥판 내의 변형이 없으므로 마스터 노드에 집중됨.
        # 별도 fix는 하지 않고 하중점 역할만 수행하게 함 (또는 해석 안정성을 위해 Z축 등 불필요 자유도 구속 가능)
        # 여기서는 6자유도 모두 풀어둠 (필요시 수정)
        
        master_nodes[k] = master_id
        
        # 해당 층의 모든 기둥 절점(슬레이브 노드) 수집
        slave_nodes = [node_map[(k, i)] for i in range(len(column_locations))]
        
        # Rigid Diaphragm 적용
        # dir=3 (Z축이 법선인 X-Y 평면 격막)
        ops.rigidDiaphragm(3, master_id, *slave_nodes)

    column_elem_ids, beam_elem_ids = [], []; elem_id_counter = 1
    num_locations = len(column_locations)
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
    return column_elem_ids, beam_elem_ids, node_map, master_nodes
