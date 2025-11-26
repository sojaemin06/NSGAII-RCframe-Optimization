
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
    ops.geomTransf('Linear', 1, 1, 0, 0); ops.geomTransf('Linear', 2, 0, 1, 0); ops.geomTransf('Linear', 3, 0, 0, 1)
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
    return column_elem_ids, beam_elem_ids, node_map
