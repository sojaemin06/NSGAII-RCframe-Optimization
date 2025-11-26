
import math
import numpy as np
from scipy.spatial import ConvexHull

def get_grouping_maps(GROUPING_STRATEGY, floors, column_locations, beam_connections):
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
    
    return col_map, beam_map, num_col_groups, num_beam_groups
