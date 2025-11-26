import openseespy.opensees as ops
import matplotlib.pyplot as plt
import opsvis as opsv

# --- Matplotlib 전역 글꼴 설정 ---
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우 'Malgun Gothic', Mac의 경우 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# =================================================================
# ===                   1. 구조물 형상 정보 정의                  ===
# =================================================================

# --- 1.1. 건물 기본 정보 ---
floors = 4
H = 4.0

# --- 1.2. 건물 평면 형상 정보 ---
# simpledata_2obj_NSGA2.py 파일의 비정형 평면과 동일
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

# =================================================================
# ===                 2. OpenSees 모델 생성 함수                  ===
# =================================================================

def build_simple_model(floors, H, column_locations, beam_connections):
    """
    시각화만을 위한 최소한의 OpenSees 모델을 생성합니다.
    재료나 단면 정의 없이 노드와 요소의 위치 정보만으로 구성됩니다.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # 1. 노드(절점) 생성
    node_map = {}
    node_id_counter = 1
    for k in range(floors + 1):
        for i, (x, y) in enumerate(column_locations):
            ops.node(node_id_counter, x, y, k * H)
            node_map[(k, i)] = node_id_counter
            if k == 0:
                ops.fix(node_id_counter, 1, 1, 1, 1, 1, 1) # 기초는 고정단으로 설정
            node_id_counter += 1

    # 2. 더미 재료 및 단면 정의 (형상만 필요하므로 값은 중요하지 않음)
    # 기둥과 보에 대한 별도의 기하학적 변환을 정의합니다.
    # 기둥 변환 (tag=1): 부재의 로컬 z축이 글로벌 X축과 평행하도록 설정
    ops.geomTransf('Linear', 1, 1, 0, 0)
    # 보 변환 (tag=2): 부재의 로컬 z축이 글로벌 Z축과 평행하도록(수직) 설정
    ops.geomTransf('Linear', 2, 0, 0, 1)
    
    # 3. 부재(요소) 생성
    elem_id_counter = 1
    # 기둥 생성
    for k in range(floors):
        for i in range(len(column_locations)):
            n1 = node_map[(k, i)]
            n2 = node_map[(k + 1, i)]
            # 기둥에 대해 변환 태그 1번을 적용합니다.
            ops.element('elasticBeamColumn', elem_id_counter, n1, n2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1) 
            elem_id_counter += 1
    
    # 보 생성
    for k in range(1, floors + 1):
        for (loc_idx1, loc_idx2) in beam_connections:
            n1 = node_map[(k, loc_idx1)]
            n2 = node_map[(k, loc_idx2)]
            # 보에 대해 변환 태그 2번을 적용합니다.
            ops.element('elasticBeamColumn', elem_id_counter, n1, n2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2)
            elem_id_counter += 1

def plot_2d_plan(ax, column_locations, beam_connections):
    """주어진 축(ax)에 2D 평면도를 그립니다."""
    ax.set_title("2D 평면도 (Plan View)")
    ax.clear()
    # 보 그리기
    for (idx1, idx2) in beam_connections:
        p1, p2 = column_locations[idx1], column_locations[idx2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=3, zorder=1)
    # 기둥 그리기
    xs = [loc[0] for loc in column_locations]
    ys = [loc[1] for loc in column_locations]
    ax.scatter(xs, ys, c='b', s=100, zorder=2, label='Columns')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True)

if __name__ == '__main__':
    # 1. OpenSees 모델 생성
    build_simple_model(floors, H, column_locations, beam_connections)
    print("OpenSees 모델이 생성되었습니다.")

    # 2. 시각화
    print("구조물 형상을 시각화합니다...")
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('구조물 형상 시각화 (Structural Shape Visualization)', fontsize=20)

    # --- Plot 1: 2D 평면도 ---
    ax1 = fig.add_subplot(2, 2, 1)
    plot_2d_plan(ax1, column_locations, beam_connections)

    # --- Plot 2: 3D 기본 뷰 ---
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    opsv.plot_model(ax=ax2, node_labels=0, element_labels=0, az_el=(-60, 30))
    ax2.set_title("3D 기본 뷰")

    # --- Plot 3: 3D 절점 번호 뷰 ---
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    opsv.plot_model(ax=ax3, node_labels=1, element_labels=0, az_el=(-60, 30))
    ax3.set_title("3D 뷰 (절점 번호 포함)")

    # --- Plot 4: 3D 부재 번호 뷰 ---
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    opsv.plot_model(ax=ax4, node_labels=0, element_labels=1, az_el=(-60, 30))
    ax4.set_title("3D 뷰 (부재 번호 포함)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../results/structure_visualization.png")
    print("시각화 결과가 'structure_visualization.png' 파일로 저장되었습니다.")
    plt.show()