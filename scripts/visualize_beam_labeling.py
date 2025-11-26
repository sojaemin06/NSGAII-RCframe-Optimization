import matplotlib.pyplot as plt
import numpy as np

# --- Matplotlib 한글 글꼴 설정 ---
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
# plt.rcParams['font.family'] = 'AppleGothic' # Mac
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# --- simpledata_2obj_NSGA2.py 에서 가져온 구조 정보 ---
column_locations = [(0, 0), (5, 0), (10, 0), (15, 0),
                    (0, 6), (5, 6), (10, 6), (15, 6),
                    (0, 10), (5, 10), (10, 10), (15, 10),
                    (5, 15), (10, 15), (15, 15)]

beam_connections = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14),
    (0, 4), (4, 8),
    (1, 5), (5, 9), (9, 12),
    (2, 6), (6, 10), (10, 13),
    (3, 7), (7, 11), (11, 14)
]

# --- 사용자가 정의한 하중 패턴 ---
patterns_by_floor = {
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20],
    2: [1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    3: [3, 4, 6, 7, 9, 12, 14, 15, 17, 18],
    4: [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 19, 20]
}

def visualize_beam_labeling(column_locations, beam_connections, patterns_by_floor):
    """
    평면도에 보의 인덱스(0~21)를 라벨링하고, 각 층의 하중 패턴을 시각화합니다.
    """
    num_floors = len(patterns_by_floor)
    ncols = min(num_floors, 2)
    nrows = (num_floors - 1) // ncols + 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 9 * nrows), squeeze=False)
    fig.suptitle('층별 하중 재하 패턴 시각화 (보 인덱스 기준)', fontsize=20)

    for i, (floor_num, loaded_indices) in enumerate(patterns_by_floor.items()):
        ax = axes[i // ncols, i % ncols]
        ax.set_title(f"{floor_num}층 하중 패턴", fontsize=16)
        
        # 1. 모든 보를 회색으로 그립니다.
        for p1_idx, p2_idx in beam_connections:
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=3, zorder=1)

        # 2. 하중이 재하된 보를 빨간색으로 덧그립니다.
        for conn_idx in loaded_indices:
            p1_idx, p2_idx = beam_connections[conn_idx]
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=5, zorder=2, 
                    label='재하된 보' if '재하된 보' not in [l.get_label() for l in ax.get_lines()] else "")

        # 3. 모든 기둥을 파란색 점으로 그립니다.
        xs = [loc[0] for loc in column_locations]
        ys = [loc[1] for loc in column_locations]
        ax.scatter(xs, ys, c='blue', s=120, zorder=3, label='기둥')

        # 4. 각 보의 중앙에 인덱스 번호(0~21)를 라벨링합니다.
        for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(center_x, center_y, str(conn_idx), color='black', fontsize=12, weight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X 좌표 (m)', fontsize=12); ax.set_ylabel('Y 좌표 (m)', fontsize=12)
        ax.legend(fontsize=12); ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../results/beam_labeling_by_floor.png")
    print("시각화 결과가 'beam_labeling_by_floor.png' 파일로 저장되었습니다.")
    plt.show()

if __name__ == '__main__':
    visualize_beam_labeling(column_locations, beam_connections, patterns_by_floor)