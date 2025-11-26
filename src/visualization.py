
import matplotlib.pyplot as plt
import opsvis as opsv
import os

def plot_Structure(column_locations, beam_connections, title='Structure Shape', view='3D', ax=None):
    """건물 구조의 형상을 2D 또는 3D로 시각화하는 함수."""
    if view == '2D_plan':
        if ax is None: fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title); ax.clear()
        for (idx1, idx2) in beam_connections:
            p1, p2 = column_locations[idx1], column_locations[idx2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=3, zorder=1)
        xs, ys = [loc[0] for loc in column_locations], [loc[1] for loc in column_locations]
        ax.scatter(xs, ys, c='b', s=100, zorder=2, label='Columns')
        ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(); ax.grid(True)
    elif view == '3D':
        plt.figure(figsize=(12, 10))
        opsv.plot_model(node_labels=0, element_labels=0, az_el=(-60, 30))
        plt.title(title)

def visualize_load_patterns(column_locations, beam_connections, patterns_by_floor, output_folder="."):
    """
    주어진 하중 패턴을 각 층의 평면도에 시각화하고 이미지 파일로 저장하는 함수.
    """
    print("\n[Visualization] Generating load pattern plots...")
    
    num_floors = len(patterns_by_floor)
    # 층 수에 맞춰 subplot 개수 동적 조정
    ncols = min(num_floors, 4) 
    nrows = (num_floors - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle('Visualization of Applied Load Patterns by Floor', fontsize=16)

    for i, (floor_num, loaded_indices) in enumerate(patterns_by_floor.items()):
        ax = axes[i // ncols, i % ncols]
        ax.set_title(f"Floor {floor_num} Load Pattern")
        
        # 모든 기둥 위치 그리기
        xs = [loc[0] for loc in column_locations]
        ys = [loc[1] for loc in column_locations]
        ax.scatter(xs, ys, c='black', s=50, zorder=2)

        # 모든 보와 인덱스 번호 그리기 (기본 색상)
        for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=2, zorder=1)
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(center_x, center_y, str(conn_idx), color='gray', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

        # 하중이 재하된 보 그리기 (강조 색상)
        for conn_idx in loaded_indices:
            p1_idx, p2_idx = beam_connections[conn_idx]
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=4, zorder=3, label='Loaded Beams' if 'Loaded Beams' not in [l.get_label() for l in ax.get_lines()] else "")

        ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0: ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_folder, "load_pattern_visualization.png")
    plt.savefig(save_path); print(f"Load pattern visualization saved to '{save_path}'")
    # plt.show()
