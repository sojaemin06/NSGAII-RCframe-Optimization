
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import openseespy.opensees as ops
import h5py
import os

from src.config import *

# =================================================================
# ===                  데이터 로드 및 처리                        ===
# =================================================================

def load_section_data():
    """CSV 파일에서 단면 정보를 로드합니다."""
    beam_sections_df = pd.read_csv("beam_sections_simple02.csv")
    column_sections_df = pd.read_csv("column_sections_simple02.csv")
    
    beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
    column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
    
    return beam_sections_df, column_sections_df, beam_sections, column_sections

def get_beam_lengths(column_locations, beam_connections):
    return [math.sqrt((column_locations[p2][0] - column_locations[p1][0])**2 + 
                      (column_locations[p2][1] - column_locations[p1][1])**2) 
            for p1, p2 in beam_connections]

def calculate_fixed_scale(column_sections_df, beam_sections_df, total_column_length, total_beam_length):
    """데이터 기반 고정 스케일(정규화용)을 계산합니다."""
    min_col_cost_per_m = column_sections_df['Cost'].min()
    max_col_cost_per_m = column_sections_df['Cost'].max()
    min_beam_cost_per_m = beam_sections_df['Cost'].min()
    max_beam_cost_per_m = beam_sections_df['Cost'].max()
    
    min_col_co2_per_m = column_sections_df['CO2'].min()
    max_col_co2_per_m = column_sections_df['CO2'].max()
    min_beam_co2_per_m = beam_sections_df['CO2'].min()
    max_beam_co2_per_m = beam_sections_df['CO2'].max()

    fixed_min_cost = (min_col_cost_per_m * total_column_length) + (min_beam_cost_per_m * total_beam_length)
    fixed_max_cost = (max_col_cost_per_m * total_column_length) + (max_beam_cost_per_m * total_beam_length)
    fixed_range_cost = fixed_max_cost - fixed_min_cost
    if fixed_range_cost == 0: fixed_range_cost = 1.0

    fixed_min_co2 = (min_col_co2_per_m * total_column_length) + (min_beam_co2_per_m * total_beam_length)
    fixed_max_co2 = (max_col_co2_per_m * total_column_length) + (max_beam_co2_per_m * total_beam_length)
    fixed_range_co2 = fixed_max_co2 - fixed_min_co2
    if fixed_range_co2 == 0: fixed_range_co2 = 1.0
    
    return fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2

# =================================================================
# ===                  그룹핑 전략 로직                           ===
# =================================================================

def get_grouping_maps(strategy, num_locations, num_columns, num_beams, floors, beam_connections, column_locations):
    col_map, beam_map = {}, {}
    num_col_groups, num_beam_groups = 0, 0 # Initialize
    
    if strategy == "Individual":
        num_col_groups, num_beam_groups = num_columns, num_beams
        for i in range(num_columns): col_map[i + 1] = i
        for i in range(num_beams): beam_map[num_columns + i + 1] = i
        
    elif strategy == "Uniform":
        num_col_groups, num_beam_groups = 1, 1
        for i in range(num_columns): col_map[i + 1] = 0
        for i in range(num_beams): beam_map[num_columns + i + 1] = 0
        
    elif strategy == "ByFloor":
        num_col_groups, num_beam_groups = floors, floors
        cols_per_floor = num_locations
        beams_per_floor = len(beam_connections)
        for k in range(floors):
            for i in range(cols_per_floor): col_map[k * cols_per_floor + i + 1] = k
            for i in range(beams_per_floor): beam_map[num_columns + k * beams_per_floor + i + 1] = k
            
    elif strategy == "Hybrid":
        print("\n[Hybrid Grouping] Analyzing column connectivity and local geometry for grouping...")
        
        node_connectivity = {i: 0 for i in range(len(column_locations))}
        for p1_idx, p2_idx in beam_connections:
            node_connectivity[p1_idx] += 1
            node_connectivity[p2_idx] += 1

        memo_col_type = {}
        
        def is_point_inside_hull(point, hull):
            return np.all(np.add(np.dot(hull.equations[:, :-1], point), hull.equations[:, -1]) < 1e-9)

        def get_col_loc_type(loc_idx):
            if loc_idx in memo_col_type:
                return memo_col_type[loc_idx]

            connections = node_connectivity.get(loc_idx, 0)
            loc_type = -1 

            if connections <= 2:
                loc_type = 0  # 코너
            elif connections == 3:
                loc_type = 1  # 엣지
            else:
                neighbor_indices = {p[1] if p[0] == loc_idx else p[0] for p in beam_connections if loc_idx in p}
                if len(neighbor_indices) < 3:
                    loc_type = 2
                else:
                    neighbor_points = np.array([column_locations[i] for i in neighbor_indices])
                    current_point = np.array(column_locations[loc_idx])
                    try:
                        neighbors_hull = ConvexHull(neighbor_points)
                        if is_point_inside_hull(current_point, neighbors_hull):
                            loc_type = 2
                        else:
                            loc_type = 0
                    except Exception:
                        loc_type = 2
            memo_col_type[loc_idx] = loc_type
            return loc_type

        print("[Hybrid Grouping] Analyzing beam location using global Convex Hull...")
        points = np.array(column_locations)
        hull = ConvexHull(points)
        equations = hull.equations
        perimeter_beam_indices = set()
        for i, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            for eq in equations:
                on_line1 = abs(eq[0] * p1[0] + eq[1] * p1[1] + eq[2]) < 1e-9
                on_line2 = abs(eq[0] * p2[0] + eq[1] * p2[1] + eq[2]) < 1e-9
                if on_line1 and on_line2:
                    perimeter_beam_indices.add(i)
                    break

        floor_step = 2
        num_floor_groups = math.ceil(floors / floor_step)
        num_col_groups = num_floor_groups * 3
        num_beam_groups = num_floor_groups * 2
        
        for abs_col_idx in range(num_columns):
            floor_idx, loc_idx = divmod(abs_col_idx, num_locations)
            floor_group_idx = floor_idx // floor_step
            loc_type = get_col_loc_type(loc_idx)
            group_id = floor_group_idx * 3 + loc_type
            col_map[abs_col_idx + 1] = group_id
            
        for abs_beam_idx in range(num_beams):
            floor_idx, conn_idx = divmod(abs_beam_idx, len(beam_connections))
            floor_group_idx = floor_idx // floor_step
            loc_type = 0 if conn_idx in perimeter_beam_indices else 1
            group_id = floor_group_idx * 2 + loc_type
            beam_map[num_columns + abs_beam_idx + 1] = group_id
            
    return num_col_groups, num_beam_groups, col_map, beam_map

# =================================================================
# ===                  강도 계산 유틸리티                         ===
# =================================================================

def load_pm_data_for_column(h5_file, column_index):
    try:
        ref = h5_file['Column_Mdata'][0, column_index]
        pm_data = h5_file[ref][()]
        return pd.DataFrame(pm_data, columns=['Mnb_z','Pnb_z','Mnb_y','Pnb_y','Alpha_PI_Mnb_z','Alpha_PI_Pnb_z','Alpha_PI_Mnb_y','Alpha_PI_Pnb_y'])
    except Exception:
        return pd.DataFrame()

def get_pm_capacity_from_df(slope, pm_df, axis='z'):
    if pm_df.empty: return 0.0, 0.0
    moment_col, axial_col = (f'Alpha_PI_Mnb_{axis}', f'Alpha_PI_Pnb_{axis}')
    design_m, design_p = pm_df[moment_col].values, pm_df[axial_col].values
    curve_slopes = abs(design_p) / (abs(design_m) + 1e-9)
    for i in range(len(curve_slopes) - 1):
        if curve_slopes[i] >= slope >= curve_slopes[i+1]:
            m1, p1, m2, p2 = design_m[i], design_p[i], design_m[i+1], design_p[i+1]
            if abs(m2 - m1) < 1e-9: continue
            a1 = (p2 - p1) / (m2 - m1); b1 = p1 - a1 * m1
            if abs(a1 - slope) < 1e-9: continue
            Mn = -b1 / (a1 - slope)
            Pn = slope * Mn
            return Pn, Mn
    return 0.0, 0.0

def get_precalculated_strength(element_type, index, col_df, beam_df):
    strengths = {}
    try:
        if element_type == 'Column':
            row = col_df.iloc[index]
            strengths.update({'Pn': row['PI_Pn_max'], 'Vn_y': row['PI_Vn_y'], 'Vn_z': row['PI_Vn_z']})
        elif element_type == 'Beam':
            row = beam_df.iloc[index]
            strengths.update({'Pn': float('inf'), 'Vn_y': float('inf'), 'Vn_z': row['PiVn'], 'Mn_y': float('inf'), 'Mn_z': row['PiM']})
    except IndexError: 
        return {'Pn': 0, 'Vn_y': 0, 'Vn_z': 0, 'Mn_y': 0, 'Mn_z': 0}
    return strengths

# =================================================================
# ===                  시각화 유틸리티                            ===
# =================================================================

def plot_Structure(title='Structure Shape', view='3D', ax=None, column_locations=COLUMN_LOCATIONS, beam_connections=BEAM_CONNECTIONS):
    import opsvis as opsv # 내부 import
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
    print("\n[Visualization] Generating load pattern plots...")
    
    num_floors = len(patterns_by_floor)
    ncols = min(num_floors, 4) 
    nrows = (num_floors - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle('Visualization of Applied Load Patterns by Floor', fontsize=16)

    for i, (floor_num, loaded_indices) in enumerate(patterns_by_floor.items()):
        ax = axes[i // ncols, i % ncols]
        ax.set_title(f"Floor {floor_num} Load Pattern")
        
        xs = [loc[0] for loc in column_locations]
        ys = [loc[1] for loc in column_locations]
        ax.scatter(xs, ys, c='black', s=50, zorder=2)

        for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=2, zorder=1)
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(center_x, center_y, str(conn_idx), color='gray', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

        for conn_idx in loaded_indices:
            p1_idx, p2_idx = beam_connections[conn_idx]
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=4, zorder=3, label='Loaded Beams' if 'Loaded Beams' not in [l.get_label() for l in ax.get_lines()] else '')

        ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0: ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_folder, "load_pattern_visualization.png")
    plt.savefig(save_path)
    plt.close(fig) # Close figure to prevent display
    print(f"Load pattern visualization saved to '{save_path}'")
