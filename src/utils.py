import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import openseespy.opensees as ops
import h5py
import os

import src.config as cfg

# =================================================================
# ===                  데이터 로드 및 처리                        ===
# =================================================================

# 전역 매핑 변수 (Reduced ID -> Original ID)
COLUMN_ID_MAPPING = {}

def load_section_data(beam_path="beam_sections_reduced.csv", col_path="column_sections_reduced.csv"):
    """CSV 파일에서 단면 정보를 로드합니다. (기본값: 축소된 DB)"""
    global COLUMN_ID_MAPPING
    
    if not os.path.exists(col_path):
        print(f"Warning: {col_path} not found. Falling back to original.")
        col_path = "column_sections_simple02.csv"
        beam_path = "beam_sections_simple02.csv"
    
    beam_sections_df = pd.read_csv(beam_path)
    column_sections_df = pd.read_csv(col_path)
    
    beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
    column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
    
    if "reduced" in col_path and "expanded" not in col_path:
        mapping_path = "column_id_mapping.csv"
        if os.path.exists(mapping_path):
            map_df = pd.read_csv(mapping_path)
            COLUMN_ID_MAPPING = dict(zip(map_df['Reduced_ID'] - 1, map_df['Original_ID'] - 1))
        else:
            COLUMN_ID_MAPPING = {i: i for i in range(len(column_sections))}
    elif "expanded" in col_path:
        # For expanded DB, map current index to original_name (which corresponds to HDF5 index)
        # Note: 'original_name' in CSV is 1-based, we need 0-based for HDF5
        if 'original_name' in column_sections_df.columns:
            COLUMN_ID_MAPPING = {i: int(row['original_name']) - 1 for i, row in column_sections_df.iterrows()}
        else:
            print("Warning: 'original_name' column not found in expanded DB. PM data mapping may fail.")
            COLUMN_ID_MAPPING = {i: i for i in range(len(column_sections))}
    else:
        COLUMN_ID_MAPPING = {i: i for i in range(len(column_sections))}
    
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

def get_grouping_maps(strategy, num_locations, num_columns, num_beams, floors, beam_connections, column_locations):
    col_map, beam_map = {}, {}
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
        for k in range(floors):
            for i in range(num_locations): col_map[k * num_locations + i + 1] = k
            for i in range(len(beam_connections)): beam_map[num_columns + k * len(beam_connections) + i + 1] = k
    else: # Hybrid (Floor-wise + Position-wise)
        # 1. 위치별 분류 (Corner, Edge, Interior)
        xs = [loc[0] for loc in column_locations]; ys = [loc[1] for loc in column_locations]
        min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
        
        col_local_types = []
        for x, y in column_locations:
            is_x_boundary = (abs(x - min_x) < 1e-3 or abs(x - max_x) < 1e-3)
            is_y_boundary = (abs(y - min_y) < 1e-3 or abs(y - max_y) < 1e-3)
            if is_x_boundary and is_y_boundary: col_local_types.append(2) # Corner
            elif is_x_boundary or is_y_boundary: col_local_types.append(1) # Edge
            else: col_local_types.append(0) # Interior
            
        beam_local_types = []
        for i1, i2 in beam_connections:
            x1, y1 = column_locations[i1]; x2, y2 = column_locations[i2]
            # 보가 평면의 외곽선(Boundary)에 놓여있는지 확인
            is_exterior = (abs(x1 - x2) < 1e-3 and (abs(x1 - min_x) < 1e-3 or abs(x1 - max_x) < 1e-3)) or \
                          (abs(y1 - y2) < 1e-3 and (abs(y1 - min_y) < 1e-3 or abs(y1 - max_y) < 1e-3))
            beam_local_types.append(1 if is_exterior else 0)
            
        num_col_groups_per_floor = 3 # Corner, Edge, Interior
        num_beam_groups_per_floor = 2 # Exterior, Interior
        
        num_col_groups = floors * num_col_groups_per_floor
        num_beam_groups = floors * num_beam_groups_per_floor
        
        for k in range(floors):
            base_col_grp = k * num_col_groups_per_floor
            for i in range(num_locations):
                col_map[k * num_locations + i + 1] = base_col_grp + col_local_types[i]
                
            base_beam_grp = k * num_beam_groups_per_floor
            for i in range(len(beam_connections)):
                beam_map[num_columns + k * len(beam_connections) + i + 1] = base_beam_grp + beam_local_types[i]
                
    return num_col_groups, num_beam_groups, col_map, beam_map

# =================================================================
# ===                  강도 계산 유틸리티                         === 
# =================================================================

def load_pm_data_for_column(h5_file, column_index):
    try:
        original_idx = COLUMN_ID_MAPPING.get(column_index, column_index)
        ref = h5_file['Column_Mdata'][0, original_idx]
        pm_data = h5_file[ref][()]
        return pd.DataFrame(pm_data, columns=['Mnb_z','Pnb_z','Mnb_y','Pnb_y','Alpha_PI_Mnb_z','Alpha_PI_Pnb_z','Alpha_PI_Mnb_y','Alpha_PI_Pnb_y'])
    except Exception: return pd.DataFrame()

def get_pm_capacity_from_df(slope, pm_df, axis='z'):
    if pm_df.empty: return 0.0, 0.0
    design_m, design_p = pm_df[f'Alpha_PI_Mnb_{axis}'].values, pm_df[f'Alpha_PI_Pnb_{axis}'].values
    curve_slopes = abs(design_p) / (abs(design_m) + 1e-9)
    sorted_indices = np.argsort(curve_slopes)[::-1]
    design_m, design_p, curve_slopes = design_m[sorted_indices], design_p[sorted_indices], curve_slopes[sorted_indices]
    if slope >= curve_slopes[0]: return design_p[0], design_m[0]
    if slope <= curve_slopes[-1]: return design_p[-1], design_m[-1]
    for i in range(len(curve_slopes) - 1):
        if curve_slopes[i] >= slope >= curve_slopes[i+1]:
            m1, p1, m2, p2 = design_m[i], design_p[i], design_m[i+1], design_p[i+1]
            if abs(m2 - m1) < 1e-9: return p1, m1
            a1 = (p2 - p1) / (m2 - m1); b1 = p1 - a1 * m1
            Mn = b1 / (slope - a1); Pn = slope * Mn
            return abs(Pn), abs(Mn)
    return 0.0, 0.0

def get_precalculated_strength(element_type, index, col_df, beam_df):
    try:
        if element_type == 'Column':
            row = col_df.iloc[index]
            return {'Pn': row['PI_Pn_max'], 'Vn_y': row['PI_Vn_y'], 'Vn_z': row['PI_Vn_z']}
        elif element_type == 'Beam':
            row = beam_df.iloc[index]
            return {'Pn': float('inf'), 'Vn_y': float('inf'), 'Vn_z': row['PiVn'], 'Mn_y': float('inf'), 'Mn_z': row['PiM']}
    except (IndexError, KeyError): return {'Pn': 0, 'Vn_y': 0, 'Vn_z': 0, 'Mn_y': 0, 'Mn_z': 0}

# =================================================================
# ===                  시각화 유틸리티                            === 
# =================================================================

def plot_Structure(title='Structure Shape', view='3D', ax=None, column_locations=None, beam_connections=None):
    import opsvis as opsv
    if column_locations is None: column_locations = cfg.COLUMN_LOCATIONS
    if beam_connections is None: beam_connections = cfg.BEAM_CONNECTIONS
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
    print("\n[Visualization] Generating architectural load pattern plots...")
    num_floors = len(patterns_by_floor)
    ncols = min(num_floors, 2); nrows = (num_floors - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows), squeeze=False, dpi=150)
    fig.suptitle('Architectural Load Pattern Plan', fontsize=20, fontweight='bold', y=0.98)
    for i, (floor_num, loaded_indices) in enumerate(patterns_by_floor.items()):
        ax = axes[i // ncols, i % ncols]; ax.set_title(f"Floor {floor_num} Load Pattern", fontsize=16, pad=15)
        ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.3)
        for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.5, zorder=1)
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            ax.text(center_x, center_y, f"{length:.1f}m", color='blue', fontsize=9, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5), zorder=4)
        for conn_idx in loaded_indices:
            p1_idx, p2_idx = beam_connections[conn_idx]; p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=5, alpha=0.6, zorder=2)
        xs, ys = [loc[0] for loc in column_locations], [loc[1] for loc in column_locations]
        ax.scatter(xs, ys, c='black', marker='s', s=150, zorder=3)
        ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "load_pattern_visualization.png"))
    plt.close(fig)