
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

def load_section_data(beam_path="beam_sections_simple02.csv", col_path="column_sections_simple02.csv"):
    """CSV 파일에서 단면 정보를 로드합니다. (경로 지정 가능)"""
    beam_sections_df = pd.read_csv(beam_path)
    column_sections_df = pd.read_csv(col_path)
    
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
    
    # Extract arrays
    design_m = pm_df[moment_col].values
    design_p = pm_df[axial_col].values
    
    # Calculate slopes (P/M)
    # Use small epsilon for M to avoid div by zero, resulting in Inf slope for pure compression
    curve_slopes = abs(design_p) / (abs(design_m) + 1e-9)
    
    # Sort by slope Descending (Infinity -> 0)
    # This ensures the loop condition (slopes[i] >= slope >= slopes[i+1]) works correctly
    sorted_indices = np.argsort(curve_slopes)[::-1]
    design_m = design_m[sorted_indices]
    design_p = design_p[sorted_indices]
    curve_slopes = curve_slopes[sorted_indices]
    
    # 1. Check if slope is larger than max slope (Pure Compression region)
    if slope >= curve_slopes[0]:
        return design_p[0], design_m[0]
        
    # 2. Check if slope is smaller than min slope (Pure Bending region)
    if slope <= curve_slopes[-1]:
        return design_p[-1], design_m[-1]

    # 3. Interpolate
    for i in range(len(curve_slopes) - 1):
        if curve_slopes[i] >= slope >= curve_slopes[i+1]:
            m1, p1, m2, p2 = design_m[i], design_p[i], design_m[i+1], design_p[i+1]
            
            # Avoid division by zero if points are identical
            if abs(m2 - m1) < 1e-9: 
                return p1, m1
                
            # Linear Interpolation on P-M diagram
            # Line eq: P - p1 = a * (M - m1)  => P = a*M + b
            a1 = (p2 - p1) / (m2 - m1)
            b1 = p1 - a1 * m1
            
            # Intersection with P = slope * M
            # slope * M = a1 * M + b1  => M * (slope - a1) = b1
            if abs(a1 - slope) < 1e-9: # Parallel lines (should unlikely happen if logic is correct)
                return p1, m1
                
            Mn = b1 / (slope - a1)
            Pn = slope * Mn
            
            # Additional safety: ensure Pn, Mn are positive/consistent with demand sign if needed
            # But here capacity is usually positive magnitude.
            return abs(Pn), abs(Mn)
            
    # Fallback (should be covered by edge checks)
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
    print("\n[Visualization] Generating architectural load pattern plots...")
    
    num_floors = len(patterns_by_floor)
    ncols = min(num_floors, 2) 
    nrows = (num_floors - 1) // ncols + 1
    
    # Figure Size & Resolution increased for clarity
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows), squeeze=False, dpi=150)
    fig.suptitle('Architectural Load Pattern Plan', fontsize=20, fontweight='bold', y=0.98)

    # Pre-calculate beam lengths for annotation
    beam_lengths = {}
    for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
        p1 = column_locations[p1_idx]
        p2 = column_locations[p2_idx]
        length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        beam_lengths[conn_idx] = length

    for i, (floor_num, loaded_indices) in enumerate(patterns_by_floor.items()):
        ax = axes[i // ncols, i % ncols]
        ax.set_title(f"Floor {floor_num} Load Pattern", fontsize=16, pad=15)
        
        # 1. Draw Grid (Background)
        ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.3)
        
        # 2. Draw All Beams (Base) - Black lines
        for conn_idx, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            
            # Draw Beam Line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.5, zorder=1)
            
            # Annotate Span Length (only for non-loaded for clarity, or smaller text)
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            
            # Calculate offset for text to avoid overlapping the line too much
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Display length text
            ax.text(center_x, center_y, f"{beam_lengths[conn_idx]:.1f}m", 
                    color='blue', fontsize=9, ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5), zorder=4)

        # 3. Highlight Loaded Beams - Thick Red lines
        for conn_idx in loaded_indices:
            p1_idx, p2_idx = beam_connections[conn_idx]
            p1, p2 = column_locations[p1_idx], column_locations[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=5, alpha=0.6, zorder=2, 
                    label='Loaded Beam' if 'Loaded Beam' not in [l.get_label() for l in ax.get_lines()] else '')

        # 4. Draw Columns - Square Markers
        xs = [loc[0] for loc in column_locations]
        ys = [loc[1] for loc in column_locations]
        ax.scatter(xs, ys, c='black', marker='s', s=150, zorder=3, label='Column')

        # Axis settings
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # Set ticks to match column grid roughly
        unique_x = sorted(list(set(xs)))
        unique_y = sorted(list(set(ys)))
        ax.set_xticks(unique_x)
        ax.set_yticks(unique_y)
        
        if i == 0: 
            # Custom legend
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='black', lw=2),
                            Line2D([0], [0], color='red', lw=5, alpha=0.6),
                            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10)]
            ax.legend(custom_lines, ['Structure Beam', 'Loaded Beam', 'Column'], loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_folder, "load_pattern_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Architectural load pattern visualization saved to '{save_path}'")

def generate_load_patterns(floors, num_beams_per_floor, column_locations=None, beam_connections=None):
    """
    Generates load patterns based on slab bays (Checkerboard style).
    Identifies rectangular bays enclosed by beams and applies loads to beams supporting selected bays.
    """
    patterns = {}
    if column_locations is None or beam_connections is None:
        # Fallback to random if geometry not provided
        all_beam_indices = list(range(num_beams_per_floor))
        num_to_load = max(1, num_beams_per_floor // 2)
        for k in range(1, floors + 1):
            selected = np.random.choice(all_beam_indices, num_to_load, replace=False)
            patterns[k] = selected.tolist()
        return patterns

    # 1. Identify Rectangular Bays
    # A simple approach: iterate all unique X and Y intervals to find potential grid cells,
    # then check if columns exist at 4 corners and beams exist on 4 edges.
    
    # Map beam connection to index for quick lookup: (min, max) -> index
    beam_lookup = {}
    for idx, (p1, p2) in enumerate(beam_connections):
        beam_lookup[tuple(sorted((p1, p2)))] = idx

    bays = [] # List of {'beams': [idx1, idx2, idx3, idx4], 'centroid': (cx, cy)}
    
    col_indices = {loc: i for i, loc in enumerate(column_locations)}
    xs = sorted(list(set(loc[0] for loc in column_locations)))
    ys = sorted(list(set(loc[1] for loc in column_locations)))
    
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1, x2 = xs[i], xs[i+1]
            y1, y2 = ys[j], ys[j+1]
            
            # Check if 4 corners exist as columns
            p_bl = (x1, y1); p_br = (x2, y1)
            p_tl = (x1, y2); p_tr = (x2, y2)
            
            if not all(p in col_indices for p in [p_bl, p_br, p_tl, p_tr]):
                continue
                
            idx_bl, idx_br = col_indices[p_bl], col_indices[p_br]
            idx_tl, idx_tr = col_indices[p_tl], col_indices[p_tr]
            
            # Check if 4 enclosing beams exist
            b_bott = tuple(sorted((idx_bl, idx_br)))
            b_top  = tuple(sorted((idx_tl, idx_tr)))
            b_left = tuple(sorted((idx_bl, idx_tl)))
            b_right= tuple(sorted((idx_br, idx_tr)))
            
            if all(b in beam_lookup for b in [b_bott, b_top, b_left, b_right]):
                # Valid Bay Found
                bay_beams = [beam_lookup[b] for b in [b_bott, b_top, b_left, b_right]]
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                bays.append({'beams': bay_beams, 'centroid': centroid})

    # 2. Generate Checkerboard Patterns
    # Sort bays by X then Y to assign checkerboard indices
    # However, bays might be irregular. We can use coordinate based parity.
    # (i + j) % 2 == 0 vs 1
    
    # Assign grid index (i, j) to each bay based on centroid rank
    unique_cx = sorted(list(set(b['centroid'][0] for b in bays)))
    unique_cy = sorted(list(set(b['centroid'][1] for b in bays)))
    
    for k in range(1, floors + 1):
        selected_beam_indices = set()
        
        # Define pattern type for this floor
        # Pattern A: (i+j) even, Pattern B: (i+j) odd
        target_parity = k % 2 
        
        for bay in bays:
            cx, cy = bay['centroid']
            i_idx = unique_cx.index(cx)
            j_idx = unique_cy.index(cy)
            
            if (i_idx + j_idx) % 2 == target_parity:
                # This bay is loaded -> All its beams take load
                for b_idx in bay['beams']:
                    selected_beam_indices.add(b_idx)
        
        # If no bays found (e.g. single frame), fallback to all beams
        if not bays:
             selected_beam_indices = set(range(num_beams_per_floor))
             
        patterns[k] = list(selected_beam_indices)
        
    return patterns
