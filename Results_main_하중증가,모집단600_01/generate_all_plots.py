import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os
import math
from scipy.spatial import ConvexHull

# --- ✅ 0. 스크립트 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ✅ 1. Matplotlib 전역 글꼴 설정: Times New Roman ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# --- ✅ 2. 그룹핑 로직 재현을 위한 전역 변수 및 함수 ---
GROUPING_STRATEGY = "Hybrid"
floors = 4
H = 4.0
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

def get_path(filename):
    """스크립트 위치를 기준으로 파일의 절대 경로를 반환합니다."""
    return os.path.join(SCRIPT_DIR, filename)

def get_grouping_maps():
    """원본 스크립트의 그룹핑 로직을 재현하여 col_map과 beam_map을 반환합니다."""
    num_locations = len(column_locations)
    num_columns = num_locations * floors
    num_beams = len(beam_connections) * floors
    col_map, beam_map = {}, {}
    num_col_groups, num_beam_groups = 0, 0

    if GROUPING_STRATEGY == "Hybrid":
        node_connectivity = {i: 0 for i in range(len(column_locations))}
        for p1_idx, p2_idx in beam_connections:
            node_connectivity[p1_idx] += 1
            node_connectivity[p2_idx] += 1

        memo_col_type = {}
        def is_point_inside_hull(point, hull):
            return np.all(np.add(np.dot(hull.equations[:, :-1], point), hull.equations[:, -1]) < 1e-9)

        def get_col_loc_type(loc_idx):
            if loc_idx in memo_col_type: return memo_col_type[loc_idx]
            connections = node_connectivity.get(loc_idx, 0)
            loc_type = -1 
            if connections <= 2: loc_type = 0
            elif connections == 3: loc_type = 1
            else:
                neighbor_indices = {p[1] if p[0] == loc_idx else p[0] for p in beam_connections if loc_idx in p}
                if len(neighbor_indices) < 3: loc_type = 2
                else:
                    neighbor_points = np.array([column_locations[i] for i in neighbor_indices])
                    current_point = np.array(column_locations[loc_idx])
                    try:
                        neighbors_hull = ConvexHull(neighbor_points)
                        loc_type = 2 if is_point_inside_hull(current_point, neighbors_hull) else 0
                    except Exception: loc_type = 2
            memo_col_type[loc_idx] = loc_type
            return loc_type

        points = np.array(column_locations)
        hull = ConvexHull(points)
        equations = hull.equations
        perimeter_beam_indices = set()
        for i, (p1_idx, p2_idx) in enumerate(beam_connections):
            p1, p2 = points[p1_idx], points[p2_idx]
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
    
    return col_map, beam_map, num_col_groups, num_beam_groups, num_columns, num_beams

def create_displacement_plots():
    try:
        df = pd.read_csv(get_path('displacement_checks_all.csv'))
    except FileNotFoundError:
        print(f"Error: 'displacement_checks_all.csv' not found in {SCRIPT_DIR}. Skipping displacement plots.")
        return

    solutions = df['Solution_ID'].unique()
    num_solutions = len(solutions)
    colormap = plt.cm.viridis_r

    plot_details = {
        'Seismic_Drift_X': {'xlabel': 'Drift Ratio','limit': 0.015,'filename': 'displacement_check_seismic_drift_x_inverted.png'},
        'Seismic_Drift_Y': {'xlabel': 'Drift Ratio','limit': 0.015,'filename': 'displacement_check_seismic_drift_y_inverted.png'},
        'Wind_Disp_X': {'xlabel': 'Displacement (m)','limit': 0.040,'filename': 'displacement_check_wind_disp_x_inverted.png'},
        'Wind_Disp_Y': {'xlabel': 'Displacement (m)','limit': 0.040,'filename': 'displacement_check_wind_disp_y_inverted.png'}
    }

    for check_type, details in plot_details.items():
        plt.figure(figsize=(7, 6))
        df_check = df[df['Check_Type'] == check_type]
        if df_check.empty: continue
        for i, solution in enumerate(solutions):
            df_sol = df_check[df_check['Solution_ID'] == solution]
            if not df_sol.empty:
                sol_num = int(solution.split('#')[-1])
                color = colormap((sol_num - 1) / (num_solutions - 1)) if num_solutions > 1 else colormap(0.5)
                plt.plot(df_sol['Value'], df_sol['Story'], marker='o', color=color)
        plt.axvline(x=details['limit'], color='r', linestyle='--', label='Limit')
        plt.xlabel(details['xlabel'], fontsize=16)
        plt.ylabel('Story', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=14)
        plt.savefig(get_path(details['filename']))
        plt.close()
    print("Generated 4 displacement plots.")

def create_pareto_plots():
    try:
        df = pd.read_csv(get_path('pareto_summary.csv'), thousands=',')
    except FileNotFoundError:
        print(f"Error: 'pareto_summary.csv' not found in {SCRIPT_DIR}. Skipping pareto plots.")
        return

    fit1_col = 'Fit1(NormCost+CO2)'
    fit2_col = 'Fit2(Mean_DCR)'
    if fit1_col in df.columns and fit2_col in df.columns:
        x, y, colors = df[fit2_col], df[fit1_col], np.arange(len(df))
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(x, y, c=colors, cmap='viridis_r', s=80, alpha=0.8, edgecolors='k')
        ax.set_xlabel('Structural Conservatism ($f_2$, Mean DCR)', fontsize=16)
        ax.set_ylabel('Economic & Environmental Demand ($f_1$)', fontsize=16)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(get_path('analysis_pareto_objective_space_inverted.png'))
        plt.close(fig)

    cost_col, co2_col = 'Cost', 'CO2'
    if cost_col in df.columns and co2_col in df.columns and fit2_col in df.columns:
        real_costs, real_co2s, fitness2_vals = df[cost_col], df[co2_col], df[fit2_col]
        fig_p_sol, ax_p_sol = plt.subplots(figsize=(9, 7))
        sc = ax_p_sol.scatter(real_co2s, real_costs, c=fitness2_vals, cmap='viridis', s=80, edgecolors='k', alpha=0.8)
        cbar = fig_p_sol.colorbar(sc, ax=ax_p_sol)
        cbar.set_label('Mean DCR', size=14)
        ax_p_sol.set_xlabel('Total CO2 (kgCO2e)', fontsize=16)
        ax_p_sol.set_ylabel('Total Cost (won)', fontsize=16)
        formatter = FuncFormatter(lambda x, pos: f'{int(x):,}')
        ax_p_sol.get_xaxis().set_major_formatter(formatter)
        ax_p_sol.get_yaxis().set_major_formatter(formatter)
        ax_p_sol.grid(True)
        plt.tight_layout()
        plt.savefig(get_path('analysis_pareto_solution_space_new.png'))
        plt.close(fig_p_sol)
    print("Generated 2 pareto plots.")

def create_representative_displacement_plots():
    try:
        df = pd.read_csv(get_path('displacement_checks_all.csv'))
    except FileNotFoundError:
        print(f"Error: 'displacement_checks_all.csv' not found. Skipping representative displacement plots.")
        return

    all_solutions, colormap = df['Solution_ID'].unique(), plt.cm.viridis_r
    num_solutions = len(all_solutions)
    representative_sols = ['Sol #1', 'Sol #12', 'Sol #23']
    plot_details = {
        'Seismic_Drift_X': {'xlabel': 'Drift Ratio','limit': 0.015,'filename': 'representative_displacement_check_seismic_drift_x.png'},
        'Seismic_Drift_Y': {'xlabel': 'Drift Ratio','limit': 0.015,'filename': 'representative_displacement_check_seismic_drift_y.png'},
        'Wind_Disp_X': {'xlabel': 'Displacement (m)','limit': 0.040,'filename': 'representative_displacement_check_wind_disp_x.png'},
        'Wind_Disp_Y': {'xlabel': 'Displacement (m)','limit': 0.040,'filename': 'representative_displacement_check_wind_disp_y.png'}
    }
    for check_type, details in plot_details.items():
        plt.figure(figsize=(10, 8))
        df_check = df[df['Check_Type'] == check_type]
        if df_check.empty: continue
        for sol_id in representative_sols:
            df_sol = df_check[df_check['Solution_ID'] == sol_id]
            if not df_sol.empty:
                sol_num = int(sol_id.split('#')[-1])
                color = colormap((sol_num - 1) / (num_solutions - 1)) if num_solutions > 1 else colormap(0.5)
                plt.plot(df_sol['Value'], df_sol['Story'], marker='o', color=color, label=sol_id)
        plt.axvline(x=details['limit'], color='r', linestyle='--', label='Limit')
        plt.xlabel(details['xlabel'], fontsize=16)
        plt.ylabel('Story', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=14)
        plt.savefig(get_path(details['filename']))
        plt.close()
    print("Generated 4 representative displacement plots.")

def create_convergence_plots():
    try:
        df_hof = pd.read_csv(get_path('hof_convergence.csv'))
        fig_conv, ax_conv = plt.subplots(figsize=(8, 7))
        line1, = ax_conv.plot(df_hof['gen'], df_hof['best_obj1'], color='tab:blue', marker='o', linestyle='-', label="Best Fitness1")
        ax_conv.set_xlabel("Generation", fontsize=16)
        ax_conv.set_ylabel("Fitness1 (Normalized Cost+CO2)", color='tab:blue', fontsize=16)
        ax_conv.tick_params(axis='y', labelcolor='tab:blue')
        ax_conv.grid(True)
        ax_conv_twin = ax_conv.twinx()
        line2, = ax_conv_twin.plot(df_hof['gen'], df_hof['best_obj2'], color='tab:red', marker='s', linestyle='-', label="Best Fitness2")
        ax_conv_twin.set_ylabel("Fitness2 (Mean DCR)", color='tab:red', fontsize=16)
        ax_conv_twin.tick_params(axis='y', labelcolor='tab:red')
        ax_conv.legend(handles=[line1, line2], loc='upper right', fontsize=14)
        plt.tight_layout()
        plt.savefig(get_path("analysis_convergence_hof.png"))
        plt.close()
        print("Generated HOF convergence plot.")
    except FileNotFoundError:
        print(f"Error: 'hof_convergence.csv' not found. Skipping HOF plot.")

    try:
        df_log = pd.read_csv(get_path('optimization_log.csv'))
        fig_valid, ax_valid = plt.subplots(figsize=(8, 7))
        ax_valid.plot(df_log["gen"], df_log["valid_ratio"], marker='o', linestyle='-', color='g')
        ax_valid.set_xlabel("Generation", fontsize=16)
        ax_valid.set_ylabel("Feasible Solutions (%)", fontsize=16)
        ax_valid.set_ylim(0, 105)
        ax_valid.grid(True)
        plt.tight_layout()
        plt.savefig(get_path("analysis_feasible_ratio.png"))
        plt.close()
        print("Generated feasible solution ratio plot.")
    except FileNotFoundError:
        print(f"Error: 'optimization_log.csv' not found. Skipping feasible ratio plot.")

def create_performance_comparison_plots():
    try:
        df_summary = pd.read_csv(get_path('pareto_summary.csv'), thousands=',')
        df_dcr = pd.read_csv(get_path('dcr_by_element.csv'))
        df_vars = pd.read_csv(get_path('design_variables.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Skipping performance plots.")
        return

    max_dcr_per_sol = df_dcr.groupby('Solution_ID')['DCR'].max().rename('max_strength_ratio')
    df_summary = pd.merge(df_summary, max_dcr_per_sol, left_on='ID', right_on='Solution_ID')

    # Radar Chart
    radar_metrics = ['Cost', 'CO2', 'Mean_DCR', 'max_strength_ratio']
    radar_labels = ['Cost Demand', 'CO2 Demand', 'Average DCR', 'Max DCR']
    radar_data = df_summary[radar_metrics]
    radar_data_normalized = (radar_data.max() - radar_data) / (radar_data.max() - radar_data.min())
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist() + [0]
    fig_radar, ax_radar = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_summary)))
    for i, row in radar_data_normalized.iterrows():
        values = row.values.flatten().tolist() + [row.values[0]]
        ax_radar.plot(angles, values, 'o-', linewidth=2, color=colors[i])
        ax_radar.fill(angles, values, color=colors[i], alpha=0.15)
    ax_radar.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(get_path("solution_comparison_radar.png"))
    plt.close()
    print("Generated radar chart for all solutions.")

    # Box Plot
    fig_box, ax_box = plt.subplots(figsize=(12, 9))
    col_sec_cols = [c for c in df_vars.columns if 'col_sec' in c]
    beam_sec_cols = [c for c in df_vars.columns if 'beam_sec' in c]
    bp_col = ax_box.boxplot(df_vars[col_sec_cols].T, positions=np.array(range(len(df_vars))) * 2.0 - 0.4, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp_beam = ax_box.boxplot(df_vars[beam_sec_cols].T, positions=np.array(range(len(df_vars))) * 2.0 + 0.4, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax_box.legend([bp_col["boxes"][0], bp_beam["boxes"][0]], ['Column Sections', 'Beam Sections'], fontsize=14)
    ax_box.set_xticks(np.arange(0, len(df_vars) * 2, 2))
    ax_box.set_xticklabels(df_vars['Solution_ID'], rotation=45, ha='right')
    ax_box.set_xlabel('Pareto Solution ID', fontsize=16)
    ax_box.set_ylabel('Section Index ID', fontsize=16)
    ax_box.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(get_path("solution_comparison_boxplot.png"))
    plt.close()
    print("Generated box plot for all solutions.")

def create_dcr_analysis_plots(col_map, beam_map, num_col_groups, num_beam_groups, num_columns, num_beams):
    try:
        df_dcr = pd.read_csv(get_path('dcr_by_element.csv'))
        df_summary = pd.read_csv(get_path('pareto_summary.csv'), thousands=',')
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Skipping DCR analysis plots.")
        return

    for i, result in df_summary.iterrows():
        sol_id = result['ID']
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        # ... (plotting logic as before)
        plt.savefig(get_path(f"stress_ratios_{sol_id.replace('#','')}.png"))
        plt.close()
    print("Generated DCR distribution plots for all solutions.")

def create_representative_performance_plots():
    try:
        df_summary = pd.read_csv(get_path('pareto_summary.csv'), thousands=',')
        df_dcr = pd.read_csv(get_path('dcr_by_element.csv'))
        df_vars = pd.read_csv(get_path('design_variables.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Skipping representative performance plots.")
        return

    representative_sols_ids = ['Sol #1', 'Sol #12', 'Sol #23']
    df_summary_rep = df_summary[df_summary['ID'].isin(representative_sols_ids)]
    df_vars_rep = df_vars[df_vars['Solution_ID'].isin(representative_sols_ids)]

    # Radar Chart for Reps
    max_dcr_per_sol = df_dcr.groupby('Solution_ID')['DCR'].max().rename('max_strength_ratio')
    df_summary_rep = pd.merge(df_summary_rep, max_dcr_per_sol, left_on='ID', right_on='Solution_ID')
    radar_metrics = ['Cost', 'CO2', 'Mean_DCR', 'max_strength_ratio']
    radar_labels = ['Cost Demand', 'CO2 Demand', 'Average DCR', 'Max DCR']
    radar_data = df_summary_rep[radar_metrics]
    radar_data_normalized = (radar_data.max() - radar_data) / (radar_data.max() - radar_data.min())
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist() + [0]
    fig_radar, ax_radar = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    colors_rep = plt.cm.jet(np.linspace(0, 1, len(df_summary_rep)))
    for i, (idx, row) in enumerate(df_summary_rep.iterrows()):
        values = radar_data_normalized.loc[idx].values.flatten().tolist() + [radar_data_normalized.loc[idx].values[0]]
        ax_radar.plot(angles, values, 'o-', linewidth=2, color=colors_rep[i], label=row['ID'])
        ax_radar.fill(angles, values, color=colors_rep[i], alpha=0.15)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14)
    ax_radar.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(get_path("representative_solution_comparison_radar.png"))
    plt.close()
    print("Generated radar chart for representative solutions.")

    # Box Plot for Reps
    fig_box, ax_box = plt.subplots(figsize=(12, 9))
    col_sec_cols = [c for c in df_vars_rep.columns if 'col_sec' in c]
    beam_sec_cols = [c for c in df_vars_rep.columns if 'beam_sec' in c]
    bp_col = ax_box.boxplot(df_vars_rep[col_sec_cols].T, positions=np.array(range(len(df_vars_rep))) * 2.0 - 0.4, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp_beam = ax_box.boxplot(df_vars_rep[beam_sec_cols].T, positions=np.array(range(len(df_vars_rep))) * 2.0 + 0.4, sym='', widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax_box.legend([bp_col["boxes"][0], bp_beam["boxes"][0]], ['Column Sections', 'Beam Sections'], fontsize=14)
    ax_box.set_xticks(np.arange(0, len(df_vars_rep) * 2, 2))
    ax_box.set_xticklabels(df_vars_rep['Solution_ID'], rotation=45, ha='right')
    ax_box.set_xlabel('Representative Solution ID', fontsize=16)
    ax_box.set_ylabel('Section Index ID', fontsize=16)
    ax_box.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(get_path("representative_solution_comparison_boxplot.png"))
    plt.close()
    print("Generated box plot for representative solutions.")

def create_representative_dcr_analysis_plots(col_map, beam_map, num_col_groups, num_beam_groups, num_columns, num_beams):
    """
    Generates detailed DCR analysis plots for each representative solution.
    """
    try:
        df_dcr = pd.read_csv(get_path('dcr_by_element.csv'))
        df_summary = pd.read_csv(get_path('pareto_summary.csv'), thousands=',')
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found in {SCRIPT_DIR}. Skipping representative DCR analysis plots.")
        return

    representative_sols_ids = ['Sol #1', 'Sol #12', 'Sol #23']
    rep_results = df_summary[df_summary['ID'].isin(representative_sols_ids)].to_dict('records')

    if not rep_results:
        print("Representative solutions not found in summary. Skipping plots.")
        return

    for result in rep_results:
        sol_id = result['ID']
        rep_id_str = sol_id.replace('#', '').replace(' ', '')

        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
        
        ax_dist, ax_bar_col, ax_bar_beam = axs[0], axs[1], axs[2]
        
        ratios_df = df_dcr[df_dcr['Solution_ID'] == sol_id]
        if ratios_df.empty:
            print(f"No DCR data for {sol_id}. Skipping its plot.")
            plt.close(fig)
            continue

        ratios = ratios_df['DCR'].tolist()
        mean_ratio = result['Mean_DCR']

        # 1. Sorted DCR Distribution
        sorted_ratios = sorted(ratios, reverse=True)
        ax_dist.plot(sorted_ratios, marker='.', linestyle='-', color='gray')
        ax_dist.axhline(y=1.0, color='r', linestyle='--', label='Allowable Ratio (1.0)')
        ax_dist.axhline(y=mean_ratio, color='b', linestyle='--', label=f'Avg DCR: {mean_ratio:.3f}')
        if sorted_ratios:
            ax_dist.plot(0, sorted_ratios[0], 'o', color='red', markersize=8, label=f'Max DCR: {sorted_ratios[0]:.3f}')
        ax_dist.set_xlabel('Members (Sorted)', fontsize=16)
        ax_dist.set_ylabel('DCR', fontsize=16)
        ax_dist.set_ylim(bottom=0, top=max(1.2, sorted_ratios[0] * 1.1 if sorted_ratios else 1.2))
        ax_dist.legend(fontsize=14)
        ax_dist.grid(axis='y', linestyle=':', alpha=0.7)

        # 2 & 3. Max DCR per Group
        max_dcr_by_col_group = {i: 0.0 for i in range(num_col_groups)}
        max_dcr_by_beam_group = {i: 0.0 for i in range(num_beam_groups)}

        elem_dcr_df = ratios_df.set_index('Element_ID')
        for c_idx in range(1, num_columns + 1):
            if c_idx in elem_dcr_df.index and c_idx in col_map:
                group_id = col_map[c_idx]
                max_dcr_by_col_group[group_id] = max(max_dcr_by_col_group[group_id], elem_dcr_df.loc[c_idx, 'DCR'])
        
        for b_idx in range(1, num_beams + 1):
            elem_id = num_columns + b_idx
            if elem_id in elem_dcr_df.index and elem_id in beam_map:
                group_id = beam_map[elem_id]
                max_dcr_by_beam_group[group_id] = max(max_dcr_by_beam_group[group_id], elem_dcr_df.loc[elem_id, 'DCR'])

        if max_dcr_by_col_group:
            col_groups = sorted(max_dcr_by_col_group.keys())
            col_values = [max_dcr_by_col_group[g] for g in col_groups]
            ax_bar_col.bar(col_groups, col_values, color='royalblue', edgecolor='black')
            ax_bar_col.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
            ax_bar_col.set_xlabel('Column Group ID', fontsize=16)
            ax_bar_col.set_ylabel('Max DCR', fontsize=16)
            ax_bar_col.set_xticks(col_groups)
            ax_bar_col.set_ylim(bottom=0, top=1.2)
            ax_bar_col.grid(axis='y', linestyle=':', alpha=0.7)

        if max_dcr_by_beam_group:
            beam_groups = sorted(max_dcr_by_beam_group.keys())
            beam_values = [max_dcr_by_beam_group[g] for g in beam_groups]
            ax_bar_beam.bar(beam_groups, beam_values, color='seagreen', edgecolor='black')
            ax_bar_beam.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
            ax_bar_beam.set_xlabel('Beam Group ID', fontsize=16)
            ax_bar_beam.set_ylabel('Max DCR', fontsize=16)
            ax_bar_beam.set_xticks(beam_groups)
            ax_bar_beam.set_ylim(bottom=0, top=1.2)
            ax_bar_beam.grid(axis='y', linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.savefig(get_path(f"representative_stress_ratios_{rep_id_str}.png"))
        plt.close(fig)
        print(f"Generated detailed DCR plot for {sol_id}.")

if __name__ == '__main__':
    col_map, beam_map, num_col_groups, num_beam_groups, num_columns, num_beams = get_grouping_maps()
    print("--- Generating All Plots ---")
    # create_displacement_plots()
    # create_representative_displacement_plots()
    # create_pareto_plots()
    # create_convergence_plots()
    # create_performance_comparison_plots()
    # create_dcr_analysis_plots(col_map, beam_map, num_col_groups, num_beam_groups, num_columns, num_beams)
    # create_representative_performance_plots()
    create_representative_dcr_analysis_plots(col_map, beam_map, num_col_groups, num_beam_groups, num_columns, num_beams)
    print("\nAll plot generation tasks complete.")