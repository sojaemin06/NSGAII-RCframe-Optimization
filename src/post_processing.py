import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D plotting
import opsvis as opsv
from src.config import *
import src.config as cfg
from src.utils import *
from src.structural_analysis import build_model_for_section

# --- SCI Paper Style Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-' 
plt.rcParams['grid.alpha'] = 0.7

def save_results_to_csv(output_folder, all_results, logbook, hof_stats_history, chromosome_structure):
    """최적화 결과를 CSV 파일로 저장합니다. (Data 하위 폴더 사용)"""
    
    data_dir = os.path.join(output_folder, "Data")
    os.makedirs(data_dir, exist_ok=True)

    beam_sections_df_full, column_sections_df_full, _, _ = load_section_data()
    
    # 1. 파레토 최적해 요약
    summary_data_list = []
    for r in all_results:
        ind_obj = r.get('ind_object')
        # 이미 실제 변위비가 넘어오므로 추가 변환 없음
        actual_drift_ratio_val = r.get('max_drift_ratio', 0)
        
        row = {
            'ID': r['ID'],
            'Cost': r['cost'],
            'CO2': r['co2'],
            'Max_Drift_Ratio': actual_drift_ratio_val, 
            'Max_Drift_Ratio_Percent': actual_drift_ratio_val * 100, 
            'Mean_DCR': r.get('mean_strength_ratio', -1),
            'N_types': r.get('N_types', -1)
        }
        
        if ind_obj:
            row['Fit1(CostCO2)'] = ind_obj.fitness.values[0]
            row['Fit2(MaxDrift)'] = ind_obj.fitness.values[1]
        else:
            row['Fit1(CostCO2)'] = -1
            row['Fit2(MaxDrift)'] = -1
            
        summary_data_list.append(row)
    summary_df = pd.DataFrame(summary_data_list)
    summary_df.to_csv(os.path.join(data_dir, "pareto_summary.csv"), index=False)

    # 2. 최적화 과정 로그
    log_df = pd.DataFrame(logbook)
    log_df = log_df.drop(columns=[col for col in log_df.columns if 'sep' in str(col)], errors='ignore')
    log_df.to_csv(os.path.join(data_dir, "optimization_log.csv"), index=False)

    # 3. HOF 수렴도
    if hof_stats_history:
        hof_df = pd.DataFrame(hof_stats_history)
        hof_df.to_csv(os.path.join(data_dir, "hof_convergence.csv"), index=False)

    # 4. 설계 변수 상세
    design_vars_data = []
    detail_cols_col = ['h', 'b', 'fck', 'fy', 'required_rebar', 'MainRebar_size', 'Stirrup_size', 'Stirrup_verticle_size']
    detail_cols_beam = ['h', 'b', 'fck', 'fy', 'N_r', 'dimension', 'stirrup', 'strup_space'] 
    
    for r in all_results:
        ind = r.get('ind_object')
        if ind is None:
            continue
            
        len_col_sec = chromosome_structure['col_sec']
        len_col_rot = chromosome_structure['col_rot']
        
        row_data = {'Solution_ID': r['ID']}
        for i in range(len_col_sec):
            sec_id = ind[i]
            row_data[f'col_grp_{i}_ID'] = sec_id
            row_data[f'col_grp_{i}_Rot'] = ind[len_col_sec + i]
            try:
                sec_details = column_sections_df_full.loc[sec_id, detail_cols_col]
                for col_name, val in sec_details.items():
                    row_data[f'col_grp_{i}_{col_name}'] = val
            except (KeyError, IndexError):
                for col_name in detail_cols_col:
                     row_data[f'col_grp_{i}_{col_name}'] = 'N/A'

        for i in range(chromosome_structure['beam_sec']):
            sec_id = ind[len_col_sec + len_col_rot + i]
            row_data[f'beam_grp_{i}_ID'] = sec_id
            try:
                sec_details = beam_sections_df_full.loc[sec_id, detail_cols_beam]
                for col_name, val in sec_details.items():
                    row_data[f'beam_grp_{i}_{col_name}'] = val
            except (KeyError, IndexError):
                for col_name in detail_cols_beam:
                    row_data[f'beam_grp_{i}_{col_name}'] = 'N/A'
                    
        design_vars_data.append(row_data)
        
    design_vars_df = pd.DataFrame(design_vars_data)
    design_vars_df.to_csv(os.path.join(data_dir, "design_variables.csv"), index=False)

    # 5. 변위 검토 결과
    displacement_data = []
    for r in all_results:
        sol_id = r['ID']
        if drifts_x := r.get('story_drifts_x'):
            for i, val in enumerate(drifts_x): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Seismic_Drift_X', 'Story': i + 1, 'Value': val})
        if drifts_y := r.get('story_drifts_y'):
            for i, val in enumerate(drifts_y): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Seismic_Drift_Y', 'Story': i + 1, 'Value': val})
        if disps_x := r.get('wind_displacements_x'):
            for i, val in enumerate(disps_x): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Wind_Disp_X', 'Story': i + 1, 'Value': val})
        if disps_y := r.get('wind_displacements_y'):
            for i, val in enumerate(disps_y): displacement_data.append({'Solution_ID': sol_id, 'Check_Type': 'Wind_Disp_Y', 'Story': i + 1, 'Value': val})
    if displacement_data:
        displacement_df = pd.DataFrame(displacement_data)
        displacement_df.to_csv(os.path.join(data_dir, "displacement_checks_all.csv"), index=False)

    # 6. 부재별 DCR
    dcr_data = []
    num_columns = len(COLUMN_LOCATIONS) * FLOORS
    for r in all_results:
        sol_id = r['ID']
        ratios = r.get('strength_ratios', [])
        for i, ratio in enumerate(ratios):
            elem_id = i + 1
            elem_type = 'Column' if elem_id <= num_columns else 'Beam'
            dcr_data.append({'Solution_ID': sol_id, 'Element_ID': elem_id, 'ElementType': elem_type, 'DCR': ratio})
    if dcr_data:
        dcr_df = pd.DataFrame(dcr_data)
        dcr_df.to_csv(os.path.join(data_dir, "dcr_by_element.csv"), index=False)

    print(f"All CSV results saved in {data_dir}.")

def plot_results(output_folder, all_results, logbook, hof_stats_history, chromosome_structure, 
                 col_map, beam_map, beam_sections, column_sections, ex_name=""):
    """최적화 결과 그래프를 생성하고 저장합니다 (SCI 논문 스타일, Figures 하위 폴더 사용)."""
    
    if not all_results:
        print("No valid results to plot.")
        return

    # [수정] Figures 폴더 생성
    fig_dir = os.path.join(output_folder, "Figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 최적 구조물 형상
    ind = all_results[0]['ind_object']
    len_col_sec = chromosome_structure['col_sec']
    len_col_rot = chromosome_structure['col_rot']
    col_indices = ind[:len_col_sec]
    col_rotations = ind[len_col_sec : len_col_sec + len_col_rot]
    beam_indices = ind[len_col_sec + len_col_rot :]
    
    try:
        build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections)
        
        # [수정] 예제 2, 3에서는 평면도 출력을 제외 (사용자 요청)
        if "Example_1" in ex_name or ex_name == "":
            # 2D Plan
            fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
            plot_Structure(title='', view='2D_plan', ax=ax_2d)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "optimal_structure_2D_plan.png"))
            plt.close(fig_2d)

        # 3D Element
        fig_3d_elem = plt.figure(figsize=(10, 8))
        ax_3d_elem = fig_3d_elem.add_subplot(111, projection='3d')
        opsv.plot_model(node_labels=0, element_labels=0, az_el=(-60, 30), ax=ax_3d_elem)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "optimal_structure_3D_elements.png"))
        plt.close(fig_3d_elem)
    except Exception as e:
        print(f"Visualization Skipped due to error: {e}")

    # 2. 파레토 전선 (Objective Space)
    # [수정] 해들을 비용순으로 정렬
    sorted_results = sorted(all_results, key=lambda x: x['cost'])
    
    fitness1_vals = [r['ind_object'].fitness.values[0] for r in sorted_results] 
    fitness2_vals = [r['ind_object'].fitness.values[1] for r in sorted_results] # Raw Drift
    drifts = [r.get('max_drift_ratio', 0) for r in sorted_results] # Actual Drift Values
    
    fig_p, ax_p = plt.subplots(figsize=(7, 6))
    
    # [수정] 두 그래프 색상 통일: Drift Ratio에 따라 색상 매핑
    sc_obj = ax_p.scatter(fitness2_vals, fitness1_vals, c=drifts, cmap='viridis', s=80, edgecolors='k', alpha=0.8, zorder=3)
    
    # 축 제목 설정
    ax_p.set_xlabel('Objective 2 (Max. Inter-story Drift Ratio)')
    ax_p.set_ylabel('Objective 1 (Normalized Cost + CO$_2$)')
    plt.grid(True, linestyle='-', alpha=0.7) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "analysis_pareto_objective_space.png"))
    plt.close(fig_p)

    # 3. 수렴 그래프 (Hypervolume)
    if hof_stats_history:
        gens = [s['gen'] for s in hof_stats_history]
        if 'hypervolume' in hof_stats_history[0]:
            hvs = [s['hypervolume'] for s in hof_stats_history]
            fig_conv, ax_conv = plt.subplots(figsize=(7, 5))
            ax_conv.plot(gens, hvs, 'k-o', linewidth=1.5, markersize=5)
            ax_conv.set_xlabel("Generation")
            ax_conv.set_ylabel("Hypervolume Indicator")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "analysis_convergence_hof.png"))
            plt.close(fig_conv)

    # --- [추가 1] 세대별 유효해 수 (Feasible Count History) ---
    if logbook:
        log_df = pd.DataFrame(logbook)
        if 'valid_ratio' in log_df.columns:
            gens = log_df['gen']
            valid_counts = log_df['valid_ratio'] / 100.0 * cfg.POPULATION_SIZE 
            
            fig_feas, ax_feas = plt.subplots(figsize=(7, 5))
            ax_feas.plot(gens, valid_counts, 'g-s', linewidth=1.5, markersize=5)
            ax_feas.set_xlabel("Generation")
            ax_feas.set_ylabel("Number of Feasible Solutions")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "analysis_feasible_count_history.png"))
            plt.close(fig_feas)

    # --- [수정 2] Pareto Solution Space (Cost vs CO2 with Drift Color) ---
    costs = [r['cost'] for r in sorted_results]
    co2s = [r['co2'] for r in sorted_results]
    
    fig_sol_space, ax_sol = plt.subplots(figsize=(10, 8))
    
    # [수정] Objective Space와 동일하게 Drift 값으로 색상 매핑
    sc_sol = ax_sol.scatter(co2s, costs, c=drifts, cmap='viridis', s=100, edgecolors='k', alpha=0.8, zorder=3)
    
    ax_sol.set_xlabel(r'Total CO$_2$ (kgCO$_2$e)')
    ax_sol.set_ylabel('Total Cost (won)')
    ax_sol.grid(True, linestyle='-', alpha=0.7)
    
    # [수정] 컬러바: 실제 Drift Ratio 값 표시
    cbar = plt.colorbar(sc_sol, ax=ax_sol)
    cbar.set_label('Max. Inter-story Drift Ratio', rotation=270, labelpad=20)
    
    # 텍스트 라벨 없음

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "analysis_pareto_solution_space_new.png"))
    plt.close(fig_sol_space)

    # --- [추가 3] 각 최적해의 단면 분포 (Solution Comparison Boxplot) ---
    # Solution Comparison Boxplot (Section ID Range)
    if sorted_results:
        fig_box, ax_box = plt.subplots(figsize=(12, 8))
        
        box_data_col = []
        box_data_beam = []
        labels = []
        
        for i, r in enumerate(sorted_results):
            ind = r['ind_object']
            c_inds = ind[:chromosome_structure['col_sec']]
            b_inds = ind[chromosome_structure['col_sec'] + chromosome_structure['col_rot']:]
            
            box_data_col.append(c_inds)
            box_data_beam.append(b_inds)
            labels.append(f'Sol #{i+1}')
            
        x_pos = np.arange(len(labels))
        width = 0.35
        
        bp_col = ax_box.boxplot(box_data_col, positions=x_pos - width/2, widths=width, patch_artist=True, 
                                boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='orange'))
        
        bp_beam = ax_box.boxplot(box_data_beam, positions=x_pos + width/2, widths=width, patch_artist=True,
                                 boxprops=dict(facecolor='lightgreen', color='black'), medianprops=dict(color='orange'))
        
        ax_box.legend([bp_col["boxes"][0], bp_beam["boxes"][0]], ['Column Sections', 'Beam Sections'], loc='upper left')
        
        ax_box.set_xticks(x_pos)
        ax_box.set_xticklabels(labels, rotation=45, ha='right')
        ax_box.set_ylabel('Section Index ID')
        ax_box.set_xlabel('Pareto Solution ID')
        ax_box.grid(True, axis='y', linestyle='-', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "solution_comparison_boxplot.png"))
        plt.close(fig_box)

    print(f"All plots saved in {fig_dir} with SCI style.")