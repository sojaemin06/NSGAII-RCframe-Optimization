
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opsvis as opsv
from src.config import *
from src.utils import *
from src.structural_analysis import build_model_for_section

def save_results_to_csv(output_folder, all_results, logbook, hof_stats_history, chromosome_structure):
    """최적화 결과를 CSV 파일로 저장합니다."""
    
    # 1. 파레토 최적해 요약
    summary_data_list = []
    for r in all_results:
        ind_obj = r['ind_object']
        summary_data_list.append({
            'ID': r['ID'],
            'Cost': r['cost'],
            'CO2': r['co2'],
            'Mean_DCR': r['mean_strength_ratio'],
            'Fit1(NormCost+CO2)': ind_obj.fitness.values[0],
            'Fit2(Mean_DCR)': ind_obj.fitness.values[1]
        })
    summary_df = pd.DataFrame(summary_data_list)
    summary_df.to_csv(os.path.join(output_folder, "pareto_summary.csv"), index=False)

    # 2. 최적화 과정 로그
    log_df = pd.DataFrame(logbook)
    log_df = log_df.drop(columns=[col for col in log_df.columns if 'sep' in str(col)], errors='ignore')
    log_df.to_csv(os.path.join(output_folder, "optimization_log.csv"), index=False)

    # 3. HOF 수렴도
    if hof_stats_history:
        hof_df = pd.DataFrame(hof_stats_history)
        hof_df.to_csv(os.path.join(output_folder, "hof_convergence.csv"), index=False)

    # 4. 최적해별 설계 변수
    design_vars_data = []
    for r in all_results:
        ind = r['individual']
        len_col_sec = chromosome_structure['col_sec']
        len_col_rot = chromosome_structure['col_rot']
        
        row_data = {'Solution_ID': r['ID']}
        for i in range(len_col_sec):
            row_data[f'col_sec_grp_{i}'] = ind[i]
        for i in range(len_col_rot):
            row_data[f'col_rot_grp_{i}'] = ind[len_col_sec + i]
        for i in range(chromosome_structure['beam_sec']):
            row_data[f'beam_sec_grp_{i}'] = ind[len_col_sec + len_col_rot + i]
        design_vars_data.append(row_data)
    design_vars_df = pd.DataFrame(design_vars_data)
    design_vars_df.to_csv(os.path.join(output_folder, "design_variables.csv"), index=False)

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
        displacement_df.to_csv(os.path.join(output_folder, "displacement_checks_all.csv"), index=False)

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
        dcr_df.to_csv(os.path.join(output_folder, "dcr_by_element.csv"), index=False)

    print("All CSV results saved.")

def plot_results(output_folder, all_results, logbook, hof_stats_history, chromosome_structure, 
                 col_map, beam_map, beam_sections, column_sections):
    """최적화 결과 그래프를 생성하고 저장합니다."""
    
    # 1. 최적 구조물 형상 (첫 번째 해 기준)
    ind = all_results[0]['individual']
    len_col_sec = chromosome_structure['col_sec']
    len_col_rot = chromosome_structure['col_rot']
    col_indices = ind[:len_col_sec]
    col_rotations = ind[len_col_sec : len_col_sec + len_col_rot]
    beam_indices = ind[len_col_sec + len_col_rot :]
    
    # 모델 빌드 (시각화를 위해)
    build_model_for_section(col_indices, col_rotations, beam_indices, col_map, beam_map, beam_sections, column_sections)
    
    # 2D Plan
    fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
    plot_Structure(title='2D Plan View of Optimal Design', view='2D_plan', ax=ax_2d)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "optimal_structure_2D_plan.png"))
    plt.close(fig_2d)

    # 3D Node
    fig_3d_node = plt.figure(figsize=(10, 8))
    ax_3d_node = fig_3d_node.add_subplot(111, projection='3d')
    opsv.plot_model(node_labels=1, element_labels=0, az_el=(-60, 30), ax=ax_3d_node)
    ax_3d_node.set_title('Node Numbering')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "optimal_structure_3D_nodes.png"))
    plt.close(fig_3d_node)

    # 3D Element
    fig_3d_elem = plt.figure(figsize=(10, 8))
    ax_3d_elem = fig_3d_elem.add_subplot(111, projection='3d')
    opsv.plot_model(node_labels=0, element_labels=1, az_el=(-60, 30), ax=ax_3d_elem)
    ax_3d_elem.set_title('Element Numbering')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "optimal_structure_3D_elements.png"))
    plt.close(fig_3d_elem)

    # 2. 파레토 전선
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    fitness1_vals = [r['ind_object'].fitness.values[0] for r in all_results]
    fitness2_vals = [r['ind_object'].fitness.values[1] for r in all_results]
    
    fig_p, ax_p = plt.subplots(figsize=(8, 7))
    ax_p.scatter(fitness2_vals, fitness1_vals, c=colors, s=80, edgecolors='k', alpha=0.8)
    ax_p.set_title('Pareto Front (Objective Space)')
    ax_p.set_xlabel('Mean DCR (f2)')
    ax_p.set_ylabel('Normalized Cost+CO2 (f1)')
    ax_p.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "analysis_pareto_objective_space.png"))
    plt.close(fig_p)

    # 3. 수렴 그래프
    if hof_stats_history:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 7))
        gen_hof = [s['gen'] for s in hof_stats_history]
        hof_best_obj1 = [s['best_obj1'] for s in hof_stats_history]
        hof_best_obj2 = [s['best_obj2'] for s in hof_stats_history]
        
        color1 = 'tab:blue'
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Fitness1", color=color1)
        ax_conv.plot(gen_hof, hof_best_obj1, color=color1, marker='o')
        ax_conv.tick_params(axis='y', labelcolor=color1)
        
        ax_conv_twin = ax_conv.twinx()
        color2 = 'tab:red'
        ax_conv_twin.set_ylabel("Fitness2", color=color2)
        ax_conv_twin.plot(gen_hof, hof_best_obj2, color=color2, marker='s')
        ax_conv_twin.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Convergence of HOF')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "analysis_convergence_hof.png"))
        plt.close(fig_conv)

    print("All plots saved.")
