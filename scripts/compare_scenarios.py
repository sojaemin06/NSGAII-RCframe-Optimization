import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    """
    두 시나리오(A, B)의 최적화 결과를 읽어와 수렴 과정과 최종 파레토 전선을 비교하는
    그래프를 생성하고 저장합니다.
    """
    # --- 그래프 전역 설정 ---
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 14 # 개별적으로 폰트 크기 설정
    plt.rcParams['axes.unicode_minus'] = False

    # --- 데이터 로드 ---
    try:
        # 수렴 과정 데이터
        hof_conv_a = pd.read_csv(os.path.join("../results/Results_Scenario_A_02", "hof_convergence_A.csv"))
        hof_conv_b = pd.read_csv(os.path.join("../results/Results_Scenario_C_02", "hof_convergence_C.csv"))
        
        # 최종 파레토 해 데이터
        pareto_a = pd.read_csv(os.path.join("../results/Results_Scenario_A_02", "pareto_solutions_A.csv"))
        pareto_b = pd.read_csv(os.path.join("../results/Results_Scenario_C_02", "pareto_solutions_C.csv"))
    except FileNotFoundError as e:
        print(f"오류: 결과 파일을 찾을 수 없습니다. '{e.filename}'")
        print("먼저 experiment_scenario_A.py와 experiment_scenario_C.py를 실행하여 결과 파일을 생성해야 합니다.")
        return

    # --- 각 전략에 대한 스타일 정의 (Scenario C에 맞게 레이블 수정) ---
    style_a = {'color': '#1f77b4', 'marker': 'o', 'label': 'Scenario A (Base DB + Rotation Var)'}
    style_b = {'color': '#ff7f0e', 'marker': 's', 'label': 'Scenario B (Expanded DB, No Rotation)'}

    # --- 1. 수렴도 그래프 생성 ---
    fig_conv, axes_conv = plt.subplots(1, 2, figsize=(18, 7))
    
    # 목표함수 1 (Cost+CO2) 수렴도
    ax1 = axes_conv[0]
    ax1.plot(hof_conv_a['gen'], hof_conv_a['best_obj1'], **style_a, linestyle='-', linewidth=2.5, markersize=4)
    ax1.plot(hof_conv_b['gen'], hof_conv_b['best_obj1'], **style_b, linestyle='--', linewidth=2.5, markersize=4)
    ax1.set_xlabel('Generation', fontsize=16)
    ax1.set_ylabel('Best Fitness 1 in Hall of Fame', fontsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 목표함수 2 (Mean DCR) 수렴도
    ax2 = axes_conv[1]
    ax2.plot(hof_conv_a['gen'], hof_conv_a['best_obj2'], **style_a, linestyle='-', markersize=4)
    ax2.plot(hof_conv_b['gen'], hof_conv_b['best_obj2'], **style_b, linestyle='--', markersize=4)
    ax2.set_xlabel('Generation', fontsize=16)
    ax2.set_ylabel('Best Fitness 2 in Hall of Fame', fontsize=16)
    ax2.legend(fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig_conv.tight_layout()
    save_path_conv = "../results/scenarios_convergence_comparison.png"
    fig_conv.savefig(save_path_conv, dpi=300)
    print(f"수렴도 비교 그래프가 '{save_path_conv}' 파일로 저장되었습니다.")
    plt.show()

    # --- 2. 파레토 최적 전선 그래프 생성 ---
    fig_pareto, axes_pareto = plt.subplots(1, 2, figsize=(18, 7))

    # 최종 파레토 전선 (목표 공간)
    ax3 = axes_pareto[0]
    ax3.scatter(pareto_a['fitness_obj2'], pareto_a['fitness_obj1'], s=100, edgecolors='k', alpha=0.7, **style_a)
    ax3.scatter(pareto_b['fitness_obj2'], pareto_b['fitness_obj1'], s=100, edgecolors='k', alpha=0.7, **style_b)
    ax3.set_xlabel('Fitness 2 (Mean DCR)', fontsize=16)
    ax3.set_ylabel('Fitness 1 (Normalized Cost+CO2)', fontsize=16)
    ax3.legend(fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 최종 파레토 전선 (실제 값 공간)
    ax4 = axes_pareto[1]
    ax4.scatter(pareto_a['cost'], pareto_a['co2'], s=100, edgecolors='k', alpha=0.7, **style_a)
    ax4.scatter(pareto_b['cost'], pareto_b['co2'], s=100, edgecolors='k', alpha=0.7, **style_b)
    ax4.set_xlabel('Total Cost', fontsize=16)
    ax4.set_ylabel('Total CO2', fontsize=16)
    ax4.legend(fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.6)

    fig_pareto.tight_layout()
    save_path_pareto = "../results/scenarios_pareto_front_comparison.png"
    fig_pareto.savefig(save_path_pareto, dpi=300)
    print(f"파레토 전선 비교 그래프가 '{save_path_pareto}' 파일로 저장되었습니다.")
    plt.show()

if __name__ == '__main__':
    plot_comparison()
