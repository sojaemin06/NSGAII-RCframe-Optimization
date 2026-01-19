import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ----------------------------------------------------------- 
# [설정] 데이터 경로 및 출력 설정
# ----------------------------------------------------------- 
INPUT_DIR = "Results_Param_Optimization"
OUTPUT_DIR = "Results_Param_Optimization"

# 논문용 폰트 설정
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 2

STEP_CONFIGS = {
    'Step1': {'param': 'Crossover', 'files': ['Step1_Crossover_HV_History.csv', 'Step1_Crossover_Pareto_Data.csv']},
    'Step2': {'param': 'Tournament', 'files': ['Step2_Tournament_HV_History.csv', 'Step2_Tournament_Pareto_Data.csv']},
    'Step3': {'param': 'CXPB', 'files': ['Step3_CXPB_HV_History.csv', 'Step3_CXPB_Pareto_Data.csv']},
    'Step4': {'param': 'MUTPB', 'files': ['Step4_MUTPB_HV_History.csv', 'Step4_MUTPB_Pareto_Data.csv']},
    'Step5': {'param': 'PopSize', 'files': ['Step5_PopSize_HV_History.csv', 'Step5_PopSize_Pareto_Data.csv'], 'selected': 400, 'filter_range': [100, 600]}
}

def get_best_worst_params(df_hist, param_col):
    """최종 HV 기준으로 Best/Worst 파라미터 식별"""
    final_hvs = df_hist.groupby(param_col)['hypervolume'].last()
    best_param = final_hvs.idxmax()
    worst_param = final_hvs.idxmin()
    return best_param, worst_param

def plot_hv_history(step_name, config):
    """Hypervolume 수렴 그래프 (Title 제거)"""
    hist_file = os.path.join(INPUT_DIR, config['files'][0])
    if not os.path.exists(hist_file):
        print(f"Skipping {step_name} HV: {hist_file} not found.")
        return

    df = pd.read_csv(hist_file)
    param_col = config['param']
    
    # 데이터 타입 정리 (숫자형 파라미터인 경우)
    if param_col in ['Tournament', 'PopSize']:
        df[param_col] = pd.to_numeric(df[param_col], errors='coerce')

    # 필터링 (Step 5의 경우)
    if 'filter_range' in config:
        min_val, max_val = config['filter_range']
        df = df[(df[param_col] >= min_val) & (df[param_col] <= max_val)]
    # Best/Worst 식별
    best_param, worst_param = get_best_worst_params(df, param_col)
    selected_param = config.get('selected', None)
    
    # Selected가 있으면 이를 Best로 간주
    if selected_param is not None:
        best_param = selected_param

    plt.figure(figsize=(8, 6))
    
    # 그룹별 플롯
    # 정렬을 위해 unique 값 추출 후 정렬
    unique_params = df[param_col].unique()
    try:
        sorted_params = sorted(unique_params)
    except:
        sorted_params = unique_params # 정렬 불가시 그대로

    for label in sorted_params:
        group = df[df[param_col] == label]
        
        # 라벨에 Best/Worst 표기 추가
        plot_label = f"{label}"
        if label == best_param:
            plot_label += " (Best)"
        elif label == worst_param:
            plot_label += " (Worst)"
            
        plt.plot(group['gen'], group['hypervolume'], label=plot_label)
        
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    # 범례 위치 조정
    plt.legend(title=param_col, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, f"{step_name}_{param_col}_HV_Paper.png")
    plt.savefig(out_file, dpi=300)
    print(f"-> Generated: {out_file}")
    plt.close()

def plot_pareto_front(step_name, config):
    """Pareto Front 비교 그래프 (Title 제거, Best/Worst/Selected 표기)"""
    hist_file = os.path.join(INPUT_DIR, config['files'][0])
    pareto_file = os.path.join(INPUT_DIR, config['files'][1])
    
    if not os.path.exists(hist_file) or not os.path.exists(pareto_file):
        print(f"Skipping {step_name} Pareto: Files not found.")
        return

    df_hist = pd.read_csv(hist_file)
    df_pareto = pd.read_csv(pareto_file)
    param_col = config['param']
    
    # 파라미터 컬럼명 통일 (Pareto 파일은 'Parameter' 컬럼 사용됨)
    df_pareto.rename(columns={'Parameter': param_col}, inplace=True)

    # 데이터 타입 정리
    if param_col in ['Tournament', 'PopSize']:
        df_hist[param_col] = pd.to_numeric(df_hist[param_col], errors='coerce')
        df_pareto[param_col] = pd.to_numeric(df_pareto[param_col], errors='coerce')

    # 필터링 (Step 5의 경우)
    if 'filter_range' in config:
        min_val, max_val = config['filter_range']
        df_hist = df_hist[(df_hist[param_col] >= min_val) & (df_hist[param_col] <= max_val)]
        df_pareto = df_pareto[(df_pareto[param_col] >= min_val) & (df_pareto[param_col] <= max_val)]

    # Best/Worst 식별
    best_param, worst_param = get_best_worst_params(df_hist, param_col)
    selected_param = config.get('selected', None)
    
    # Selected가 있으면 이를 Best로 간주 (연구 선정 파라미터 강조)
    if selected_param is not None:
        best_param = selected_param
    
    plt.figure(figsize=(8, 6))
    
    # 정렬된 파라미터 리스트
    try:
        all_params = sorted(df_pareto[param_col].unique())
    except:
        all_params = df_pareto[param_col].unique()

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_params)))
    
    # --- 플롯 순서: 일반 -> Worst -> Best -> Selected (가장 위) ---
    
    # 1. Others & General Drawing (범례 생성을 위해 루프) 
    for param, color in zip(all_params, colors):
        subset = df_pareto[df_pareto[param_col] == param]
        if subset.empty: continue
        
        # 기본 설정
        label_str = f"{param}"
        marker = 'o'
        size = 40
        edge = None
        zorder = 6
        alpha = 0.7
        
        is_special = False

        if param == worst_param:
            label_str = f"{param} (Worst)"
            marker = 'x'
            color = 'gray'
            size = 50
            alpha = 0.6
            zorder = 5
            is_special = True
            
        # Best 처리
        if param == best_param:
            label_str = f"{param} (Best)"
            marker = '*'
            color = 'red' # Best는 빨강
            size = 120
            edge = 'black'
            zorder = 10
            alpha = 1.0
            is_special = True

        # 일반 컬러맵 적용 (Special이 아니면)
        if not is_special:
            plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'],
                        c=[color], label=label_str, marker=marker, s=size, 
                        edgecolors=edge, linewidth=0.8 if edge else 1.0, 
                        alpha=alpha, zorder=zorder)
        else:
             plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'],
                        c=[color], label=label_str, marker=marker, s=size, 
                        edgecolors=edge, linewidth=0.8 if edge else 1.0, 
                        alpha=alpha, zorder=zorder)

    plt.xlabel('Resilience (Max Drift Ratio)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    
    # 범례 정렬
    handles, labels = plt.gca().get_legend_handles_labels()
    
    def sort_key(item):
        val = item[1].split()[0]
        try:
            return float(val)
        except:
            return val
            
    sorted_legend = sorted(zip(handles, labels), key=sort_key)
    plt.legend(*zip(*sorted_legend), loc='upper right', frameon=True, framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, f"{step_name}_{param_col}_Pareto_Paper.png")
    plt.savefig(out_file, dpi=300)
    print(f"-> Generated: {out_file}")
    plt.close()

def main():
    print("### Generating Paper Figures (No Titles, Best/Worst Marked) ###")
    
    for step_name, config in STEP_CONFIGS.items():
        print(f"\nProcessing {step_name} ({config['param']})...")
        plot_hv_history(step_name, config)
        plot_pareto_front(step_name, config)
        
    print("\nAll figures generated successfully.")

if __name__ == "__main__":
    main()
