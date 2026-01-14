import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------------------------------------
# [설정] 데이터 경로 및 출력 설정
# -----------------------------------------------------------
INPUT_DIR = "Results_Param_Optimization"
OUTPUT_DIR = "Results_Param_Optimization"
HV_HISTORY_FILE = os.path.join(INPUT_DIR, "Step5_PopSize_HV_History.csv")
PARETO_DATA_FILE = os.path.join(INPUT_DIR, "Step5_PopSize_Pareto_Data.csv")

# 논문용 폰트 설정
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.grid'] = True

def finalize_analysis():
    print("### Finalizing Step 5 Analysis (PopSize 100-600) ###")

    if not os.path.exists(HV_HISTORY_FILE) or not os.path.exists(PARETO_DATA_FILE):
        print(f"Error: Input files not found in {INPUT_DIR}")
        return

    # 1. Hypervolume History (Convergence)
    df_hist = pd.read_csv(HV_HISTORY_FILE)
    
    # 100~600 필터링
    df_hist_filtered = df_hist[(df_hist['PopSize'] >= 100) & (df_hist['PopSize'] <= 600)].copy()
    
    # CSV 저장
    hist_out_path = os.path.join(OUTPUT_DIR, "Step5_PopSize_100_600_HV_History.csv")
    df_hist_filtered.to_csv(hist_out_path, index=False)
    print(f"-> Saved filtered history to {hist_out_path}")

    # HV 그래프 그리기
    plt.figure(figsize=(10, 6))
    for pop_size, group in df_hist_filtered.groupby("PopSize"):
        plt.plot(group['gen'], group['hypervolume'], marker='o', markersize=3, label=f"Pop {pop_size}")
    
    plt.title('Hypervolume Convergence (PopSize 100-600)', fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend(title='Population Size', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Step5_PopSize_100_600_HV.png"), dpi=300)
    print("-> Saved filtered HV plot.")
    plt.close()

    # 2. Pareto Front (HOF)
    df_pareto = pd.read_csv(PARETO_DATA_FILE)
    
    # Parameter 컬럼이 PopSize를 의미한다고 가정 (병합 스크립트 로직 따름)
    # 데이터 타입 확인 및 변환
    df_pareto['Parameter'] = pd.to_numeric(df_pareto['Parameter'], errors='coerce')
    df_pareto_filtered = df_pareto[(df_pareto['Parameter'] >= 100) & (df_pareto['Parameter'] <= 600)].copy()
    
    # CSV 저장
    pareto_out_path = os.path.join(OUTPUT_DIR, "Step5_PopSize_100_600_Pareto_Data.csv")
    df_pareto_filtered.to_csv(pareto_out_path, index=False)
    print(f"-> Saved filtered Pareto data to {pareto_out_path}")

    # Pareto 그래프 그리기
    plt.figure(figsize=(10, 8))
    
    params = sorted(df_pareto_filtered['Parameter'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
    
    # Pop 400 강조를 위한 설정
    best_pop = 400

    for param, color in zip(params, colors):
        subset = df_pareto_filtered[df_pareto_filtered['Parameter'] == param]
        if subset.empty: continue
        
        # 라벨 및 스타일 설정
        label_str = f"Pop {int(param)}"
        
        if param == best_pop:
            plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'], 
                        color='red', label=f"{label_str} (Selected)", 
                        s=80, marker='*', edgecolors='black', zorder=10)
        else:
            plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'], 
                        color=color, label=label_str, 
                        s=40, alpha=0.7, zorder=5)

    plt.title('Pareto Front Comparison (PopSize 100-600)', fontweight='bold')
    plt.xlabel('Resilience (Max Drift Ratio)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Step5_PopSize_100_600_Pareto_All.png"), dpi=300)
    print("-> Saved filtered Pareto plot.")
    plt.close()

if __name__ == "__main__":
    finalize_analysis()
