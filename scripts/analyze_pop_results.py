import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

OUTPUT_ROOT = "Results_Param_Optimization"
STEP_NAME = "Step5_PopSize"

def main():
    print("Combining and Analyzing Population Size Experiment Results...")
    
    # 1. Hypervolume Convergence History
    all_hist_files = glob.glob(os.path.join(OUTPUT_ROOT, "Step5_Pop_*_History.csv"))
    if not all_hist_files:
        print("No history files found!")
        return

    plt.figure(figsize=(10, 6))
    final_hvs = {}
    
    # 색상 맵 설정 (10개 구분)
    pop_sizes = sorted([int(f.split('_Pop_')[1].split('_History')[0]) for f in all_hist_files])
    colors = plt.cm.viridis(np.linspace(0, 1, len(pop_sizes)))
    color_map = {p: c for p, c in zip(pop_sizes, colors)}

    for f in all_hist_files:
        df = pd.read_csv(f)
        pop_size = df['PopSize'].iloc[0]
        
        # 마지막 세대 HV 저장
        final_hvs[pop_size] = df.iloc[-1]['hypervolume']
        
        plt.plot(df['gen'], df['hypervolume'], label=f"Pop {pop_size}", color=color_map[pop_size])

    plt.title('Population Size - Hypervolume Convergence')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, f'{STEP_NAME}_HV.png'))
    print(f"Saved: {STEP_NAME}_HV.png")

    # 2. Pareto Front Comparison
    all_pareto_files = glob.glob(os.path.join(OUTPUT_ROOT, "Step5_Pop_*_Pareto.csv"))
    if not all_pareto_files:
        print("No pareto files found!")
        return

    df_combined = pd.concat([pd.read_csv(f) for f in all_pareto_files], ignore_index=True)
    
    # Best Param 찾기 (HV 기준)
    best_pop = max(final_hvs, key=final_hvs.get)
    worst_pop = min(final_hvs, key=final_hvs.get)
    
    print(f"Best PopSize: {best_pop} (HV={final_hvs[best_pop]:.4f})")
    
    # Pareto Plot
    plt.figure(figsize=(10, 8))
    
    for pop in pop_sizes:
        subset = df_combined[df_combined['Parameter'] == pop]
        if subset.empty: continue
        
        label_str = f"Pop {pop}"
        color = color_map[pop]
        
        if pop == best_pop:
            plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'], 
                        c='red', label=f'{label_str} (Best)', s=100, marker='*', edgecolors='black', zorder=10)
        elif pop == worst_pop:
            plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'], 
                        c='gray', label=f'{label_str} (Worst)', s=30, marker='x', alpha=0.5, zorder=1)
        else:
            plt.scatter(subset['Obj2_MaxDrift'], subset['Obj1_NormCostCO2'], 
                        color=color, label=label_str, s=40, alpha=0.7, zorder=5)

    plt.title('Population Size - Pareto Front Comparison')
    plt.xlabel('Resilience (Max Drift Ratio)')
    plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, f'{STEP_NAME}_Pareto_All.png'))
    print(f"Saved: {STEP_NAME}_Pareto_All.png")
    
    # 3. Save Summary CSV
    summary_csv_path = os.path.join(OUTPUT_ROOT, f'{STEP_NAME}_Summary.csv')
    
    summary_data = []
    for pop, hv in final_hvs.items():
        # 실행 시간은 로그에서 추출해야 하지만 여기선 생략하고 HV 위주로 기록
        summary_data.append({'PopSize': pop, 'Final_Hypervolume': hv})
        
    pd.DataFrame(summary_data).sort_values('PopSize').to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")

if __name__ == "__main__":
    main()
