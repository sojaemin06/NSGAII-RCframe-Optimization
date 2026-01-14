import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Output Directory
OUTPUT_ROOT = "Results_Param_Optimization"

def main():
    print("### Merging Step 5 (Population Size) Results ###")
    
    parts = ["Step5_PopSize_Part1", "Step5_PopSize_Part2", "Step5_PopSize_Part3"]
    
    # ---------------------------------------------------------
    # 1. Merge History Data (Convergence)
    # ---------------------------------------------------------
    history_dfs = []
    for part in parts:
        csv_path = os.path.join(OUTPUT_ROOT, f"{part}_HV_History.csv")
        if os.path.exists(csv_path):
            print(f"Loading {csv_path}...")
            history_dfs.append(pd.read_csv(csv_path))
        else:
            print(f"[Warning] {csv_path} not found. Skipping.")
    
    if history_dfs:
        full_history = pd.concat(history_dfs, ignore_index=True)
        # Save Combined History
        full_hist_path = os.path.join(OUTPUT_ROOT, "Step5_PopSize_HV_History.csv")
        full_history.to_csv(full_hist_path, index=False)
        print(f"-> Combined history saved to {full_hist_path}")
        
        # Plot Hypervolume Convergence
        plt.figure(figsize=(12, 7))
        final_hvs = {}
        # PopSize별로 그룹화하여 그리기
        for label, group in full_history.groupby("PopSize"):
            plt.plot(group['gen'], group['hypervolume'], marker='o', markersize=3, label=f"Pop: {label}")
            final_hvs[label] = group.iloc[-1]['hypervolume']
            
        plt.title('Step 5: Population Size - Hypervolume Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume')
        plt.legend(title="Population Size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, 'Step5_PopSize_HV.png'))
        plt.close()
        print("-> Convergence plot saved.")
    else:
        print("[Error] No history data found.")

    # ---------------------------------------------------------
    # 2. Merge Pareto Data (HOF)
    # ---------------------------------------------------------
    pareto_dfs = []
    for part in parts:
        csv_path = os.path.join(OUTPUT_ROOT, f"{part}_Pareto_Data.csv")
        if os.path.exists(csv_path):
            print(f"Loading {csv_path}...")
            pareto_dfs.append(pd.read_csv(csv_path))
    
    if pareto_dfs:
        full_pareto = pd.concat(pareto_dfs, ignore_index=True)
        # Save Combined Pareto
        full_pareto_path = os.path.join(OUTPUT_ROOT, "Step5_PopSize_Pareto_Data.csv")
        full_pareto.to_csv(full_pareto_path, index=False)
        print(f"-> Combined Pareto data saved to {full_pareto_path}")
        
        # Plot Pareto Fronts
        plt.figure(figsize=(12, 9))
        
        all_params = sorted(full_pareto['Parameter'].unique()) # PopSizes
        
        # Best/Worst (HV 기준)
        if final_hvs:
            best_param = max(final_hvs, key=final_hvs.get)
            worst_param = min(final_hvs, key=final_hvs.get)
            print(f"Best PopSize: {best_param} (HV={final_hvs[best_param]:.4f})")
        else:
            best_param, worst_param = None, None

        colors = plt.cm.viridis(np.linspace(0, 1, len(all_params)))
        
        for param, color in zip(all_params, colors):
            subset = full_pareto[full_pareto['Parameter'] == param]
            if subset.empty: continue

            fit1 = subset['Obj1_NormCostCO2']
            fit2 = subset['Obj2_MaxDrift']
            
            label_str = f"Pop {param}"
            
            if param == best_param:
                plt.scatter(fit2, fit1, c='red', label=f'{label_str} (Best)', s=100, marker='*', edgecolors='black', zorder=10)
            elif param == worst_param:
                plt.scatter(fit2, fit1, c='gray', label=f'{label_str} (Worst)', s=40, marker='x', alpha=0.5, zorder=1)
            else:
                plt.scatter(fit2, fit1, color=color, label=label_str, s=50, alpha=0.7, zorder=5)
        
        plt.title('Step 5: Population Size - Pareto Front Comparison')
        plt.xlabel('Resilience (Max Drift Ratio)')
        plt.ylabel('Economic & Env. Demand (Norm Cost+CO2)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, 'Step5_PopSize_Pareto_All.png'))
        plt.close()
        print("-> Pareto comparison plot saved.")
    else:
        print("[Error] No Pareto data found.")

if __name__ == "__main__":
    main()
