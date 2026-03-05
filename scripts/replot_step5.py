import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# --- SCI Paper Style Configuration (from plot_param_optimizaion.py) ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.7

def replot_step5(base_dir, output_path, target_norm=0.02):
    pop_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    all_data = []
    
    # 1. HV 데이터 수집하여 Best/Worst 판별
    hv_summary = []
    for size in pop_sizes:
        csv_name = f"Step5_Pop_{size}_History.csv"
        csv_path = os.path.join(base_dir, csv_name)
        if os.path.exists(csv_path):
            df_hv = pd.read_csv(csv_path)
            hv_summary.append({'Pop_Size': size, 'Hypervolume': df_hv['hypervolume'].iloc[-1]})
    
    if not hv_summary:
        print("No Step 5 HV data found.")
        return
        
    hv_df = pd.DataFrame(hv_summary)
    best_size = hv_df.loc[hv_df['Hypervolume'].idxmax(), 'Pop_Size']
    worst_size = hv_df.loc[hv_df['Hypervolume'].idxmin(), 'Pop_Size']

    # 2. Pareto 데이터 수집 및 시각화
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, size in enumerate(pop_sizes):
        csv_name = f"Step5_Pop_{size}_Pareto.csv"
        csv_path = os.path.join(base_dir, csv_name)
        if not os.path.exists(csv_path): continue
        
        df = pd.read_csv(csv_path)
        x_col = 'Obj2_MaxDrift' if 'Obj2_MaxDrift' in df.columns else ('Max_Drift' if 'Max_Drift' in df.columns else 'max_drift')
        y_col = 'Obj1_NormCostCO2' if 'Obj1_NormCostCO2' in df.columns else ('obj1' if 'obj1' in df.columns else 'obj1')
        
        df['Obj2_Norm'] = df[x_col] / target_norm
        
        label_str = f"Pop {size}"
        color = colors[i % len(colors)]
        alpha = 0.4 # Increased from 0.2
        zorder = 1
        
        if size == best_size:
            label_str += " (best)"
            alpha = 1.0
            zorder = 10
        elif size == worst_size:
            label_str += " (worst)"
            alpha = 0.8
            zorder = 5
        else:
            color = 'darkgrey' # Changed from lightgrey

        plt.scatter(df['Obj2_Norm'], df[y_col], 
                    marker=markers[i % len(markers)], 
                    s=50, 
                    color=color,
                    alpha=alpha,
                    edgecolors='none' if color == 'darkgrey' else 'k',
                    label=label_str,
                    zorder=zorder)

    plt.xlabel('Objective 2 (Normalized Max. Drift Ratio)')
    plt.ylabel(r'Objective 1 (Normalized Cost + CO$_2$)')
    plt.legend(frameon=True, loc='upper right', ncol=2, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Successfully synchronized Step 5: {output_path}")

if __name__ == "__main__":
    replot_step5("Results_Param_Optimization", "Results_Param_Optimization/Step5_PopSize_Pareto_Paper.png")
