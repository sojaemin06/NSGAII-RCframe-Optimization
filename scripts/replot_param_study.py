import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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

def replot_pareto(csv_path, output_path, target_norm=0.02):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    x_col = 'Obj2_MaxDrift' if 'Obj2_MaxDrift' in df.columns else ('Max_Drift' if 'Max_Drift' in df.columns else 'Max_Drift')
    y_col = 'Obj1_NormCostCO2' if 'Obj1_NormCostCO2' in df.columns else ('obj1' if 'obj1' in df.columns else 'obj1')
    hue_col = 'Parameter' if 'Parameter' in df.columns else 'Parameter'
    
    # Identify best and worst based on Hypervolume column (Same logic as HV plot)
    if 'Hypervolume' in df.columns:
        hv_stats = df.groupby(hue_col)['Hypervolume'].first()
        best_param = hv_stats.idxmax()
        worst_param = hv_stats.idxmin()
    else:
        best_param = None
        worst_param = None

    df['Obj2_Norm'] = df[x_col] / target_norm
    plt.figure(figsize=(8, 6))
    
    params = df[hue_col].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, param in enumerate(params):
        subset = df[df[hue_col] == param]
        label_str = str(param).replace('_', ' ')
        
        color = colors[i % len(colors)]
        alpha = 0.4 # Increased from 0.2
        zorder = 1
        
        if param == best_param:
            label_str += " (best)"
            alpha = 1.0
            zorder = 10
        elif param == worst_param:
            label_str += " (worst)"
            alpha = 0.8
            zorder = 5
        else:
            color = 'darkgrey' # Changed from lightgrey
            
        plt.scatter(subset['Obj2_Norm'], subset[y_col], 
                    marker=markers[i % len(markers)], 
                    s=60, 
                    color=color,
                    alpha=alpha,
                    edgecolors='none' if color == 'darkgrey' else 'k',
                    label=label_str,
                    zorder=zorder)

    plt.xlabel('Objective 2 (Normalized Max. Drift Ratio)')
    plt.ylabel(r'Objective 1 (Normalized Cost + CO$_2$)')
    plt.legend(frameon=True, loc='upper right')
    
    # X축 범위 조정
    current_xmax = df['Obj2_Norm'].max()
    plt.xlim(left=0, right=max(1.05, current_xmax * 1.05))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Successfully synchronized: {output_path}")

if __name__ == "__main__":
    base_dir = "Results_Param_Optimization"
    steps = [
        ("Step1_Crossover_Pareto_Data.csv", "Step1_Crossover_Pareto_Paper.png"),
        ("Step2_Tournament_Pareto_Data.csv", "Step2_Tournament_Pareto_Paper.png"),
        ("Step3_CXPB_Pareto_Data.csv", "Step3_CXPB_Pareto_Paper.png"),
        ("Step4_MUTPB_Pareto_Data.csv", "Step4_MUTPB_Pareto_Paper.png")
    ]
    for csv, img in steps:
        replot_pareto(os.path.join(base_dir, csv), os.path.join(base_dir, img))
