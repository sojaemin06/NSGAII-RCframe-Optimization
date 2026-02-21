
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- SCI Paper Style Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.7

BASE_DIR = "Results_Param_Optimization"

def plot_hv_history(step_name, csv_file, x_col='gen', y_col='hypervolume', hue_col='Experiment', df=None):
    if df is None:
        data_path = os.path.join(BASE_DIR, csv_file)
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            return
        df = pd.read_csv(data_path)
    
    plt.figure(figsize=(8, 6))
    
    # Identify best and worst based on final hypervolume
    final_gen = df[x_col].max()
    hv_stats = df[df[x_col] == final_gen].groupby(hue_col)[y_col].first()
    if hv_stats.empty:
        hv_stats = df.groupby(hue_col)[y_col].last()
    
    best_exp = hv_stats.idxmax()
    worst_exp = hv_stats.idxmin()

    # Get unique experiments for styling
    experiments = df[hue_col].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, exp in enumerate(experiments):
        subset = df[df[hue_col] == exp]
        if subset.empty: continue
            
        label_str = str(exp).replace('_', ' ')
        color = colors[i % len(colors)]
        alpha = 0.3
        zorder = 1
        linewidth = 1
        
        if exp == best_exp:
            label_str += " (best)"
            alpha = 1.0
            zorder = 10
            linewidth = 2.5
        elif exp == worst_exp:
            label_str += " (worst)"
            alpha = 0.8
            zorder = 5
            linewidth = 2.0
        else:
            color = 'lightgrey'
            
        plt.plot(subset[x_col], subset[y_col], 
                 marker=markers[i % len(markers)], 
                 markersize=4, 
                 linestyle='-', 
                 color=color,
                 alpha=alpha,
                 linewidth=linewidth,
                 label=label_str,
                 zorder=zorder)

    plt.xlabel('Generation')
    plt.ylabel('Hypervolume Indicator')
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, f"{step_name}_HV_Paper.png")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def plot_pareto_front(step_name, csv_file, x_col='Obj2_MaxDrift', y_col='Obj1_NormCostCO2', hue_col='Parameter', df=None):
    if df is None:
        data_path = os.path.join(BASE_DIR, csv_file)
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            return
        df = pd.read_csv(data_path)
    
    plt.figure(figsize=(8, 6))
    
    # Identify best and worst based on Hypervolume column
    if 'Hypervolume' in df.columns:
        hv_stats = df.groupby(hue_col)['Hypervolume'].first()
        best_param = hv_stats.idxmax()
        worst_param = hv_stats.idxmin()
    else:
        best_param = None
        worst_param = None

    # Get unique parameters for styling
    params = df[hue_col].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, param in enumerate(params):
        subset = df[df[hue_col] == param]
        label_str = str(param).replace('_', ' ')
        
        color = colors[i % len(colors)]
        alpha = 0.2
        zorder = 1
        
        if param == best_param:
            label_str += " (best)"
            alpha = 0.9
            zorder = 10
        elif param == worst_param:
            label_str += " (worst)"
            alpha = 0.7
            zorder = 5
        else:
            color = 'lightgrey'
            
        plt.scatter(subset[x_col], subset[y_col], 
                    marker=markers[i % len(markers)], 
                    s=50, 
                    color=color,
                    alpha=alpha,
                    edgecolors='none' if color == 'lightgrey' else 'k',
                    label=label_str,
                    zorder=zorder)

    plt.xlabel('Objective 2 (Max. Inter-story Drift Ratio)')
    plt.ylabel(r'Objective 1 (Normalized Cost + CO$_2$)')
    plt.legend(frameon=True)
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, f"{step_name}_Pareto_Paper.png")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def collect_step5_data():
    import glob
    import re
    
    hv_files = glob.glob(os.path.join(BASE_DIR, "Step5_Pop_*_History.csv"))
    pareto_files = glob.glob(os.path.join(BASE_DIR, "Step5_Pop_*_Pareto.csv"))
    
    hv_dfs = []
    for f in hv_files:
        pop_size = re.search(r"Pop_(\d+)_", f).group(1)
        temp_df = pd.read_csv(f)
        temp_df['Experiment'] = f"Pop {pop_size}"
        hv_dfs.append(temp_df)
    
    pareto_dfs = []
    for f in pareto_files:
        pop_size = re.search(r"Pop_(\d+)_", f).group(1)
        temp_df = pd.read_csv(f)
        temp_df['Parameter'] = f"Pop {pop_size}"
        pareto_dfs.append(temp_df)
        
    hv_all = pd.concat(hv_dfs, ignore_index=True) if hv_dfs else None
    pareto_all = pd.concat(pareto_dfs, ignore_index=True) if pareto_dfs else None
    
    return hv_all, pareto_all

def main():
    steps = [
        ("Step1_Crossover", "Step1_Crossover_HV_History.csv", "Step1_Crossover_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step2_Tournament", "Step2_Tournament_HV_History.csv", "Step2_Tournament_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step3_CXPB", "Step3_CXPB_HV_History.csv", "Step3_CXPB_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step4_MUTPB", "Step4_MUTPB_HV_History.csv", "Step4_MUTPB_Pareto_Data.csv", "Experiment", "Parameter"),
    ]

    for step_name, hv_file, pareto_file, hv_hue, pareto_hue in steps:
        plot_hv_history(step_name, hv_file, hue_col=hv_hue)
        plot_pareto_front(step_name, pareto_file, hue_col=pareto_hue)
        
    # Handle Step 5 separately
    print("Processing Step 5: Population Size (Aggregated)")
    hv_step5, pareto_step5 = collect_step5_data()
    if hv_step5 is not None:
        # Sort by population size for consistent legend
        def get_pop(x):
            try: return int(x.split(' ')[1])
            except: return 0
        
        hv_step5['pop_val'] = hv_step5['Experiment'].apply(get_pop)
        hv_step5 = hv_step5.sort_values(['pop_val', 'gen'])
        
        pareto_step5['pop_val'] = pareto_step5['Parameter'].apply(get_pop)
        pareto_step5 = pareto_step5.sort_values(['pop_val', 'Obj2_MaxDrift'])
        
        plot_hv_history("Step5_PopSize", None, hue_col="Experiment", df=hv_step5)
        plot_pareto_front("Step5_PopSize", None, hue_col="Parameter", df=pareto_step5)

if __name__ == "__main__":
    main()
