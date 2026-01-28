
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

def plot_hv_history(step_name, csv_file, x_col='gen', y_col='hypervolume', hue_col='Experiment'):
    data_path = os.path.join(BASE_DIR, csv_file)
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    plt.figure(figsize=(8, 6))
    
    # Get unique experiments for styling
    experiments = df[hue_col].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, exp in enumerate(experiments):
        subset = df[df[hue_col] == exp]
        # Only plot if we have data
        if not subset.empty:
            plt.plot(subset[x_col], subset[y_col], 
                     marker=markers[i % len(markers)], 
                     markersize=4, 
                     linestyle='-', 
                     label=exp.replace('_', ' '))

    plt.xlabel('Generation')
    plt.ylabel('Hypervolume Indicator')
    # plt.title(f'{step_name}: Hypervolume Convergence') # No title for paper
    plt.legend(frameon=True)
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, f"{step_name}_HV_Paper.png")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def plot_pareto_front(step_name, csv_file, x_col='Obj2_MaxDrift', y_col='Obj1_NormCostCO2', hue_col='Parameter'):
    data_path = os.path.join(BASE_DIR, csv_file)
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    plt.figure(figsize=(8, 6))
    
    # Get unique parameters for styling
    params = df[hue_col].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(params))) if len(params) > 0 else []
    
    # Custom color palette if needed, or use seaborn/matplotlib defaults
    
    for i, param in enumerate(params):
        subset = df[df[hue_col] == param]
        label_str = str(param).replace('_', ' ')
        
        plt.scatter(subset[x_col], subset[y_col], 
                    marker=markers[i % len(markers)], 
                    s=50, 
                    alpha=0.7, 
                    edgecolors='k',
                    label=label_str)

    plt.xlabel('Objective 2 (Max. Inter-story Drift Ratio)')
    plt.ylabel(r'Objective 1 (Normalized Cost + CO$_2$)')
    # plt.title(f'{step_name}: Pareto Front') # No title for paper
    plt.legend(frameon=True)
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, f"{step_name}_Pareto_Paper.png")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    import numpy as np # Import locally if needed or at top level
    
    steps = [
        ("Step1_Crossover", "Step1_Crossover_HV_History.csv", "Step1_Crossover_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step2_Tournament", "Step2_Tournament_HV_History.csv", "Step2_Tournament_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step3_CXPB", "Step3_CXPB_HV_History.csv", "Step3_CXPB_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step4_MUTPB", "Step4_MUTPB_HV_History.csv", "Step4_MUTPB_Pareto_Data.csv", "Experiment", "Parameter"),
        ("Step5_PopSize", "Step5_PopSize_HV_History.csv", "Step5_PopSize_Pareto_Data.csv", "Experiment", "Parameter"),
    ]

    for step_name, hv_file, pareto_file, hv_hue, pareto_hue in steps:
        plot_hv_history(step_name, hv_file, hue_col=hv_hue)
        plot_pareto_front(step_name, pareto_file, hue_col=pareto_hue)

if __name__ == "__main__":
    main()
