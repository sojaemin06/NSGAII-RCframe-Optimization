
import os
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_pareto_comparison():
    # Data Paths
    path_a = os.path.join("Results_Optimization_Paper_Final", "Example_1_4Story", "Data", "pareto_summary.csv")
    path_b = os.path.join("Results_Scenario_Comparison", "Scenario_B_Conventional", "Data", "pareto_summary.csv")
    
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print("Pareto summary files not found.")
        return

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    
    plt.figure(figsize=(8, 6))
    
    # Plot Scenario A
    # Fit2 is Max Drift, Fit1 is Normalized Cost + CO2
    plt.scatter(df_a['Fit2(MaxDrift)'], df_a['Fit1(CostCO2)'], 
                c='blue', marker='o', s=50, alpha=0.7, edgecolors='k', label='Scenario A (Proposed)')
    
    # Plot Scenario B
    plt.scatter(df_b['Fit2(MaxDrift)'], df_b['Fit1(CostCO2)'], 
                c='red', marker='x', s=50, alpha=0.8, label='Scenario B (Conventional)')
    
    # Standardizing axis range for fair comparison (optional, but good for Pareto fronts)
    # plt.xlim(0.005, 0.021) 
    
    plt.xlabel('Objective 2 (Max. Inter-story Drift Ratio)')
    plt.ylabel(r'Objective 1 (Normalized Cost + CO$_2$)')
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join("Results_Scenario_Comparison", "Comparison_Pareto_Front.png")
    plt.savefig(output_path)
    print(f"Pareto comparison plot saved to: {output_path}")

if __name__ == "__main__":
    plot_pareto_comparison()
