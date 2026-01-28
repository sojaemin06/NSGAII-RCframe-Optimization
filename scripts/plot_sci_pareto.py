
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuration
DATA_FILE = r"Results_Optimization_Paper_Final\Example_1_4Story\Data\pareto_summary.csv"
FIGURE_DIR = r"Results_Optimization_Paper_Final\Example_1_4Story\Figures"

# SCI Style Settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

def plot_figures():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    
    # --- Figure 1: Objective Space (Colorbar REMOVED, but Color PRESERVED) ---
    plt.figure(figsize=(7, 6))
    ax1 = plt.gca()
    
    x1 = df['Fit2(MaxDrift)']
    y1 = df['Fit1(CostCO2)']
    c1 = df['Max_Drift_Ratio_Percent'] # Use drift ratio for color mapping
    
    # Plot with viridis colormap to maintain color variation
    sc1 = ax1.scatter(x1, y1, c=c1, cmap='viridis', s=80, edgecolors='black', linewidth=0.8, alpha=0.9, zorder=3)
    
    # Do NOT add colorbar
    
    ax1.set_xlabel('Objective 2 (Max. Inter-story Drift Ratio)', fontsize=14)
    ax1.set_ylabel(r'Objective 1 (Normalized Cost + CO$_2$)', fontsize=14)
    
    limit_drift = 0.02
    ax1.axvline(x=limit_drift, color='red', linestyle='--', linewidth=1.5, label='Constraint Limit (2.0%)', zorder=2)
    ax1.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    f1_path = os.path.join(FIGURE_DIR, "analysis_pareto_objective_space.png")
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    print(f"Generated: {f1_path} (Color maintained, Colorbar removed)")
    plt.close()

    # --- Figure 2: Solution Space (Cost vs CO2 with Scaled Drift Colorbar) ---
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    
    x2 = df['CO2']
    y2 = df['Cost']
    c2 = df['Max_Drift_Ratio_Percent'] # Already scaled in previous step
    
    sc2 = ax2.scatter(x2, y2, c=c2, cmap='viridis', s=80, edgecolors='black', linewidth=0.8, alpha=0.9, zorder=3)
    
    cbar = plt.colorbar(sc2, ax=ax2)
    cbar.set_label('Max. Drift Ratio (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    
    ax2.set_xlabel(r'Total CO$_2$ emission (kgCO$_2$e)', fontsize=14)
    ax2.set_ylabel('Total Construction Cost (won)', fontsize=14)
    
    # Use Scientific notation for Y axis if values are large
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    f2_path = os.path.join(FIGURE_DIR, "analysis_pareto_solution_space_new.png")
    plt.savefig(f2_path, dpi=300, bbox_inches='tight')
    print(f"Generated: {f2_path} (Scaled Colorbar included)")
    plt.close()

if __name__ == "__main__":
    plot_figures()
