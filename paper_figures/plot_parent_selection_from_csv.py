import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison_from_csv():
    """
    Generates comparison plots for parent selection strategies
    by reading data from previously saved CSV files.
    """
    print("="*80)
    print("### Generating Final Comparison Plots from CSV Data ###")

    # --- Configuration ---
    # List of experiment methods to plot. This should match the folder names.
    parent_selection_methods = [
        'Random_Selection',
        'Tournament_Size_2',
        'Tournament_Size_3',
        'Tournament_Size_5',
        'Tournament_Size_7',
        'Tournament_Size_9',
        'Tournament_Size_11',
        'Best_Selection'
    ]

    # Style mapping for plots
    style_map = {
        'Random_Selection':    {'color': 'C0', 'marker': 'o', 'linestyle': '-'},
        'Tournament_Size_2':   {'color': 'C1', 'marker': 's', 'linestyle': '--'},
        'Tournament_Size_3':   {'color': 'C2', 'marker': 'P', 'linestyle': ':'},
        'Tournament_Size_5':   {'color': 'C3', 'marker': 'D', 'linestyle': '-.'},
        'Tournament_Size_7':   {'color': 'C4', 'marker': 'v', 'linestyle': '-'},
        'Tournament_Size_9':   {'color': 'C5', 'marker': '^', 'linestyle': '--'},
        'Tournament_Size_11':  {'color': 'C6', 'marker': '<', 'linestyle': ':'},
        'Best_Selection':      {'color': 'C7', 'marker': '>', 'linestyle': '-.'}
    }

    # --- Plotting Setup ---
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Figure 1 for Convergence
    fig_conv, (ax_conv1, ax_conv2) = plt.subplots(1, 2, figsize=(18, 7))

    # Figure 2 for Pareto Fronts
    fig_pareto, (ax_pareto_obj, ax_pareto_real) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Loop through methods, load data, and plot ---
    for method in parent_selection_methods:
        print(f"Processing method: {method}")
        
        method_folder = f"parent_selection_{method}_02"
        
        if not os.path.isdir(method_folder):
            print(f"  [Warning] Directory not found, skipping: {method_folder}")
            continue

        style = style_map.get(method, {'color': 'gray', 'marker': 'x', 'linestyle': '-.'})

        # 1. Plot Convergence Data from HOF CSV
        hof_csv_path = os.path.join(method_folder, "convergence_hof_data.csv")
        if os.path.exists(hof_csv_path):
            try:
                df_hof = pd.read_csv(hof_csv_path)
                gen = df_hof['gen']
                best_obj1 = df_hof['best_obj1']
                best_obj2 = df_hof['best_obj2']
                
                ax_conv1.plot(gen, best_obj1, label=method, alpha=0.9, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], markersize=4)
                ax_conv2.plot(gen, best_obj2, label=method, alpha=0.9, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], markersize=4)
                print(f"  - Plotted convergence data from {hof_csv_path}")
            except Exception as e:
                print(f"  [Error] Could not read or plot {hof_csv_path}: {e}")
        else:
            print(f"  [Warning] Convergence CSV not found: {hof_csv_path}")

        # 2. Plot Pareto Front Data from Pareto CSV
        pareto_csv_path = os.path.join(method_folder, "pareto_front_data.csv")
        if os.path.exists(pareto_csv_path):
            try:
                df_pareto = pd.read_csv(pareto_csv_path)
                
                # Objective Space Plot
                fitness1_vals = df_pareto['fitness1_obj']
                fitness2_vals = df_pareto['fitness2_obj']
                ax_pareto_obj.scatter(fitness2_vals, fitness1_vals, s=80, edgecolors='k', alpha=0.7, label=method, color=style['color'], marker=style['marker'])

                # Real Space Plot
                costs = df_pareto['cost']
                co2s = df_pareto['co2']
                ax_pareto_real.scatter(costs, co2s, s=80, edgecolors='k', alpha=0.7, label=method, color=style['color'], marker=style['marker'])
                print(f"  - Plotted Pareto front data from {pareto_csv_path}")
            except Exception as e:
                print(f"  [Error] Could not read or plot {pareto_csv_path}: {e}")
        else:
            print(f"  [Warning] Pareto front CSV not found: {pareto_csv_path}")

    # --- Finalize and Save Plot ---
    # Finalize Convergence Plot
    ax_conv1.set_xlabel('Generation', fontsize=16); ax_conv1.set_ylabel('Best Fitness1 in Hof', fontsize=16); ax_conv1.legend(fontsize=14); ax_conv1.grid(True)
    ax_conv2.set_xlabel('Generation', fontsize=16); ax_conv2.set_ylabel('Best Fitness2 in Hof', fontsize=16); ax_conv2.legend(fontsize=14); ax_conv2.grid(True)
    fig_conv.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path_conv = "parent_selection_convergence.png"
    fig_conv.savefig(save_path_conv)
    print(f"\nConvergence comparison plot saved to: {save_path_conv}")

    # Finalize Pareto Plot
    ax_pareto_obj.set_xlabel('Fitness2 (Mean DCR)', fontsize=16); ax_pareto_obj.set_ylabel('Fitness1 (Normalized Cost + CO2)', fontsize=16); ax_pareto_obj.legend(fontsize=14); ax_pareto_obj.grid(True)
    ax_pareto_real.set_xlabel('Total Cost', fontsize=16); ax_pareto_real.set_ylabel('Total CO2', fontsize=16); ax_pareto_real.legend(fontsize=14); ax_pareto_real.grid(True)
    fig_pareto.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path_pareto = "parent_selection_pareto.png"
    fig_pareto.savefig(save_path_pareto)
    print(f"Pareto front comparison plot saved to: {save_path_pareto}")
    
    plt.show()

if __name__ == '__main__':
    plot_comparison_from_csv()
