import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_comparison_from_csv():
    """
    Generates comparison plots for crossover strategies
    and a bar chart for final solution values
    by reading data from previously saved CSV files.
    """
    # --- 경로 수정 ---
    # 기존 코드는 한 단계 상위 폴더에서 결과 폴더를 찾으려 했습니다.
    # 결과 생성 스크립트와 플로팅 스크립트가 같은 폴더에 있으므로,
    # 결과 폴더도 같은 위치에 있다고 가정하고 스크립트의 현재 위치를 기본 폴더로 설정합니다.
    base_output_folder = os.path.abspath(os.path.dirname(__file__))
    
    # List of experiment methods to plot. This should match the folder names.
    experiment_ids_to_test = [
        "01", 
        "04", 
        "05"
    ]

    # Style mapping for plots
    style_map = {
        '01':  {'color': 'C0', 'marker': 'o', 'linestyle': '-', 'label': 'Experiment 1'},
        '04': {'color': 'C1', 'marker': 's', 'linestyle': '--', 'label': 'Experiment 2'},
        '05': {'color': 'C2', 'marker': 'P', 'linestyle': ':', 'label': 'Experiment 3'}
    }

    # --- Plotting Setup ---
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    # fig.suptitle('Hall of Fame Convergence Comparison', fontsize=20)
    
    ax_conv1, ax_conv2 = axes

    # --- Data storage for bar chart ---
    bar_chart_data = []
    experiment_labels = []

    # --- Load Actual Values Data Once ---
    # The actual values file is in the base folder, not in each experiment folder.
    # Load it once before the loop.
    df_actual_all = None
    actual_values_csv_path = os.path.join(base_output_folder, "../data/실제값.csv")
    if os.path.exists(actual_values_csv_path):
        try:
            # 1. Load the CSV with the first column as the index
            df_transposed = pd.read_csv(actual_values_csv_path, index_col=0)
            # 2. Transpose the DataFrame to have experiments as rows
            df_actual_all = df_transposed.transpose()
            # 3. Clean up the data: remove commas and convert to numeric
            for col in ['cost', 'co2']:
                if col in df_actual_all.columns:
                    df_actual_all[col] = df_actual_all[col].astype(str).str.replace(',', '').astype(float)
            # 4. Create a 'method' column to match with experiment IDs
            # Assuming 'Experiment 1' -> '01', 'Experiment 2' -> '04', etc.
            # This mapping needs to be correct.
            exp_map = {'Experiment 1': '01', 'Experiment 2': '04', 'Experiment 3': '05'}
            df_actual_all['method'] = df_actual_all.index.map(exp_map)

            print(f"Successfully loaded and processed actual values from: {actual_values_csv_path}")
        except Exception as e:
            print(f"  [Error] Could not read or process {actual_values_csv_path}: {e}")

    # --- Loop through methods, load data, and plot ---
    for method in experiment_ids_to_test:
        print(f"Processing method: {method}")
        
        # Construct path to the result folder for the current method
        method_folder = os.path.join(base_output_folder, f"../results/Results_main_하중증가,모집단600_{method}")
        
        if not os.path.isdir(method_folder):
            print(f"  [Warning] Directory not found, skipping: {method_folder}")
            continue

        style = style_map.get(method, {'color': 'gray', 'marker': 'x', 'linestyle': '-.'})
        plot_label = style.get('label', method)

        # 1. Plot Convergence Data from HOF CSV
        hof_csv_path = os.path.join(method_folder, "hof_convergence.csv")
        if os.path.exists(hof_csv_path):
            try:
                df_hof = pd.read_csv(hof_csv_path)
                # 100세대를 초과하는 데이터를 필터링합니다.
                df_hof = df_hof[df_hof['gen'] <= 100]
                gen = df_hof['gen']
                best_obj1 = df_hof['best_obj1']
                best_obj2 = df_hof['best_obj2']
                
                ax_conv1.plot(gen, best_obj1, label=plot_label, alpha=0.9, color=style['color'], 
                              marker=style['marker'], linestyle=style['linestyle'], markersize=4)
                ax_conv2.plot(gen, best_obj2, label=plot_label, alpha=0.9, color=style['color'], 
                              marker=style['marker'], linestyle=style['linestyle'], markersize=4)
                print(f"  - Plotted convergence data from {hof_csv_path}")
            except Exception as e:
                print(f"  [Error] Could not read or plot {hof_csv_path}: {e}")
        else:
            print(f"  [Warning] Convergence CSV not found: {hof_csv_path}")

        # 2. Get Actual Values for the current method for Bar Chart
        if df_actual_all is not None and 'method' in df_actual_all.columns:
            try:
                # Find the row corresponding to the current method
                actual_row = df_actual_all[df_actual_all['method'] == method]
                if not actual_row.empty:
                    cost = actual_row['cost'].iloc[0]
                    co2 = actual_row['co2'].iloc[0]
                    bar_chart_data.append({'method': method, 'cost': cost, 'co2': co2})
                    experiment_labels.append(plot_label)
                    print(f"  - Found actual values for method '{method}'")
                else:
                    print(f"  [Warning] No actual values found for method '{method}' in 실제값.csv")
            except KeyError as e:
                 print(f"  [Error] Missing column in 실제값.csv: {e}. Required columns are 'method', 'cost', 'co2'.")
        else:
            print(f"  [Warning] Skipping actual values bar chart data. File '실제값.csv' not found or is missing 'method' column.")

    # --- Finalize and Save Plot ---
    # ax_conv1.set_title('HOF Convergence - Objective 1 (Cost+CO2)')
    ax_conv1.set_xlabel('Generation', fontsize=16)
    ax_conv1.set_ylabel('Best Fitness 1 in HOF', fontsize=16)
    if ax_conv1.get_legend_handles_labels()[0]: # Only show legend if there are items to display
        ax_conv1.legend(fontsize=14)
    ax_conv1.grid(True)
    
    # ax_conv2.set_title('HOF Convergence - Objective 2 (Mean DCR)')
    ax_conv2.set_xlabel('Generation', fontsize=16)
    ax_conv2.set_ylabel('Best Fitness 2 in HOF', fontsize=16)
    if ax_conv2.get_legend_handles_labels()[0]: # Only show legend if there are items to display
        ax_conv2.legend(fontsize=14)
    ax_conv2.grid(True)
    
    # X축(세대) 범위를 0-100으로 설정하고 좌우에 여백을 추가합니다.
    ax_conv1.set_xlim(-5, 105)
    ax_conv1.set_xticks(range(0, 101, 20)) # x축 눈금을 20단위로 설정하여 가독성 향상
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 저장 경로를 스크립트가 있는 폴더로 변경
    save_path = os.path.join(base_output_folder, "../results/comparison_summary_hof_from_csv.png")
    plt.savefig(save_path)
    print(f"\nComparison summary plot saved to: {save_path}")
    plt.show()

    # --- Create and Save Bar Chart for Actual Values ---
    if bar_chart_data:
        print("\nGenerating bar chart for final actual values...")
        
        methods = [item['method'] for item in bar_chart_data]
        costs = [item['cost'] for item in bar_chart_data]
        co2s = [item['co2'] for item in bar_chart_data]
        labels = [style_map[m]['label'] for m in methods]

        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars

        fig_bar, ax1_bar = plt.subplots(figsize=(12, 8))
        # fig_bar.suptitle('Final Solution - Actual Values Comparison', fontsize=16)

        # 천 단위 쉼표 포매터 생성
        formatter = FuncFormatter(lambda x, pos: f'{int(x):,}')

        # Cost와 CO2를 위한 색상 정의
        cost_color = 'C0'  # 파란색 계열
        co2_color = 'C3'   # 붉은색 계열

        # Cost 막대그래프 (왼쪽 Y축)
        bar1 = ax1_bar.bar(x - width/2, costs, width, label='Cost', color=cost_color, edgecolor='black', zorder=3)
        ax1_bar.set_ylabel('Cost (won)', color=cost_color, fontsize=16)
        ax1_bar.yaxis.set_major_formatter(formatter)
        ax1_bar.tick_params(axis='y', labelcolor=cost_color)
        # ax1_bar.set_xlabel('Experiment')
        ax1_bar.set_xticks(x)
        ax1_bar.set_xticklabels(labels)
        ax1_bar.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        # CO2 막대그래프 (오른쪽 Y축)
        ax2_bar = ax1_bar.twinx()
        bar2 = ax2_bar.bar(x + width/2, co2s, width, label='CO2', color=co2_color, edgecolor='black', zorder=3)
        ax2_bar.set_ylabel(r'CO$_2$ (kgCO$_2$e)', color=co2_color, fontsize=16)
        ax2_bar.yaxis.set_major_formatter(formatter)
        ax2_bar.tick_params(axis='y', labelcolor=co2_color)

        # Adding bar labels
        # 천 단위 쉼표를 포함하여 정수로 표시
        ax1_bar.bar_label(bar1, labels=[f'{c:,.0f}' for c in costs], padding=3, fontsize=14)
        ax2_bar.bar_label(bar2, labels=[f'{c:,.0f}' for c in co2s], padding=3, fontsize=14)

        fig_bar.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the bar chart to a separate file
        bar_chart_save_path = os.path.join(base_output_folder, "../results/comparison_actual_values_bar_chart.png")
        plt.savefig(bar_chart_save_path)
        print(f"Bar chart saved to: {bar_chart_save_path}")
        plt.show()
    else:
        print("\nNo data found for actual values bar chart. Skipping plot generation.")


if __name__ == '__main__':
    plot_comparison_from_csv()
