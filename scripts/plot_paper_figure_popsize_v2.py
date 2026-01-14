import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

# ----------------------------------------------------------- 
# [설정] 논문용 스타일 및 데이터 경로
# ----------------------------------------------------------- 
DATA_PATH = os.path.join("Results_Param_Optimization", "Step5_PopSize_HV_History.csv")
OUTPUT_PATH = os.path.join("Results_Param_Optimization", "Paper_Figure_PopSize_Selection_v2.png")

# 논문 폰트 설정
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def plot_paper_figure_v2():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # 1. 데이터 로드 및 가공
    df = pd.read_csv(DATA_PATH)
    
    final_stats = df.loc[df.groupby("PopSize")["gen"].idxmax()].sort_values("PopSize")
    
    # [수정] 100 ~ 600 구간만 필터링
    mask = (final_stats["PopSize"] >= 100) & (final_stats["PopSize"] <= 600)
    subset = final_stats[mask]
    
    pops = subset["PopSize"].values
    hvs = subset["hypervolume"].values
    
    # 정규화된 비용 (600을 100%로 기준 잡거나, 전체 1000 기준 유지 가능)
    # 여기서는 그래프 내에서의 상대적 비용을 보여주기 위해 600을 1.0으로 잡음
    normalized_cost = pops / 600.0

    # 2. 그래프 그리기 (Dual Axis) 
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()

    # --- 왼쪽 Y축: Performance (Hypervolume) ---
    color_hv = '#1f77b4'  # Blue
    ax1.set_xlabel('Population Size', fontweight='bold')
    ax1.set_ylabel('Hypervolume (Performance)', color=color_hv, fontweight='bold')
    
    # HV Plot
    line1 = ax1.plot(pops, hvs, marker='o', color=color_hv, linewidth=2, label='Hypervolume', zorder=10)
    ax1.tick_params(axis='y', labelcolor=color_hv)
    
    # Y축 범위 미세 조정 (데이터에 맞춰서)
    y_min = hvs.min() - 0.02
    y_max = hvs.max() + 0.02
    ax1.set_ylim(bottom=y_min, top=y_max)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 오른쪽 Y축: Cost (Computational Load) ---
    ax2 = ax1.twinx()
    color_cost = '#d62728'  # Red
    ax2.set_ylabel('Relative Computational Cost', color=color_cost, fontweight='bold')
    
    # Cost Plot (Linear)
    line2 = ax2.plot(pops, normalized_cost, linestyle='--', marker='s', color=color_cost, alpha=0.6, label='Computational Cost')
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.set_ylim(bottom=0, top=1.1)

    # --- Highlight Selected Point (Pop 400) ---
    target_pop = 400
    if target_pop in pops:
        target_idx = list(pops).index(target_pop)
        target_hv = hvs[target_idx]
        
        # HV 지점에 강조 표시
        ax1.scatter([target_pop], [target_hv], s=180, facecolors='white', edgecolors='red', linewidth=2.5, zorder=20, marker='*')
        
        # 화살표 및 텍스트 주석
        ax1.annotate(f'Optimal Point\n(Pop={target_pop})\nHigh Performance/Cost',
                     xy=(target_pop, target_hv), xycoords='data',
                     xytext=(target_pop, target_hv - 0.05), textcoords='data',
                     arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                     ha='center', fontsize=11)

    # 범례 합치기
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True, framealpha=0.9)

    plt.title('Parameter Sensitivity: Population Size (100-600)', fontweight='bold', pad=15)
    plt.tight_layout()
    
    # 저장
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_paper_figure_v2()
