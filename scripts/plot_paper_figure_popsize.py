import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

# ----------------------------------------------------------- 
# [설정] 논문용 스타일 및 데이터 경로
# ----------------------------------------------------------- 
DATA_PATH = os.path.join("Results_Param_Optimization", "Step5_PopSize_HV_History.csv")
OUTPUT_PATH = os.path.join("Results_Param_Optimization", "Paper_Figure_PopSize_Selection.png")

# 논문 폰트 설정 (시스템에 Times New Roman이 없으면 기본값 사용)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def plot_paper_figure():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # 1. 데이터 로드 및 가공
    df = pd.read_csv(DATA_PATH)
    
    # 각 PopSize 별 최종 세대의 Hypervolume 추출
    final_stats = df.loc[df.groupby("PopSize")["gen"].idxmax()].sort_values("PopSize")
    
    pops = final_stats["PopSize"].values
    hvs = final_stats["hypervolume"].values
    
    # 이론적 계산 비용 (Population Size에 선형 비례한다고 가정)
    # 최대 Pop(1000)을 1.0(100%)로 정규화
    normalized_cost = pops / pops.max()

    # 2. 그래프 그리기 (Dual Axis) 
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()

    # --- 왼쪽 Y축: Performance (Hypervolume) ---
    color_hv = '#1f77b4'  # Blue
    ax1.set_xlabel('Population Size', fontweight='bold')
    ax1.set_ylabel('Hypervolume (Performance)', color=color_hv, fontweight='bold')
    # HV Plot
    line1 = ax1.plot(pops, hvs, marker='o', color=color_hv, linewidth=2, label='Hypervolume', zorder=10)
    ax1.tick_params(axis='y', labelcolor=color_hv)
    ax1.set_ylim(bottom=5.55, top=5.75)  # HV 변화가 잘 보이도록 범위 조정 (필요시 수정)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 오른쪽 Y축: Cost (Computational Load) ---
    ax2 = ax1.twinx()
    color_cost = '#d62728'  # Red
    ax2.set_ylabel('Normalized Computational Cost', color=color_cost, fontweight='bold')
    # Cost Plot (Linear)
    line2 = ax2.plot(pops, normalized_cost, linestyle='--', marker='s', color=color_cost, alpha=0.6, label='Computational Cost')
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.set_ylim(bottom=0, top=1.1)

    # --- Highlight Selected Point (Pop 300) ---
    target_pop = 300
    target_idx = list(pops).index(target_pop)
    target_hv = hvs[target_idx]
    target_cost = normalized_cost[target_idx]

    # HV 지점에 강조 표시
    ax1.scatter([target_pop], [target_hv], s=150, facecolors='white', edgecolors=color_hv, linewidth=2, zorder=20)
    
    # 화살표 및 텍스트 주석
    ax1.annotate(f'Selected Point\n(Pop={target_pop})\nGood Trade-off',
                 xy=(target_pop, target_hv), xycoords='data',
                 xytext=(target_pop + 150, target_hv - 0.05), textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black', lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                 ha='center', fontsize=11)

    # 범례 합치기
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9)

    plt.title('Performance vs. Cost Trade-off Analysis', fontweight='bold', pad=15)
    plt.tight_layout()
    
    # 저장
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_paper_figure()
