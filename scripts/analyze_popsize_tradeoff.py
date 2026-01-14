import pandas as pd
import matplotlib.pyplot as plt
import os

# 파일 경로
DATA_PATH = os.path.join("Results_Param_Optimization", "Step5_PopSize_HV_History.csv")
OUTPUT_IMG_PATH = os.path.join("Results_Param_Optimization", "Step5_PopSize_Tradeoff.png")

def analyze_tradeoff():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 데이터 로드
    df = pd.read_csv(DATA_PATH)
    
    # PopSize 별 최종 Hypervolume 추출
    # 각 PopSize 그룹에서 'gen'이 가장 큰 행을 찾음
    final_stats = df.loc[df.groupby("PopSize")["gen"].idxmax()].sort_values("PopSize")
    
    pops = final_stats["PopSize"].values
    hvs = final_stats["hypervolume"].values
    
    print(f"{ 'PopSize':<10} | { 'Final HV':<15} | { 'Delta HV':<15} | { 'Improvement (%)':<15}")
    print("-" * 65)
    
    # 초기값 출력
    print(f"{pops[0]:<10} | {hvs[0]:<15.6f} | {'-':<15} | {'-':<15}")
    
    # 변화량 계산 및 출력
    improvements = []
    for i in range(1, len(pops)):
        delta_hv = hvs[i] - hvs[i-1]
        delta_pop = pops[i] - pops[i-1]
        # 단순히 HV 차이만 보는게 아니라, Pop 증가 단위당 효율을 볼 수도 있지만
        # 여기서는 직관적으로 이전 단계 대비 HV가 얼마나(%) 좋아졌는지 봅니다.
        imp_pct = (delta_hv / hvs[i-1]) * 100
        improvements.append(imp_pct)
        
        print(f"{pops[i]:<10} | {hvs[i]:<15.6f} | {delta_hv:<15.6f} | {imp_pct:<15.4f}%")

    # 그래프 그리기
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    
    # HV Curve
    color = 'tab:blue'
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Final Hypervolume', color=color)
    ax1.plot(pops, hvs, marker='o', color=color, label='Hypervolume')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Improvement Bar
    if len(pops) > 1:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Improvement from Previous Step (%)', color=color)
        # 첫 번째 값은 0으로 처리하거나 제외
        ax2.bar(pops[1:], improvements, width=40, color=color, alpha=0.3, label='Improvement %')
        ax2.tick_params(axis='y', labelcolor=color)
        # 0% 선 표시
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.title('Step 5: Population Size vs. Hypervolume Trade-off')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"\n-> Trade-off plot saved to {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    analyze_tradeoff()
