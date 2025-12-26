# 상세 비용 산정 로직 및 근거 (Cost Calculation Logic based on ACI 318-19)

본 문서는 최적화 프레임워크(`src/structural_analysis.py`)에 구현된 상세 비용 및 CO2 산정 로직을 설명합니다. 기존의 단순 단위 길이당 비용 방식을 탈피하고, ACI 318-19의 내진 상세 규정을 엄격히 반영하여 물량을 정밀하게 산출합니다.

## 1. 기본 비용 파라미터 (Unit Costs)

MATLAB 데이터베이스 생성 시와 동일한 단가를 적용하여 일관성을 유지합니다.

| 항목 | 기호 | 단위 | 단가 (KRW) | 비고 |
| :--- | :---: | :--- | :---: | :--- |
| **콘크리트** | $C_{conc}$ | $\text{m}^3$ | 80,000 | $f_{ck}$ 무관 고정 단가 가정 |
| **철근** | $C_{steel}$ | $\text{ton}$ | 1,000,000 | 고강도/일반강도 통합 |
| **거푸집** | $C_{form}$ | $\text{m}^2$ | 25,000 | 표면적 기준 |

---

## 2. 부재별 물량 산정 상세 로직

비용은 데이터베이스의 사전 계산된 `Cost` 값을 사용하지 않고, 선택된 단면의 치수와 배근 정보를 바탕으로 **직접 재계산(Re-calculation)**합니다.

### 2.1 콘크리트 및 주철근 (Concrete & Main Rebar)
기둥과 보 모두 전체 부재 길이에 대해 균일하게 적용합니다.

*   **콘크리트 부피 ($V_{conc}$):** $(b \times h \times L) - V_{main\_steel}$
*   **주철근 중량 ($W_{main}$):** $(b \times h \times \rho_{total}) \times L \times \rho_{steel}$
    *   $\rho_{total}$: 인장($\rho$) + 압축($\rho'$) 철근비 합계

### 2.2 전단철근 (Stirrups / Ties) - 구역별 차등 배근
ACI 318-19의 특수/중간 모멘트 골조(SMRF/IMRF) 규정에 따라 소성 힌지 구역과 중앙부를 구분합니다.

#### A. 구역 정의 (ACI 18.6.4, 18.7.5)
*   **소성 힌지 길이 ($L_{hinge}$):** 부재 양단부
    $$L_{hinge} = \max(h, b, L_{clear}/6, 450\text{mm})$$
*   **중앙부 길이 ($L_{mid}$):**
    $$L_{mid} = L_{total} - 2 \times L_{hinge}$$

#### B. 배근 간격 ($s$)
*   **양단부 ($s_{end}$):** 데이터베이스에 정의된 촘촘한 간격 사용 (내진 상세 만족).
    *   규정: $\min(d/4, 6d_b, 150\text{mm})$ (ACI 18.6.4.4)
*   **중앙부 ($s_{mid}$):** 규정이 허용하는 범위 내에서 완화.
    *   적용: $s_{mid} = \min(2 \times s_{end}, d/2)$
    *   **효과:** 중앙부 철근량이 양단부 대비 약 50% 감소하여 **비용 절감 현실화**.

#### C. 전체 개수 ($N_{stirrup}$)
$$N_{stirrup} = \lceil \frac{2 L_{hinge}}{s_{end}} \rceil + \lceil \frac{L_{mid}}{s_{mid}} \rceil$$

### 2.3 띠철근 상세 및 갈고리 (Tie Details & Hooks)

#### A. 135도 내진 갈고리 (Seismic Hooks, ACI 25.7.2.1)
모든 띠철근의 양 끝단에는 135도 갈고리가 포함되어야 합니다. 기존의 단순 여장(20cm) 대신 규정에 맞게 계산합니다.
*   **갈고리 길이 ($L_{hook}$):**
    $$L_{hook} = \max(6 d_{tie}, 75\text{mm})$$
*   **띠철근 1개 길이:** $2(b_{core} + h_{core}) + 2 L_{hook}$

#### B. 보조 띠철근 (Crossties / Supplemental Ties, ACI 25.7.2.3)
기둥 주철근이 코너 외에 배치될 때, 수평 지지를 위해 보조 띠철근이 추가됩니다.
*   **규정:** 주철근 간 순간격이 150mm를 초과하면 해당 철근을 보조 띠철근으로 구속해야 함.
*   **로직:**
    1.  각 면의 주철근 개수(`side_rebars`)와 간격을 확인.
    2.  간격 > 150mm 인 경우, 하나 건너 하나씩(Alternating bars) 크로스타이 추가.
    3.  추가된 크로스타이 길이($b$ 또는 $h$)와 갈고리($2 L_{hook}$)를 물량에 합산.

### 2.4 표피 철근 (Skin Reinforcement, ACI 9.7.2.3)
보의 깊이가 깊을 경우 복부 균열 제어를 위해 배치합니다.
*   **조건:** 보 깊이 $h > 900\text{mm}$
*   **배근:** 양 측면에 수직 간격 300mm 이내로 종방향 철근(D10 등) 배치.
*   **비용 추가:** 조건 만족 시 해당 표피 철근의 중량을 주철근 중량에 합산.

---

## 3. 총 비용 산출식 (Total Cost Formula)

$$Cost_{total} = \sum_{elements} (C_{conc} V_{net} + C_{steel} (W_{main} + W_{ties} + W_{skin}) + C_{form} A_{surf})$$

이 로직은 구조물의 **안전성(Safety)**을 담보하는 동시에, 불필요한 물량을 제거하여 **경제성(Economy)**을 현실적으로 평가할 수 있도록 설계되었습니다.
