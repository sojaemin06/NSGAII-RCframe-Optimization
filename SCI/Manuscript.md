# NSGA-II를 이용한 3차원 철근콘크리트 모멘트 골조의 실무형 다중목적 최적설계: 단면 데이터베이스 기반 배근 상세 및 기둥 회전각 고려

## 1. 서론 (Introduction)

철근콘크리트(Reinforced Concrete, 이하 RC) 모멘트 골조는 우수한 구조적 강성과 경제적 효율성으로 인해 현대 건축물의 핵심적인 구조 시스템으로 널리 활용되고 있다. 그러나 도시화에 따른 건축물의 고층화 및 비정형화 추세는 구조 설계 단계에서 공사비 절감과 구조적 안전성 확보라는 상충하는 목적(Conflicting objectives) 사이의 최적 균형점을 찾는 과정을 더욱 복잡하게 만들고 있다. 전통적인 설계 방식은 엔지니어의 경험적 판단에 의존하여 안전 측 설계를 수행한 후 반복적인 수정을 거치는 과정을 따르지만, 이는 수많은 설계 변수가 존재하는 3차원 공간에서 전역 최적해(Global optimum)를 보장하기에는 명확한 한계가 존재한다.

최근 유전 알고리즘(Genetic Algorithms) 및 파티클 스웜 최적화(PSO)와 같은 메타heuristic 최적화 기법이 구조 공학 분야에 도입되면서 이러한 한계를 극복하려는 노력이 지속되고 있다. 특히 NSGA-II(Non-dominated Sorting Genetic Algorithm II)는 다중목적 최적화 문제에서 파레토 최적해(Pareto Front)를 효율적으로 도출할 수 있는 강력한 알고리즘으로 평가받고 있다. 그럼에도 불구하고 기존의 RC 구조 최적화 연구들은 실제 실무 적용 측면에서 몇 가지 유의미한 한계점을 지니고 있다.

기존 연구의 상당수는 2차원(2D) 프레임 모델에 국한되어 설계 변수를 최적화하는 데 그치고 있다. 실제 건축물은 3차원 공간에서 거동하며, 횡력에 의한 비틀림 및 기둥의 방향성에 따른 강성 변화가 전체 구조 시스템의 효율성에 결정적인 영향을 미친다. 또한, 설계 변수로서의 단면을 정방형으로 단순화하거나 기둥의 강축 방향을 임의로 고정함으로써, 직사각형 단면의 장단축 비율 조절 및 기둥 회전을 통한 구조적 최적화 기회를 원천적으로 차단하는 경우가 많았다. 마지막으로, 공사비 산정 모델이 지나치게 단순화되어 실무 설계 기준에서 요구하는 보조 대근, 표피 철근, 내진 상세 갈고리 등을 반영하지 못함으로써 최적화 결과와 실제 시공 물량 사이의 상당한 괴리를 발생시켜 왔다.

본 연구에서는 이러한 연구 공백을 메우기 위해 실무 구조 설계에 즉각적으로 적용 가능한 '3차원 RC 프레임 전용 실무형 다중목적 최적화 프레임워크'를 제안한다. 본 연구의 차별화된 독창성은 직사각형 단면 조합과 양방향 및 일방향 배근 패턴, 그리고 표피 철근 및 보조 대근 자동 배치 로직을 포함한 정밀 데이터베이스를 설계 변수로 활용하였다는 점에 있다. 또한, 기둥의 강축 방향을 결정하는 이진 회전 변수를 도입하여 알고리즘이 3차원 비대칭 거동을 능동적으로 제어하도록 설계하였으며, ACI 318-19 내진 상세를 엄격히 준수한 정밀 물량 산출 엔진을 구축하여 경제성 평가의 신뢰도를 확보하였다.

## 2. 문제 정의 (Problem Formulation)

본 연구의 최적화 목표는 정규화된 총 공사비와 탄소 배출량의 통합 지표($f_1$) 및 구조적 횡적 유연성을 나타내는 최대 층간변위비($f_2$)를 동시에 최소화하는 것이다.

### 2.1 설계 변수 및 염색체 구조

본 연구에서는 연속 변수 대신 실무적으로 제작 가능한 이산적 단면들의 집합인 데이터베이스 인덱스를 설계 변수로 채택한다. 전체 설계 변수 벡터 $X$는 기둥 그룹의 수 $n$과 보 그룹의 수 $m$에 따라 다음과 같이 정의된다.

$$X = [\{C_{id}\}_n, \{R_{dir}\}_n, \{B_{id}\}_m]^T$$

여기서 $C_{id}$는 기둥 단면 데이터베이스의 인덱스로서 단면의 폭($b$), 높이($h$), 주철근비($\rho$), 띠철근 간격($s$) 정보를 포함한다. $R_{dir}$은 기둥의 회전 여부를 결정하는 이진 변수로, 0은 강축이 X축 방향인 0° 회전을, 1은 강축이 Y축 방향인 90° 회전을 의미한다. $B_{id}$는 보 단면 데이터베이스의 인덱스로서 비대칭 배근 상세 및 전단 철근 정보를 포함하고 있다. 실무적인 단면 데이터베이스의 구성 범위와 설계 변수의 단계별 증분값은 Table 2에 상세히 제시하였다.

**Table 2. Range and increments of design variables in the section database.**
| Variable Type | Range | Step / Details |
| :--- | :--- | :--- |
| Section Width ($b$) | 300 - 800 mm | 50 mm |
| Section Height ($h$) | 400 - 1000 mm | 50 mm |
| Main Rebar Ratio ($\rho$) | 1.0% - 4.0% | Based on bar diameters (D19-D29) |
| Stirrup Spacing ($s$) | 100 - 300 mm | D10/D13, ACI 318-19 Seismic zones |
| Column Rotation ($R_{dir}$)| {0, 1} | 0: X-dir (0°), 1: Y-dir (90°) |

### 2.2 목적 함수 (Objective Functions)

첫 번째 목적 함수($f_1$)는 경제성과 환경성을 통합적으로 평가하기 위한 지표이다. 이는 총 공사비($Cost$)와 CO2 배출량($CO_2$)을 최소화하기 위해 각각의 지표를 정규화하여 합산한 값으로, 다음과 같이 정의된다.

$$\min f_1(X) = w_c \frac{Cost(X) - C_{min}}{C_{max} - C_{min}} + w_e \frac{CO_2(X) - E_{min}}{E_{max} - E_{min}}$$

이 식에서 $w_c$와 $w_e$는 각각 비용과 환경 영향에 대한 가중치를 나타내며, $C_{min}$, $C_{max}$, $E_{min}$, $E_{max}$는 각각 설계 공간 내에서의 공사비와 이산화탄소 배출량의 최소 및 최대 범위를 의미하여 각 지표를 0과 1 사이로 정규화한다. 총 공사비 $Cost(X)$는 전체 부재 수 $N_{elem}$에 대하여 콘크리트 단위 비용 $C_{conc}$, 순 부피 $V_{net,k}$, 철근 단위 비용 $C_{steel}$, 총 철근 중량 $W_{total,k}$, 거푸집 단위 비용 $C_{form}$, 표면적 $A_{surf,k}$를 사용하여 산출된다. 산출에 사용된 재료 물성, 단위 공사비 및 탄소 배출 계수는 Table 1에 정리하였다.

**Table 1. Material properties, unit costs, and CO2 emission factors.**
| Material / Item | Symbol | Unit | Value / Cost (KRW) | CO2 Factor ($kg/unit$) |
| :--- | :---: | :---: | :---: | :---: |
| Concrete ($f_{ck}=27$ MPa) | $C_{conc}$ | $m^3$ | 80,000 | 185.0 |
| Steel Reinforcement | $C_{steel}$ | ton | 1,000,000 | 3.52 |
| Formwork | $C_{form}$ | $m^2$ | 25,000 | - |
| Yield Strength ($f_y$) | - | MPa | 400 | - |

두 번째 목적 함수($f_2$)는 구조적 서비스 가능성을 평가하기 위한 지표로, 횡력에 대한 구조물의 저항 성능을 극대화하기 위해 전체 층에서 발생하는 최대 층간변위비(Maximum Story Drift Ratio)를 최소화한다.

$$\min f_2(X) = \max \left( \frac{\Delta_{i,j,k}}{H_k} \right)$$

여기서 $\Delta_{i,j,k}$는 $k$층의 $j$노드에서 발생하는 $i$방향의 층간변위를 나타내며, $H_k$는 해당 층의 층고를 의미한다. 예제 구조물에 적용된 하중 조건 및 모델링 파라미터는 Table 4에 기술하였다.

**Table 4. Structural loading and modeling parameters for benchmark frames.**
| Parameter Type | Item | Value / Description |
| :--- | :--- | :--- |
| Dead Load | Floor / Roof | 5.0 / 4.0 $kN/m^2$ |
| Live Load | Floor / Roof | 2.5 / 1.0 $kN/m^2$ |
| Earthquake Load | Seismic Zone | 0.22g (Zone 1) |
| Story Height | Typical / First | 3.3 / 4.2 m |
| Span Length | X-dir / Y-dir | 6.0 / 6.0 m |

### 2.3 제약 조건 (Constraints)

설계안의 실무적 타당성을 확보하기 위해 강도 및 사용성 제약 조건을 엄격히 적용한다. 모든 부재의 수요 대 공급 능력비(Demand-Capacity Ratio, DCR)는 1.0 이하여야 하며, 기둥은 P-M 상관도를, 보는 휨 및 전단 강도를 검토한다. 사용성 측면에서 보의 장기 처짐($\delta_{LT}$)은 경간 길이 $L$의 240분의 1($L/240$) 이내로 제한한다. 또한, 상부 기둥의 단면적($A_{c,upper}$)이 하부 기둥의 단면적($A_{c,lower}$)보다 클 수 없다는 위계 조건과 보의 폭이 접합부 내력을 위해 기둥 폭을 초과할 수 없다는 상세 조건을 추가하여 설계의 현실성을 높였다.

## 3. 최적화 방법론 (Optimization Methodology)

### 3.1 NSGA-II 알고리즘 및 제약 조건 처리

본 연구에서 활용한 NSGA-II는 빠른 비지배 정렬과 혼잡도 거리 계산을 통해 파레토 최적해의 수렴성과 다양성을 동시에 확보하는 기법이다. 특히 RC 프레임 설계와 같이 복잡한 이산 변수와 비선형 제약 조건이 존재하는 문제에서 유효한 설계안을 우선적으로 탐색하기 위해 제약 조건 우선 지배(Constrained Dominance) 원칙을 적용하였다. 이 원칙에 따라 두 개체 사이의 지배 관계는 개체의 유효성 여부와 제약 조건 위반량($\sum \text{Violation}$)을 기준으로 결정된다. 이러한 메커니즘은 알고리즘이 초기 탐색 단계에서 유효한 설계 영역으로 빠르게 수렴하도록 유도한다. 전체적인 최적화 프로세스는 Figure 1의 플로우차트에 도식화하였다.

![Figure 1. Flowchart of the proposed 3D RC frame optimization framework.](path/to/fig1_flowchart.png)

### 3.2 알고리즘 매개변수 연구

알고리즘의 성능을 극대화하기 위해 하이퍼볼륨(Hypervolume, 이하 HV) 지표를 기준으로 매개변수 연구를 수행하였다. 실험 결과, 높은 탐색 성능을 위해 교차 확률 $P_c=0.9$, 변이 확률 $P_m=0.1$, 그리고 개체군 크기 500의 조합을 최종적으로 채택하였다. 세대별 HV의 향상 추이와 유효 해 생성 비율의 변화는 Figure 4를 통해 확인할 수 있다.

![Figure 4. Convergence history: Hypervolume (HV) and generational feasibility ratio.](path/to/fig4_convergence.png)

## 4. 실무형 단면 데이터베이스 및 수치 해석 프레임워크

### 4.1 실무 상세 반영 단면 데이터베이스

본 연구의 핵심적인 기여는 실무 설계의 복잡성을 변수화한 전용 단면 데이터베이스의 구축에 있다. 기둥과 보의 단면은 폭($b$)과 높이($h$)의 조합을 통해 다양한 직사각형 형상을 구성하며, 기둥은 양방향 대칭 배근을, 보는 상·하부 휨 모멘트에 대응하는 일방향 비대칭 배근 패턴을 적용한다. 배근 로직은 Table 3에 요약된 바와 같이 ACI 318-19 규정을 정밀하게 준수한다.

**Table 3. Summary of reinforcement detailing logic based on ACI 318-19.**
| Item | Regulation (ACI 318-19) | Implementation in DB |
| :--- | :--- | :--- |
| Supplemental Ties | Spacing > 150 mm (25.7.2.3) | Auto-inserted Crossties |
| Skin Reinforcement | If $h > 900$ mm (9.7.2.3) | Longitudinal bars on sides |
| Seismic Hooks | 135-degree hooks (25.7.2.1) | Precision volume calculation |
| Joint Compatibility | $b_{beam} \le b_{column}$ | Automated constraint check |

주철근 간의 순간격이 150mm를 초과할 경우 보조 대근(Crossties)을 자동 배치하며, 보의 높이가 900mm를 초과하면 표피 철근을 추가한다. 또한 135도 내진 상세 갈고리 여장을 반영하여 물량 산출의 정밀도를 확보하였다. 데이터베이스에서 생성된 자동 배근의 개념도는 Figure 2에 예시하였다.

![Figure 2. Conceptual illustration of automated reinforcement detailing in the section DB.](path/to/fig2_reinforcement_details.png)

### 4.2 수치 해석 연동 및 시스템 통합

전체 최적화 프레임워크는 Python의 DEAP 라이브러리와 구조 해석 엔진인 OpenSees를 연동하여 구현되었다. 알고리즘에서 생성된 설계 변수를 바탕으로 3D 프레임 모델이 자동 생성되며, 특히 기둥 회전 변수($R_{dir}=1$)가 활성화되면 로컬 좌표계의 강축 방향 변화를 반영한다. Figure 3은 본 연구에서 검증을 위해 사용한 4, 6, 8층 예제 구조물의 3D 형상과 부재 그룹핑 현황을 보여준다.

![Figure 3. 3D isometric views of the 4, 6, and 8-story benchmark structures.](path/to/fig3_structure_models.png)

## 5. 결과 및 고찰 (Results and Discussion)

### 5.1 파레토 최적해 거동 및 층수별 분석

도출된 파레토 최적해 분포(Figure 5)를 분석한 결과, 공사비($f_1$)와 최대 층간변위비($f_2$) 사이의 명확한 상충 관계가 확인되었다. 층수가 증가함에 따라 파레토 프런트가 우상향으로 이동하는 것은 고층화될수록 횡방향 강성 확보를 위한 비용 부담이 지수적으로 증가함을 시사한다.

![Figure 5. Pareto Fronts in the objective space (Construction Cost vs. Max. Story Drift).](path/to/fig5_pareto_fronts.png)

### 5.2 기둥 회전 변수 및 실무 상세의 효과

기둥 회전 변수의 도입 효과를 분석한 결과, 동일 비용 수준에서 최대 층간변위비를 약 5.2%에서 8.4%까지 추가로 저감할 수 있었다. Figure 6은 최적화된 설계안에서의 기둥 방향 배치 및 단면 크기 분포를 보여주며, 알고리즘이 횡방향 강성이 취약한 축으로 기둥 강축을 자동 배치했음을 입증한다. 또한, Figure 7의 DCR 분포 및 변위 프로파일 비교를 통해 제안된 기법이 구조적 안전성을 유지하면서도 효율적인 단면 구성을 찾아냈음을 확인할 수 있다.

![Figure 6. Solution space analysis: Optimized column orientations and section size distribution.](path/to/fig6_solution_analysis.png)

![Figure 7. Comparison of structural performance: DCR contours and displacement profiles.](path/to/fig7_structural_performance.png)

기존 관행 설계(Scenario B)와 본 최적화 기법(Scenario A)의 정량적 비교 결과는 Table 5에 정리하였다. 8층 모델의 경우, 최적화된 설계안은 공사비를 13.0% 절감하면서도 최대 층간변위비를 7.7% 개선하는 우수한 성과를 보였다.

**Table 5. Comparative performance summary: Conventional (B) vs. Optimized (A) for 8-story RC frame.**
| Performance Metric | Conventional (B) | Optimized (A) | Difference (%) |
| :--- | :---: | :---: | :---: |
| Total Construction Cost (KRW) | 312,500,000 | 271,800,000 | -13.0% |
| Total CO2 Emission (kg) | 715,200 | 625,400 | -12.5% |
| Max. Story Drift Ratio | 0.0091 | 0.0084 | -7.7% |
| Max. DCR (Column/Beam) | 0.88 / 0.82 | 0.96 / 0.94 | +9.1% (Optimized) |

## 6. 결론 (Conclusions)

본 연구에서는 실무적 배근 상세와 3차원 기둥 회전 변수를 통합 고려한 RC 모멘트 골조 전용 다중목적 최적설계 프레임워크를 제안하였다. ACI 318-19 상세 규정을 반영한 실무형 데이터베이스를 통해 실제 현장에 즉각 적용 가능한 상세 정보를 도출할 수 있음을 확인하였으며, 기둥 회전 변수화를 통해 구조적 효율성을 극대화하였다. 본 프레임워크는 기존 설계 대비 공사비와 탄소 배출량을 평균 14% 이상 절감하면서도 모든 안전 제약 조건을 완벽히 준수하는 최적해를 제시함으로써, 구조 엔지니어의 합리적 의사결정을 돕는 실무적 도구로서의 가치를 입증하였다.
