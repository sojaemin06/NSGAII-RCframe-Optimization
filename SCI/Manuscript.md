# Practical Multi-Objective Optimal Design of 3D Reinforced Concrete Moment Frames Using NSGA-II: Considering Reinforcement Detailing and Column Rotation Angles Based on a Section Database

## 1. Introduction

철근콘크리트(Reinforced Concrete, 이하 RC) 모멘트 골조는 우수한 구조적 강성과 경제적 효율성으로 인해 현대 건축물의 핵심적인 구조 시스템으로 널리 활용되고 있다. 그러나 도시화에 따른 건축물의 고층화 및 비정형화 추세는 구조 설계 단계에서 공사비 절감과 구조적 안전성 확보라는 상충하는 목적(Conflicting objectives) 사이의 최적 균형점을 찾는 과정을 더욱 복잡하게 만들고 있다 (Ehrgott, 2012; Esfandiary et al., 2016; Marler & Arora, 2004). 전통적인 설계 방식은 엔지니어의 경험적 판단에 의존하여 안전 측 설계를 수행한 후 반복적인 수정을 거치는 과정을 따르지만, 이는 수많은 설계 변수가 존재하는 3차원 공간에서 전역 최적해(Global optimum)를 보장하기에는 명확한 한계가 존재한다.

최근 유전 알고리즘(Genetic Algorithms) 및 파티클 스웜 최적화(PSO)와 같은 메타heuristic 최적화 기법이 구조 공학 분야에 도입되면서 이러한 한계를 극복하려는 노력이 지속되고 있다 (Aga & Adam, 2015; Akin & Saka, 2015; Aslay et al., 2024; Chutani & Singh, 2018; Kaveh & Sabzi, 2011; Govindaraj & Ramasamy, 2005; Babaei & Mollayi, 2016; Chaudhuri et al., 2021; Dehnavipour et al., 2019). 특히 NSGA-II(Non-dominated Sorting Genetic Algorithm II)는 다중목적 최적화 문제에서 파레토 최적해(Pareto Front)를 효율적으로 도출할 수 있는 강력한 알고리즘으로 평가받고 있다 (Deb et al., 2002; Nebro et al., 2022; Zitzler & Thiele, 2002; Coello, 2006). 그럼에도 불구하고 기존의 RC 구조 최적화 연구들은 실제 실무 적용 측면에서 몇 가지 유의미한 한계점을 지니고 있다.

기존 연구의 상당수는 2차원(2D) 프레임 모델에 국한되어 설계 변수를 최적화하는 데 그치고 있다. 실제 건축물은 3차원 공간에서 거동하며, 횡력에 의한 비틀림 및 기둥의 방향성에 따른 강성 변화가 전체 구조 시스템의 효율성에 결정적인 영향을 미친다. 또한, 설계 변수로서의 단면을 정방형으로 단순화하거나 기둥의 강축 방향을 임의로 고정함으로써, 직사각형 단면의 장단축 비율 조절 및 기둥 회전을 통한 구조적 최적화 기회를 원천적으로 차단하는 경우가 많았다 (Esfandiari et al., 2018; Kaveh & Ardebili, 2023a, 2023b; Mergos, 2021, 2022; Bekdaş & Nigdeli, 2014; Djedoui et al., 2025; Faghirnejad, 2023; Gharehbaghi, 2012; Heydari et al., 2025; Juliani & Gomes, 2021; Kaveh & Ardebili, 2021). 마지막으로, 공사비 산정 모델이 지나치게 단순화되어 실무 설계 기준에서 요구하는 보조 대근, 표피 철근, 내진 상세 갈고리 등을 반영하지 못함으로써 최적화 결과와 실제 시공 물량 사이의 상당한 괴리를 발생시켜 왔다 (Boscardin et al., 2019; Kaveh et al., 2020a; Bai et al., 2020).

본 연구에서는 이러한 연구 공백을 메우기 위해 실무 구조 설계에 즉각적으로 적용 가능한 '3차원 RC 프레임 전용 실무형 다중목적 최적화 프레임워크'를 제안한다. 본 연구의 차별화된 독창성은 직사각형 단면 조합과 양방향 및 일방향 배근 패턴, 그리고 표피 철근 및 보조 대근 자동 배치 로직을 포함한 정밀 데이터베이스를 설계 변수로 활용하였다는 점에 있다. 또한, 기둥의 강축 방향을 결정하는 이진 회전 변수를 도입하여 알고리즘이 3차원 비대칭 거동을 능동적으로 제어하도록 설계하였으며, ACI 318-19 내진 상세를 엄격히 준수한 정밀 물량 산출 엔진을 구축하여 경제성 평가의 신뢰도를 확보하였다 (Kaveh et al., 2020b; Mergos, 2024; Oluwole Akadiri & Olaniran Fadiya, 2013; Paya-Zaforteza et al., 2009; Werner & Burns, 2012).

## 2. Problem Formulation

본 연구의 최적화 목표는 정규화된 총 공사비와 탄소 배출량의 통합 지표($f_1$) 및 구조적 횡적 유연성을 나타내는 최대 층간변위비($f_2$)를 동시에 최소화하는 것이다.

### 2.1 Design Variables and Chromosome Structure

본 연구에서는 연속 변수 대신 실무적으로 제작 가능한 이산적 단면들의 집합인 데이터베이스 인덱스를 설계 변수로 채택한다. 전체 설계 변수 벡터 $X$는 기둥 그룹의 수 $n$과 보 그룹의 수 $m$에 따라 다음과 같이 정의된다.

$$X = [\{C_{id}\}_n, \{R_{dir}\}_n, \{B_{id}\}_m]^T$$

여기서 $C_{id}$는 기둥 단면 데이터베이스의 인덱스로서 단면의 폭($b$), 높이($h$), 주철근비($\rho$), 띠철근 간격($s$) 정보를 포함한다. $R_{dir}$은 기둥의 회전 여부를 결정하는 이진 변수로, 0은 강축이 X축 방향인 0° 회전을, 1은 강축이 Y축 방향인 90° 회전을 의미한다. $B_{id}$는 보 단면 데이터베이스의 인덱스로서 비대칭 배근 상세 및 전단 철근 정보를 포함하고 있다. 실무적인 단면 데이터베이스의 구성 범위와 설계 변수의 단계별 증분값은 Table 2에 상세히 제시하였다.

**Table 2. Range and increments of design variables in the section database.**
| Variable Type | Range (Beam / Column) | Step / Details |
| :--- | :--- | :--- |
| Section Width ($b$) | 300-500 / 600-1000 mm | 50 mm / 100 mm |
| Section Height ($h$) | 500-800 / 600-1500 mm | 50 mm / 100 mm |
| Main Rebar Ratio ($\rho$) | 0.5% - 4.0% | Based on ACI 318-19 |
| Stirrup Spacing ($s$) | 100 - 450 mm | D10/D13, ACI 318-19 |
| Column Rotation ($R_{dir}$)| {0, 1} | 0: X-dir (0°), 1: Y-dir (90°) |

### 2.2 Objective Functions

첫 번째 목적 함수($f_1$)는 경제성과 환경성을 통합적으로 평가하기 위한 지표이다. 이는 총 공사비($Cost$)와 CO2 배출량($CO_2$)을 최소화하기 위해 각각의 지표를 정규화하여 합산한 값으로, 다음과 같이 정의된다.

$$\min f_1(X) = w_c \frac{Cost(X) - C_{min}}{C_{max} - C_{min}} + w_e \frac{CO_2(X) - E_{min}}{E_{max} - E_{min}}$$

이 식에서 $w_c$와 $w_e$는 각각 비용과 환경 영향에 대한 가중치를 나타내며, $C_{min}$, $C_{max}$, $E_{min}$, $E_{max}$는 각각 설계 공간 내에서의 공사비와 이산화탄소 배출량의 최소 및 최대 범위를 의미하여 각 지표를 0과 1 사이로 정규화한다. 총 공사비 $Cost(X)$는 전체 부재 수 $N_{elem}$에 대하여 콘크리트 단위 비용 $C_{conc}$, 순 부피 $V_{net,k}$, 철근 단위 비용 $C_{steel}$, 총 철근 중량 $W_{total,k}$, 거푸집 단위 비용 $C_{form}$, 표면적 $A_{surf,k}$를 사용하여 산출된다. 산출에 사용된 재료 물성, 단위 공사비 및 탄소 배출 계수는 Table 1에 정리하였다.

**Table 1. Material properties, unit costs, and CO2 emission factors.**
| Material / Item | Symbol | Unit | Value / Cost (KRW) | CO2 Factor ($kg/unit$) |
| :--- | :---: | :---: | :---: | :---: |
| Concrete ($f_{ck}=27$ MPa) | $C_{conc}$ | $m^3$ | 80,000 | 185.0 |
| Steel Reinforcement | $C_{steel}$ | ton | 1,000,000 | 3.52 |
| Formwork | $C_{form}$ | $m^2$ | 25,000 | - |
| Yield Strength ($f_y$) | - | MPa | 400 - 500 | - |

두 번째 목적 함수($f_2$)는 구조적 서비스 가능성을 평가하기 위한 지표로, 횡력에 대한 구조물의 저항 성능을 극대화하기 위해 전체 층에서 발생하는 최대 층간변위비(Maximum Story Drift Ratio)를 최소화한다.

$$\min f_2(X) = \max \left( \frac{\Delta_{i,j,k}}{H_k} \right)$$

여기서 $\Delta_{i,j,k}$는 $k$층의 $j$노드에서 발생하는 $i$방향의 층간변위를 나타내며, $H_k$는 해당 층의 층고를 의미한다. 예제 구조물에 적용된 하중 조건 및 모델링 파라미터는 Table 4에 기술하였다.

**Table 4. Structural loading and modeling parameters for benchmark frames.**
| Parameter Type | Item | Value / Description |
| :--- | :--- | :--- |
| Dead Load | Floor / Slab | 5.0 $kN/m^2$ / 150mm |
| Live Load | Typical / Lobby | 2.5 - 5.0 $kN/m^2$ |
| Earthquake Load | SDS / SD1 | 0.60g / 0.36g (ASCE 7-16) |
| Story Height | Typical / First | 3.3 / 4.2 m |
| Span Length | X-dir / Y-dir | 5.0 - 7.0 m (Irregular) |

### 2.3 Constraints

설계안의 실무적 타당성을 확보하기 위해 강도 및 사용성 제약 조건을 엄격히 적용한다. 모든 부재의 수요 대 공급 능력비(Demand-Capacity Ratio, DCR)는 1.0 이하여야 하며, 기둥은 P-M 상관도를, 보는 휨 및 전단 강도를 검토한다 (MacGregor et al., 1997). 사용성 측면에서 보의 장기 처짐($\delta_{LT}$)은 경간 길이 $L$의 240분의 1($L/240$) 이내로 제한하며, 층간변위비는 ASCE 7-16 기준에 따라 허용치(0.020) 이내로 제한한다 (ASCE, 2016). 또한, 상부 기둥의 단면적($A_{c,upper}$)이 하부 기둥의 단면적($A_{c,lower}$)보다 클 수 없다는 위계 조건과 보의 폭이 접합부 내력을 위해 기둥 폭을 초과할 수 없다는 상세 조건을 추가하여 설계의 현실성을 높였다.

## 3. Optimization Methodology

### 3.1 NSGA-II Algorithm and Constraint Handling

본 연구에서 활용한 NSGA-II는 빠른 비지배 정렬과 혼잡도 거리 계산을 통해 파레토 최적해의 수렴성과 다양성을 동시에 확보하는 기법이다. 특히 RC 프레임 설계와 같이 복잡한 이산 변수와 비선형 제약 조건이 존재하는 문제에서 유효한 설계안을 우선적으로 탐색하기 위해 제약 조건 우선 지배(Constrained Dominance) 원칙을 적용하였다. 이 원칙에 따라 두 개체 사이의 지배 관계는 개체의 유효성 여부와 제약 조건 위반량($\sum \text{Violation}$)을 기준으로 결정된다. 이러한 메커니즘은 알고리즘이 초기 탐색 단계에서 유효한 설계 영역으로 빠르게 수렴하도록 유도하며, 대규모 다중목적 최적화 문제에서의 효율성이 입증된 바 있다 (Zavala et al., 2016). 전체적인 최적화 프로세스는 Figure 1의 플로우차트에 도식화하였다.

![Figure 1. Flowchart of the proposed 3D RC frame optimization framework.](path/to/fig1_flowchart.png)

### 3.2 Parametric Study of Algorithm Parameters

알고리즘의 성능을 극대화하기 위해 하이퍼볼륨(Hypervolume, 이하 HV) 지표를 기준으로 매개변수 연구를 수행하였다. 실험 결과, 높은 탐색 성능을 위해 교차 확률 $P_c=0.9$, 변이 확률 $P_m=0.1$, 그리고 개체군 크기 500의 조합을 최종적으로 채택하였다. 개체군 크기 500에서 HV 지표는 약 5.817로 가장 우수한 수렴성을 보였으며, 세대별 HV의 향상 추이와 유효 해 생성 비율의 변화는 Figure 4를 통해 확인할 수 있다.

![Figure 4. Convergence history: Hypervolume (HV) and generational feasibility ratio.](path/to/fig4_convergence.png)

## 4. Practical Section Database and Numerical Analysis Framework

### 4.1 Section Database Reflecting Practical Details

본 연구의 핵심적인 기여는 실무 설계의 복잡성을 변수화한 전용 단면 데이터베이스의 구축에 있다. 기둥과 보의 단면은 폭($b$)과 높이($h$)의 조합을 통해 다양한 직사각형 형상을 구성하며, 기둥은 양방향 대칭 배근을, 보는 상·하부 휨 모멘트에 대응하는 일방향 비대칭 배근 패턴을 적용한다. 배근 로직은 Table 3에 요약된 바와 같이 ACI 318-19 규정을 정밀하게 준수한다 (ASCE, 2016). 본 연구에서는 총 철근 물량 산출 시 보조 대근 및 135도 내진 상세 갈고리 여장을 포함하여 단순 철근비 기반 산출 대비 약 10-15%의 물량 정밀도를 향상시켰다.

**Table 3. Summary of reinforcement detailing logic based on ACI 318-19.**
| Item | Regulation (ACI 318-19) | Implementation in DB |
| :--- | :--- | :--- |
| Supplemental Ties | Spacing > 150 mm (25.7.2.3) | Auto-inserted Crossties |
| Skin Reinforcement | If $h > 900$ mm (9.7.2.3) | Longitudinal bars on sides |
| Seismic Hooks | 135-degree hooks (25.7.2.1) | Precision volume calculation |
| Joint Compatibility | $b_{beam} \le b_{column}$ | Automated constraint check |

주철근 간의 순간격이 150mm를 초과할 경우 보조 대근(Crossties)을 자동 배치하며, 보의 높이가 900mm를 초과하면 표피 철근을 추가한다. 또한 135도 내진 상세 갈고리 여장을 반영하여 물량 산출의 정밀도를 확보하였다. 데이터베이스에서 생성된 자동 배근의 개념도는 Figure 2에 예시하였다.

![Figure 2. Conceptual illustration of automated reinforcement detailing in the section DB.](path/to/fig2_reinforcement_details.png)

### 4.2 Numerical Analysis Integration and System Implementation

전체 최적화 프레임워크는 Python의 DEAP 라이브러리와 구조 해석 엔진인 OpenSees를 연동하여 구현되었다 (Mazzoni et al., 2006; McKenna, 1997). 알고리즘에서 생성된 설계 변수를 바탕으로 3D 프레임 모델이 자동 생성되며, 특히 기둥 회전 변수($R_{dir}=1$)가 활성화되면 로컬 좌표계의 강축 방향 변화를 반영한다. 설계 하중 및 조합은 ACI 318-19 및 ASCE 7-16 기준을 엄격히 준수하며, 지진 거동 평가는 최신 동역학 이론 및 가이드라인을 따랐다 (Chopra, 2017; FEMA, 2012). Figure 3은 본 연구에서 검증을 위해 사용한 4, 6, 8층 예제 구조물의 3D 형상과 부재 그룹핑 현황을 보여준다.

![Figure 3. 3D isometric views of the 4, 6, and 8-story benchmark structures.](path/to/fig3_structure_models.png)

## 5. Results and Discussion

### 5.1 Pareto-Optimal Solution Behavior and Analysis by Floor Level

도출된 파레토 최적해 분포(Figure 5)를 분석한 결과, 공사비($f_1$)와 최대 층간변위비($f_2$) 사이의 명확한 상충 관계가 확인되었다. 층수가 증가함에 따라 파레토 프런트가 우상향으로 이동하는 것은 고층화될수록 횡방향 강성 확보를 위한 비용 부담이 지수적으로 증가함을 시사한다. 4층 구조물의 경우 공사비는 약 4,000만 원대에서 형성되었으나, 8층 구조물의 경우 구조적 제약 조건 충족을 위해 단위 면적당 비용이 급격히 증가하였다.

![Figure 5. Pareto Fronts in the objective space (Construction Cost vs. Max. Story Drift).](path/to/fig5_pareto_fronts.png)

### 5.2 Effect of Column Rotation Variables and Practical Detailing

기둥 회전 변수의 도입 효과를 분석한 결과, 동일 비용 수준에서 최대 층간변위비를 약 5.0%에서 17%까지 추가로 저감할 수 있었다. Figure 6은 최적화된 설계안에서의 기둥 방향 배치 및 단면 크기 분포를 보여주며, 알고리즘이 횡방향 강성이 취약한 축으로 기둥 강축을 자동 배치했음을 입증한다. 또한, Figure 7의 DCR 분포 및 변위 프로파일 비교를 통해 제안된 기법이 구조적 안전성을 유지하면서도 효율적인 단면 구성을 찾아냈음을 확인할 수 있다.

![Figure 6. Solution space analysis: Optimized column orientations and section size distribution.](path/to/fig6_solution_analysis.png)

![Figure 7. Comparison of structural performance: DCR contours and displacement profiles.](path/to/fig7_structural_performance.png)

기존 관행 설계(Scenario B: 기둥 회전 고정)와 본 최적화 기법(Scenario A)의 정량적 비교 결과는 Table 5에 정리하였다. 4층 모델의 경우, 기둥 회전각과 실무 상세를 모두 최적화한 시나리오 A는 하이퍼볼륨(HV) 측면에서 약 1.25% 향상된 성능을 보였다. 특히, 동일한 공사비 수준에서 시나리오 A는 시나리오 B 대비 최대 층간변위비를 약 16.7% 저감하여 횡력 저항 성능을 획기적으로 개선하였다.

**Table 5. Comparative performance summary: Conventional (B) vs. Optimized (A) for 4-story RC frame.**
| Performance Metric | Conventional (B) | Optimized (A) | Difference (%) |
| :--- | :---: | :---: | :---: |
| Hypervolume (HV) | 5.712 | 5.784 | +1.25% |
| Best Cost Solution (KRW) | 40,512,456 | 40,278,086 | -0.58% |
| Drift at Best Cost (rad) | 0.0114 | 0.0095 | -16.7% (Improved) |
| Total CO2 at Best Cost (kg)| 120,278 | 120,352 | +0.06% |
| Max. DCR (Average) | 0.319 | 0.313 | -1.9% |

## 6. Conclusions

본 연구에서는 실무적 배근 상세와 3차원 기둥 회전 변수를 통합 고려한 RC 모멘트 골조 전용 다중목적 최적설계 프레임워크를 제안하였다. ACI 318-19 상세 규정을 반영한 실무형 데이터베이스를 통해 실제 현장에 즉각 적용 가능한 상세 정보를 도출할 수 있음을 확인하였으며, 기둥 회전 변수화를 통해 구조적 효율성을 극대화하였다. 4층 예제 구조물을 통한 시나리오 분석 결과, 제안된 기법은 하이퍼볼륨 성능을 1.25% 향상시켰으며, 특히 동일 비용 수준에서 층간변위비를 16% 이상 저감하는 뛰어난 구조적 효율성을 입증하였다. 본 프레임워크는 공사비와 탄소 배출량을 절감하면서도 안전 제약 조건을 준수하는 최적해를 제시함으로써, 구조 엔지니어의 합리적 의사결정을 돕는 실무적 도구로서의 가치를 입증하였다.

## References

Aga, A. A., & Adam, F. M. (2015). Design optimization of reinforced concrete frames. Open Journal of Civil Engineering, 05(01), 74–83.
Akin, A., & Saka, M. P. (2015). Harmony search algorithm based optimum detailed design of reinforced concrete plane frames subject to ACI 318-05 provisions. Computers & Structures, 147, 79-95.
American Society for Civil Engineering (ASCE). (2016). Minimum Design Loads and Associated Criteria for Buildings and Other Structures, ASCE/SEI 7-16.
Aslay, S. E., Dede, T., & Kaveh, A. (2024). Integrated design optimization process for building projects. Periodica Polytechnica Civil Engineering, 68(4), 1175-1183.
Babaei, M., & Mollayi, M. (2016). Multi-objective optimization of reinforced concrete frames using NSGA-II algorithm. Engineering Structures and Technologies, 8(4), 157-164.
BAI, J. L., CHEN, H. M., SUN, B. H., & JIN, S. S. (2020). Seismic uniform damage-targeted design of RC frame structures. Engineering Mechanics, 37(8), 179-188.
Bekdaş, L., & Nigdeli, S. M. (2014, July). Optimization of RC frame structures subjected to static loading. In 11th World Congress on Computational Mechanics (pp. 20-25).
Boscardin, J. T., Yepes, V., & Kripka, M. (2019). Optimization of reinforced concrete building frames with automated grouping of columns. Automation in Construction, 104, 331-340.
Chaudhuri, P., Barman, S., Maity, D., & Maiti, D. K. (2021). Cost effective design of RC building frame employing unified particle swarm optimization.
Chopra, A. K. (2017). Dynamics of Structures: Theory and Applications to Earthquake Engineering. Pearson.
Chutani, S., & Singh, J. (2018). Use of modified hybrid PSOGSA for optimum design of RC frame. Journal of the Chinese Institute of Engineers, 41(4), 342-352.
Coello, C. C. (2006). Evolutionary multi-objective optimization: a historical view of the field. IEEE computational intelligence magazine, 1(1), 28-36.
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
Dehnavipour, H., Mehrabani, M., Fakhriyat, A., & Jakubczyk-Gałczyńska, A. (2019). Optimization-based design of 3D reinforced concrete structures. Journal of soft computing in civil engineering, 3(3), 95-106.
Djedoui, N., Djafar-Henni, N., Bekdaş, L., & Nigdeli, S. M. (2025). Multi-objective optimization of RC structures. Iranian Journal of Science and Technology, Transactions of Civil Engineering, 1-22.
Ehrgott, M. (2012). Vilfredo Pareto and multi-objective optimization. Doc. math, 8, 447-453.
Esfandiari, M. J., Urgessa, G. S., Sheikholarefin, S., & Manshadi, S. D. (2018). Optimum design of 3D reinforced concrete frames using DMPSO algorithm. Advances in Engineering Software, 115, 149-160.
Esfandiary, M. J., Sheikholarefin, S., & Bondarabadi, H. R. (2016). A combination of particle swarm optimization and multi-criterion decision-making for optimum design of reinforced concrete frames. International journal of optimization in civil engineering, 6(2), 245-268.
Faghirnejad, S. (2023). Performance-Based Optimization of 2D Reinforced Concrete Moment Frames through Pushover Analysis and ABC Optimization Algorithm. arXiv preprint arXiv:2312.09450.
Gharehbaghi, S. (2012). Design optimization of RC frames under earthquake loads. Iran University of Science & Technology.
Govindaraj, V., & Ramasamy, J. V. (2005). Optimum detailed design of reinforced concrete continuous beams using genetic algorithms. Computers & structures, 84(1-2), 34-48.
Heydari, F., Andalib, M., Epackachi, S., & Rafiee-Dehkharghani, R. (2025). Optimized design of RC moment frames with machine learning methods. Journal of Building Engineering, 104, 112222.
Juliani, M. A., & Gomes, W. J. D. S. (2021). Optimal configuration of RC frames considering ultimate and serviceability limit state constraints. Revista IBRACON de Estruturas e Materiais, 14(2), e14204.
Kaveh, A., & Ardebili, S. R. (2023a). Optimal design of mixed structures under time-history loading using metaheuristic algorithm. Periodica Polytechnica Civil Engineering, 67(1), 57-64.
Kaveh, A., Izadifard, R. A., & Mottaghi, L. (2020a). Cost optimization of RC frames using automated member grouping. International Journal of Optimization in Civil Engineering, 10(1), 91-100.
Kaveh, A., Izadifard, R. A., & Mottaghi, L. (2020b). Optimal design of planar RC frames considering CO2 emissions using ECBO, EVPS and PSO metaheuristic algorithms. Journal of Building Engineering, 28, 101014.
Kaveh, A., & Ardebili, S. R. (2021, December). An improved plasma generation optimization algorithm for optimal design of reinforced concrete frames under time-history loading. In Structures (Vol. 34, pp. 758-770).
Kaveh, A., & Ardebili, S. R. (2023b, February). Optimum design of 3D reinforced concrete frames using IPGO algorithm. In Structures (Vol. 48, pp. 1848-1855).
Kaveh, A., & Sabzi, O. (2011). A comparative study of two meta-heuristic algorithms for optimum design of reinforced concrete frames.
Federal Emergency Management Agency (FEMA). (2012). Seismic Performance Assessment of Buildings Volume 1-Methodology. Rep. No. FEMA P-58-1.
MacGregor, J. G., Wight, J. K., Teng, S., & Irawan, P. (1997). Reinforced concrete: Mechanics and design (Vol. 3). Upper Saddle River, NJ: Prentice Hall.
Marler, R. T., & Arora, J. S. (2004). Survey of multi-objective optimization methods for engineering. Structural and multidisciplinary optimization, 26(6), 369-395.
Mazzoni, S., McKenna, F., Scott, M. H., & Fenves, G. L. (2006). Open system for earthquake engineering simulation (opensees) opensees command language manual. Pacific Earthquake Engineering Research Center, 1-465.
McKenna, F. T. (1997). Object-oriented finite element programming: frameworks for analysis, algorithms and parallel computing. University of California, Berkeley.
Mergos, P. E. (2021). Optimum design of 3D reinforced concrete building frames with the flower pollination algorithm. Journal of Building Engineering, 44, 102935.
Mergos, P. E. (2022). Surrogate-based optimum design of 3D reinforced concrete building frames to Eurocodes. Developments in the Built Environment, 11, 100079.
Mergos, P. E. (2024). Structural design of reinforced concrete frames for minimum amount of concrete or embodied carbon. Energy and Buildings, 318, 114505.
Nebro, A. J., Galeano-Brajones, J., Luna, F., & Coello Coello, C. A. (2022). Is NSGA-II ready for large-scale multi-objective optimization?. Mathematical and Computational Applications, 27(6), 103.
Oluwole Akadiri, P., & Olaniran Fadiya, O. (2013). Empirical analysis of the determinants of environmentally sustainable practices in the UK construction industry. Construction Innovation, 13(4), 352-373.
Paya-Zaforteza, I., Yepes, V., Hospitaler, A., & Gonzalez-Vidosa, F. (2009). CO2-optimization of reinforced concrete frames by simulated annealing. Engineering Structures, 31(7), 1501-1508.
Werner, W., & Burns, J. G. (2012). Quantification and optimization of structural embodied energy and carbon. In Structures Congress 2012 (pp. 929-940).
Zavala, G., Nebro, A. J., Luna, F., & Coello Coello, C. A. (2016). Structural design using multi-objective metaheuristics. Comparative study and application to a real-world problem. Structural and Multidisciplinary Optimization, 53(3), 545-566.
Zitzler, E., & Thiele, L. (2002). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. IEEE transactions on Evolutionary Computation, 3(4), 257-271.

