# 프로젝트 기술 명세서: RC 골조의 하이브리드 단면-형상 최적화

## 1. 개요 (Overview)

본 프로젝트는 철근콘크리트(RC) 모멘트 골조(MRF)의 다목적 최적화를 위한 **"하이브리드 단면-형상 최적화 프레임워크(Hybrid Size-Shape Framework)"**를 구현한 것입니다. 관련 논문인 *A Novel Hybrid Size-Shape Framework for Multi-Objective Optimization of Reinforced Concrete Frames*의 방법론을 따릅니다.

**NSGA-II** 알고리즘을 사용하여 **경제성/환경성(비용 및 탄소배출)**과 **구조적 안전성(보수성)** 사이의 트레이드오프를 고려한 파레토 최적 설계를 도출하는 것이 목표입니다.

## 2. 핵심 목표 (Core Objectives)

본 최적화 문제는 상충되는 두 가지 목적 함수를 동시에 만족시키는 해를 탐색합니다:

### 목적 함수 1: 총 비용 및 탄소 배출량 최소화 ($f_1$)

건설 비용과 내재 탄소(Embodied Carbon)를 정규화하여 합산한 지표입니다.

$$
f_1(x) = w_{cost} \left( \frac{Cost(x) - C_{min}}{C_{max} - C_{min}} \right) + w_{co2} \left( \frac{CO_2(x) - E_{min}}{E_{max} - E_{min}} \right)
$$

- **비용 ($):** 재료비(콘크리트, 철근) + 노무비.
- **탄소 배출량 ($kgCO_2e$):** 사용된 재료의 내재 탄소량.

### 목적 함수 2: 구조적 보수성 최대화 ($f_2$)

모든 부재의 평균 응력비(DCR: Demand-Capacity Ratio)를 최소화하여 구조적 여유를 확보합니다.

$$
f_2(x) = \text{Mean DCR}(x) = \frac{1}{N_{members}} \sum_{i=1}^{N_{members}} DCR_{i, max}(x)
$$

- $f_2$ 값이 낮을수록 더 보수적이고 안전한 설계를 의미합니다.

## 3. 방법론 (Methodology)

### 3.1 하이브리드 최적화 전략

- **단면 최적화 (Size Optimization):** 기둥과 보 그룹에 대해 미리 계산된 데이터베이스(DB)에서 최적의 단면(Discrete Section)을 선택합니다.
- **형상 최적화 (Shape Optimization):** 기둥의 **회전 여부(0° 또는 90°)**를 별도의 이진 변수(Binary Variable)로 다룹니다. 이를 통해 단면 크기를 변경하지 않고도 횡강성을 효율적으로 조절할 수 있습니다.

### 3.2 알고리즘: NSGA-II

- **구현:** Python `DEAP` 라이브러리 활용.
- **유전자(Chromosome) 구조:**
  - `Column Section IDs` (정수형: 단면 번호)
  - `Column Rotation Flags` (이진형: 회전 여부)
  - `Beam Section IDs` (정수형: 단면 번호)
- **제약 조건 처리:** **제약 조건 지배 원칙(Constraint-Dominance Principle)** 적용. 제약 조건을 위반한 해($CV > 0$)는 위반하지 않은 해보다 항상 열등한 것으로 간주합니다.

### 3.3 구조 해석 (Structural Analysis)

- **엔진:** **OpenSees** (Open System for Earthquake Engineering Simulation).
- **모델:** 3차원 비선형/탄성 보-기둥 요소.
- **하중:**
  - 중력 하중: 고정 하중(Dead), 활하중(Live) (체크무늬 배치 고려).
  - 횡하중: 풍하중(ASCE 7 기준), 지진 하중(등가 정적 하중).

## 4. 제약 조건 (Constraints)

ACI 318-19 및 ASCE 7 기준에 따라 다음 한계 상태를 만족해야 합니다.

1. **강도 (ULS):** 휨 및 전단에 대한 DCR $\le 1.0$.
2. **사용성 (SLS):**
3. - 지진 하중 시 층간 변위각(Inter-story Drift) 제한.
   - 풍하중 시 최상층 변위 제한 ($H/400$).
4. **시공성:**
   - **강기둥-약보 (SCWB):** 접합부에서 기둥 강도가 보 강도의 1.2배 이상이어야 함.
   - **기하학적 조건:** 보 폭이 기둥 폭보다 작거나 같아야 함.
