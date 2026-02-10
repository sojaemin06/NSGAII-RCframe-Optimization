# 통합 명세서: 비정형 RC 프레임 최적화 연구 (Revised Unified Specs)

이 문서는 새롭게 정립된 연구 방향(`Revised_Report_KO.md`)에 따라, 리뷰어의 요구사항을 충족하고 논문의 독창성을 입증하기 위한 구체적인 작업 명세서입니다.

---

## 🏗️ Phase 1: 핵심 모델링 및 해석 엔진 고도화 (Reliability) - [완료됨]

구조 해석의 정확성과 현실성을 확보하는 단계입니다. (`src/structural_analysis.py` 반영 완료)

### 1.1. 정밀 구조 해석 (Rigorous Analysis)

- [X] **P-Delta 효과:** 기하학적 비선형성 고려.
- [X] **유효 강성:** ACI 318-19 기준 (Column 0.7, Beam 0.35) 적용.
- [X] **강체 횡격막:** `ops.equalDOF` 활용 층별 거동 일체화.
- [X] **정밀 하중 산정:** ASCE 7-16 기반 지진력(Mode-based) 및 풍하중 산정.
- [X] **제약조건 고도화:** DCR, Drift, SCWB, 장기 처짐, 기둥 면적 위계 등 7종 구현 완료.

### 1.2. 데이터베이스 검증 (Database Verification)

- [X] **MATLAB 코드 분석:** 단면 생성 로직이 예제 건물의 하중 범위(축력, 모멘트)를 커버하는지 확인.
- [X] **배근 상세 다양성 검증 (Reinforcement Detailing Diversity):** `column_sections_simple02.csv`에 편심 하중 대응을 위한 비대칭 배근 등 다양한 상세가 적절히 포함되어 있는지 검증.

---

## 🛠️ Phase 2: 최적화 로직 및 도구 업데이트 (Implementation) - [완료됨]

새로운 최적화 전략(자동 그룹핑, 회전 변수, 목적함수 변경)을 코드에 반영합니다.

### 2.1. 목적함수 및 제약조건 변경

- [X] **목적함수 2번 변경:** `Mean DCR` $\to$ `Max Inter-story Drift Ratio`. (`src/optimization.py` 수정 완료)
- [X] **제약조건 변경:** 단면 종류 수($N_{types}$) 제약 삭제, 기둥 위계 등 6종 제약 로직 유지. (`src/structural_analysis.py`)
- [X] **그룹핑 전략 복귀:** `Hybrid` 전략(기둥: 층별/위치별 그룹핑, 보: 층별/방향별 그룹핑)을 기본값으로 설정하여 탐색 공간 효율화. (`src/config.py`)

### 2.2. 실행 스크립트 역할 분담 및 고도화

- [X] **`scripts/main.py` (핵심 결과 도출):** 제안된 방법론(Scenario A) 기반 3가지 예제 실행 로직 구축 완료.
- [X] **파라미터 튜닝 스크립트:** `experiment_optimization_params.py` 새로운 로직에 맞춰 업데이트 완료.
- [X] **`scripts/experiment_scenario_comparison.py` (효율성 입증):** 신규 로직 반영 및 시나리오 B(확장 DB) 연동 필요.
- [X] **`scripts/batch_run_optimization.py` (통계적 신뢰성):** 신규 로직 반영 및 반복 실행 자동화 필요.

---

## 🧪 Phase 3: 데이터 생성 및 검증 (Validation Runs) - [진행 중]

논문의 핵심 주장(10% 절감, 회복탄력성 확보)을 뒷받침할 데이터를 생성합니다.

### 3.0. 최적 파라미터 결정 (Parameter Tuning) - [완료됨]

- [ ] **스크립트:** `experiment_optimization_params.py` 및 분석 스크립트 실행 완료.

- **결과:**
  - Crossover: `TwoPoint`, Tournament: `3`, CXPB: `0.9`, MUTPB: `0.7`
  - **PopSize:** `400` (100~600 구간 분석 결과, 성능/비용 트레이드오프가 가장 우수한 지점으로 선정).
  - 산출물: `Results_Param_Optimization/Step5_PopSize_100_600_*.png` 등 생성 완료.

### 3.1. 3가지 예제 구조물 최적화 (Generality via main.py) - [완료됨]

- [ ] **스크립트:** `scripts/main.py` 실행.

- **대상:** 4층, 6층, 8층 비정형 프레임.
- **산출물:** Cost-Drift 파레토 그래프, 수렴 그래프, 예제별 요약 CSV.

### 3.2. 시나리오 비교 실험 (Efficiency Verification) - [완료됨]

- [ ] **스크립트:** `experiment_scenario_comparison.py`

- **목표:** 위상 변수 분리의 효율성 입증 (Scenario A vs B).

### 3.3. 통계적 신뢰성 검증 (Reliability)

- [ ] **스크립트:** `batch_run_optimization.py`

- **목표:** Hypervolume 변동 계수(CV) 2~3% 이내 달성 확인.

---

## 📝 Phase 4: 논문 작성 및 시각화 (Paper Writing)

확보된 데이터를 바탕으로 `Revised_Report_KO.md`의 내용을 영문 논문으로 확장합니다.
