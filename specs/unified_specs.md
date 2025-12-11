# 통합 명세서: NSGA-II RC 프레임 최적화 연구 개선 (Unified Specifications)

이 문서는 리뷰어(R1, R2, R3)의 요구사항을 반영하여 프로젝트를 개선하기 위한 통합 작업 명세서입니다. 작업은 중요도와 의존성에 따라 우선순위별로 정렬되었습니다.

---

## 🏗️ Phase 1: 핵심 모델링 및 해석 엔진 고도화 (Reliability First)
구조 해석의 정확성과 현실성을 확보하는 것이 가장 시급합니다. (완료됨)

### 1.1. 구조 해석 로직 보완 (R2-7, R2-8, R3-14, R3-15, R3-13)
- [x] **강체 횡격막 (Rigid Diaphragm):** `ops.equalDOF`를 사용하여 층별 자유도(1, 2, 6) 구속 구현.
- [x] **P-Delta 효과 (P-Delta Effect):** `geomTransf('PDelta', ...)` 적용 확인.
- [x] **유효 강성 (Effective Stiffness):** ACI 318 기준 적용 (Column 0.7, Beam 0.35) 확인.
- [x] **지진 하중 현실화 (Seismic Load):** ELF(등가정적해석법) 기반 동적 하중 산정 구현 완료. 고정 상수(`_RAND`) 제거됨.
- [x] **풍하중 (Wind Load):** ASCE 7-16 기준(V=30m/s, 노출 B) 적용 확인.
- [x] **장기 처짐 (Long-term Deflection):** ACI 318 기준 $\lambda_\Delta$ 계수 적용 완료.

### 1.2. 평가 모듈 및 비용 모델 업데이트 (R3-3, R3-4, R3-13)
- [x] **거푸집 비용 (Formwork Cost):** 기둥 및 보의 표면적 기반 비용 산정 로직 추가 완료.

---

## 🛠️ Phase 2: 최적화 및 평가 도구 강화 (Tools Update)
검증을 수행하고 데이터를 확보하기 위한 도구를 보완합니다.

### 2.1. 최적화 지표 및 로깅 (R1-3, R1-5)
- [x] **하이퍼볼륨 (Hypervolume):** 다목적 최적화 성능 지표 계산 로직 구현 확인.
- [x] **실행 시간 측정 (Timer):** 최적화 소요 시간 측정 및 로깅 기능 구현 확인.
- [x] **로그 간소화:** 실험 진행 시 불필요한 로그를 숨기고 진행 바(Progress Bar)만 표시하도록 개선.

### 2.2. 하중 조건 동적화
- [x] **하중 상수 제거:** `config.py`의 고정 하중 상수를 제거하고, 모든 실험 스크립트가 계산된 동적 하중을 사용하도록 리팩토링 완료.

---

## 🧪 Phase 3: 데이터 생성 및 검증 (Validation Runs) - [핵심 실행 단계]
논문에 수록할 정량적 데이터를 순차적으로 생성합니다.

### 3.1. 최적 파라미터 결정 (GA Parameter Tuning)
- **스크립트:** `experiment_optimization_params.py`
- **목표:** GA의 핵심 파라미터(Crossover Strategy, Tournament Size, Probabilities, Population Size)를 5단계(Step 1~5) 실험을 통해 확정.
- **현재 상태:** Step 1 진행 중. 완료 후 `PREV_BEST_PARAMS` 업데이트하며 순차 진행 필요.

### 3.2. 회전 변수 알고리즘 우수성 평가 (R2-11 대응)
- **스크립트:** `scripts/experiment_scenario_comparison.py`
- **목표:** 기둥 회전을 유전자로 다루는 제안 방식(Scenario A)이, 단순히 회전된 단면을 DB에 추가한 방식(Scenario B)보다 우수하거나 효율적임을 입증.
- **준비물:** `column_sections_expanded_rotated.csv`, `pm_dataset_expanded_rotated.mat`

### 3.3. 3가지 구조물 예시 최적화 (Case Studies: 4, 6, 8-Story)
- **스크립트:** `scripts/experiment_examples_comparison.py`
- **목표:** 층수가 다른 3가지 비정형 RC 프레임(4층, 6층, 8층)에 대해 제안된 최적화 알고리즘을 적용하고, 높이 변화에 따른 해의 품질과 수렴성을 비교 분석.
- **내용:** 각 예시별 최적화 실행 -> Hypervolume 비교 그래프, Pareto Front 비교 그래프, 요약 통계 CSV 생성.

### 3.4. 통계적 검증 (Statistical Validation)
- **스크립트:** `scripts/batch_run_optimization.py`
- **목표:** 3.3의 대표 예시(예: 4층)에 대해 최적 파라미터로 30회 반복 수행하여 알고리즘의 신뢰성(Stability) 검증.
- **산출물:** Boxplot, Convergence Plot (with CI), Accumulated Pareto Front.

---

## 📝 Phase 4: 문서화 및 시각화 (Paper Polishing)
확보된 결과를 문서와 그림에 반영하여 마무리합니다.

### 4.1. 텍스트 및 용어 수정 (R1-1, R1-4, R3-5, R3-9)
- [ ] **철근 상세 (Reinforcement Detailing) 명시:** 데이터베이스의 현실성 강조.
- [ ] **과장된 표현 수정:** "First innovation" 등 삭제.
- [ ] **기준 연도 명시:** 코드 및 결과물에 `ACI 318-19`, `ASCE 7-16` 명시.

### 4.2. 시각화 업데이트 (R3-10, R3-18)
- [ ] **그림 코드 점검:** 출판 품질의 해상도(300dpi 이상) 및 스타일 적용.
- [ ] **테이블 레이블:** 명확한 용어 사용.

### 4.3. 데이터 공개 준비 (R2-14)
- [ ] **저장소 정리:** `README.md` 업데이트 및 민감 데이터 제거 확인.
