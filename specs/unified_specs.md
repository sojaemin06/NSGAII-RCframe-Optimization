# 통합 명세서: NSGA-II RC 프레임 최적화 연구 개선 (Unified Specifications)

이 문서는 리뷰어(R1, R2, R3)의 요구사항을 반영하여 프로젝트를 개선하기 위한 통합 작업 명세서입니다. 작업은 중요도와 의존성에 따라 우선순위별로 정렬되었습니다.

---

## 🏗️ Phase 1: 핵심 모델링 및 해석 엔진 고도화 (Reliability First)
구조 해석의 정확성과 현실성을 확보하는 것이 가장 시급합니다.

### 1.1. 구조 해석 로직 보완 (R2-7, R2-8, R3-14, R3-15, R3-13)
- [x] **강체 횡격막 (Rigid Diaphragm):** `ops.rigidDiaphragm` 등을 사용하여 층별 자유도(1, 2, 6) 구속 구현.
- [x] **P-Delta 효과 (P-Delta Effect):** `geomTransf('PDelta', ...)` 적용 확인.
- [x] **유효 강성 (Effective Stiffness):** ACI 318 기준 적용 (Column 0.7, Beam 0.35) 확인.
- [x] **지진 하중 현실화 (Seismic Load):**
    - [x] 밑면 전단력($V$) 산정 로직 구현 (ELF).
    - [x] 총 중량($W$) 산정 시 슬래브 고정 하중 포함 확인.
    - [x] **검증:** `scripts/experiment_high_load.py` 실행하여 600kN 수준의 전단력에서도 해석이 수렴하는지 확인.
- [x] **풍하중 (Wind Load):** ASCE 7-16 기준(V=30m/s, 노출 B) 적용 확인.
- [x] **장기 처짐 (Long-term Deflection) 구현:** (R3-13)
    - `src/structural_analysis.py`에 장기 처짐 검토 로직 추가.
    - 식: $\Delta_{total} = \Delta_{immediate} + \lambda_\Delta \Delta_{sustained}$
    - 계수: $\lambda_\Delta = \frac{2.0}{1 + 50\rho'}$ (5년 이상 지속 하중 기준)

### 1.2. 평가 모듈 및 비용 모델 업데이트 (R3-3, R3-4, R3-13)
- [x] **거푸집 비용 (Formwork Cost) 포함:**
    - [x] 기둥 및 보의 표면적 기반 비용 산정 로직 추가.
    - [x] `src/config.py`에 단위 비용 파라미터 추가 확인.
- [ ] **모듈 구조 정비:** `src/structural_analysis.py`의 평가 로직을 점검하고 필요 시 `src/evaluation.py`로 분리 검토.

---

## 🛠️ Phase 2: 최적화 및 평가 도구 강화 (Tools Update)
검증을 수행하고 데이터를 확보하기 위한 도구를 보완합니다.

### 2.1. 최적화 지표 및 로깅 (R1-3, R1-5)
- [x] **하이퍼볼륨 (Hypervolume):** `src/optimization.py`에 다목적 최적화 성능 지표 계산 로직 구현 확인.
- [x] **실행 시간 측정 (Timer):** 최적화 소요 시간 측정 및 로깅 기능 구현 확인.

### 2.2. 일반 하중 조건 설정 (R2-11)
- [ ] **대칭 하중 조건 준비:**
    - `config.py` 또는 실험 스크립트에서 풍하중($W_x, W_y$) 및 지진하중($E_x, E_y$)을 대칭적으로 설정할 수 있도록 준비.
    - 이는 알고리즘이 특정 편심 상황이 아닌 일반적인 상황에서도 우수함을 보이기 위함임.

### 2.3. 최적 파라미터 결정 실험 (R1-3)
- [ ] **실험 스크립트 작성:** `scripts/experiment_optimization_params.py`
    - 목적: 통계적 검증(Phase 3) 전에 최적의 GA 파라미터(Pop Size, Crossover/Mutation Rate 등)를 선정.
    - 방법: 파라미터 조합별 Hypervolume 수렴도 비교.

---

## 🧪 Phase 3: 데이터 생성 및 검증 (Validation Runs)
논문에 수록할 정량적 데이터를 생성합니다.

### 3.1. 최적 파라미터 결정 (선행 작업)
- [ ] **실험 수행:** Phase 2.3에서 작성한 스크립트를 실행하여 최적 설정 도출.

### 3.2. 통계적 검증 (R2-10, R3-16)
- [ ] **스크립트 작성:** `scripts/batch_run_optimization.py`
    - 전체 최적화를 30회 반복 실행.
    - Hypervolume, 최적 비용, CO2 등의 분포(Boxplot, 표준편차) 데이터 확보.

### 3.3. 일반 하중 조건 성능 검증 (R2-11)
- [ ] **실험 수행:**
    - 대칭/일반 하중 조건에서 최적화를 수행하여 수렴성 및 해의 품질 확인.

### 3.4. 고하중 조건 안정성 확인 (R2-8)
- [x] **실험 수행:**
    - 작업 1.1에서 준비된 `scripts/experiment_high_load.py`를 실행하여 고하중(600kN) 조건에서의 모델 안정성 최종 확인.

---

## 📝 Phase 4: 문서화 및 시각화 (Paper Polishing)
확보된 결과를 문서와 그림에 반영하여 마무리합니다.

### 4.1. 텍스트 및 용어 수정 (R1-1, R1-4, R3-5, R3-9)
- [ ] **철근 상세 (Reinforcement Detailing) 명시:**
    - 데이터베이스가 실제 철근 배치(Bar Layout)를 포함하고 있음을 문서에 명확히 서술하여 한계점 반박.
- [ ] **과장된 표현 수정:** "First innovation" 등 삭제.
- [ ] **기준 연도 명시:** 코드 주석 및 출력에 `ACI 318-19`, `ASCE 7-16` 명시.
- [ ] **리뷰어 대응 텍스트:** R2-1(부등호), R3-2(용어 정의) 등 수정 사항 반영.

### 4.2. 시각화 업데이트 (R3-10, R3-18)
- [ ] **그림 코드 점검:** `visualize_load_patterns` 등 시각화 함수가 출판 품질의 이미지를 생성하는지 확인.
- [ ] **테이블 레이블:** 회전 플래그(0/1) 등의 레이블을 명확히 수정.

### 4.3. 데이터 공개 준비 (R2-14)
- [ ] **저장소 정리:** `README.md` 업데이트 및 민감 데이터 제거 확인.