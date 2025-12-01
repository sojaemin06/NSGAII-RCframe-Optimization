# 논문 수정 및 코드 개선 보고서 (Revision Report)

## 1. 개요 (Overview)
본 문서는 논문 게재 거절(Reject) 사유를 리뷰어별로 분석하고, 이에 대한 코드 수정 및 방법론 개선 내역을 체계적으로 기록합니다.

## 2. 리뷰어별 대응 상세 (Detailed Response by Reviewer)

---

### 🛡️ Reviewer #1

#### R1-1: 데이터베이스 방식의 독창성 (Novelty)
* **Comment:** "Creating a database is not considerable... used before in optimization papers."
* **Response (Accept):**
    * "Innovation" 주장을 철회하고, **"Practical Implementation Strategy"**로 수정.
    * 선행 연구(Kaveh et al.)를 인용하며, 본 연구는 이를 **Hybrid Size-Shape Optimization**과 결합하여 확장했음을 강조.

#### R1-2: 설계 제약 조건 처리 소프트웨어 (Software)
* **Comment:** "What software was used to handle the design code constraints?"
* **Response (Clarify):**
    * **Matlab**으로 단면 성능(P-M curve)을 사전 계산하고, **Python** 스크립트가 이를 조회(Query)하여 ACI 318-19 기준을 검토함을 명시.
    * 관련 문서: `docs/modeling_details.md` 작성 완료.

#### R1-3: 파라미터 설정 근거 (Parameters)
* **Comment:** "How is done parameters regulation in NSGA-II?"
* **Response (Clarify):**
    * 논문 4.2절의 민감도 분석(Sensitivity Analysis)을 재강조.
    * Mutation Rate(0.2) 등은 문헌(Deb et al.) 권장값을 따랐음을 명시.

#### R1-4, R1-6: 서술 및 결론 보완 (Editing)
* **Comment:** 연도 표기 누락, 정량적 결론 부족.
* **Response (Accept):**
    * 코드 인용 시 연도(Year) 명기 (e.g., ACI 318-19).
    * 결론에 비용 절감액, CO2 감축량 등 구체적 수치 포함.

#### R1-5: 계산 비용 (Computational Cost)
* **Comment:** "How long does the solution process take?"
* **Response (Accept):**
    * 최적화 평균 소요 시간 측정 후 논문에 명시 예정.

---

### 🛡️ Reviewer #2

#### R2-1: 파레토 지배 정의 오류 (Typo)
* **Comment:** 부등호 방향 오류.
* **Response (Accept):** 수정 예정.

#### R2-2: 구조 및 분량 조절 (Structure)
* **Comment:** 서론이 너무 길고 모델링 설명은 부족함.
* **Response (Accept):**
    * NSGA-II 이론 설명 축소.
    * 구조 모델링 및 해석 기법 설명 대폭 강화 (`docs/modeling_details.md` 활용).

#### R2-3, R2-4: 목적함수 설정 타당성 (Objective)
* **Comment:** 가중치 1:1 근거 부족, 평균 DCR 목적함수 의문.
* **Response (Rebuttal):**
    * **가중치:** Cost와 CO2의 강한 양의 상관관계(Correlation) 제시.
    * **평균 DCR:** 제약조건(Max DCR)을 만족하는 해들 중 '구조적 여유도'를 평가하는 보조 지표로서의 가치 주장.

#### R2-6: SLS(사용성) 검토 한계 (Methodology)
* **Comment:** 단면 DB만으로는 전역 처짐 검토 불가.
* **Response (Clarify):**
    * 전체 구조 해석(OpenSees) 결과를 바탕으로 층간 변위(Drift)를 검토하고 있음을 명확히 기술.

#### R2-7: 구조 해석 모델 상세화 (Modeling)
* **Comment:** P-Delta, 요소 타입, 재료 모델 등 정보 누락.
* **Response (Accept & Action):**
    * **코드 수정:** `src/modeling.py`에서 기하 변환을 `Linear` -> **`PDelta`**로 변경하여 2차 효과 고려.
    * **문서화:** `elasticBeamColumn` 사용 및 재료 물성치 상세를 `docs/modeling_details.md`에 기술.

#### R2-8: 지진 하중 현실화 (Loading)
* **Comment:** "Base shear of 40 kN is unrealistic."
* **Response (Accept & Action):**
    * **코드 수정:** `scripts/simpledata_2obj_NSGA2.py`를 수정하여 하중을 외부 인자로 입력받도록 개선.
    * **실험 계획:** 하중을 10배(400kN) 이상 증대시킨 `scripts/experiment_high_load.py` 작성 완료. 실행 및 검증 예정.

#### R2-9: 예제 다양성 및 층수 오류 (Validation)
* **Comment:** 8층 건물 하나로는 부족함.
* **Response (Correction):**
    * 논문의 "8-story" 표기는 오기(Typo)였으며 실제로는 **4-story** 모델임을 밝힘.
    * 추가 예제보다는 4층 모델의 심층 분석(Deep Dive)으로 방어.

#### R2-10: 통계적 검증 (Statistics)
* **Comment:** 3회 실행은 부족함.
* **Response (Accept & Action):**
    * **코드 작성:** 30회 이상 자동 반복 실행을 위한 `scripts/batch_run_optimization.py` 개발 완료.
    * 통계적 수치(평균, 표준편차) 제시 예정.

#### R2-11: 비대칭 하중 시나리오 (Methodology)
* **Comment:** 비교가 공정하지 않음.
* **Response (Rebuttal):**
    * 형상 최적화(기둥 회전) 효과를 검증하기 위한 **의도적(Intentional)** 시나리오임을 강조.

#### R2-12: 실무적 한계 (Practicality)
* **Comment:** 그룹핑 고정, 균일 스터럽 등은 비경제적.
* **Response (Defer):**
    * 연구의 한계점(Limitation) 및 향후 연구 과제로 기술.

#### R2-13: 비교 분석 (Comparison)
* **Comment:** 기존 설계안과의 비교 부재.
* **Response (Accept):**
    * 일반적인 규준 설계(Conventional Design) 결과를 도출하여 비교 데이터 추가 예정.

#### R2-14: 데이터 공개 (Data)
* **Comment:** 데이터 제공 안 됨.
* **Response (Accept):**
    * GitHub 레포지토리 공개 준비.

---

### 🛡️ Reviewer #3

#### R3-1: 평균 DCR 목적함수 의문 (Objective)
* **Comment:** R2-4와 동일.
* **Response:** R2-4와 동일하게 대응.

#### R3-2: "Hybrid" 용어 삭제 권고 (Terminology)
* **Comment:** 알고리즘 하이브리드로 오해 소지.
* **Response (Rebuttal):**
    * "Hybrid Optimization Algorithm"이 아니라 **"Hybrid Design Variables (Size + Shape)"**임을 명확히 정의하여 용어 고수.

#### R3-3, R3-4: 철근 상세 및 거푸집 (Detailing)
* **Comment:** 철근 상세 미비, 거푸집 비용 누락.
* **Response (Defer):**
    * 연구 범위(Scope) 밖임을 명시하고 한계점으로 기술.

#### R3-5, R3-6, R3-21: 서술 오류 (Editing)
* **Comment:** 지진은 외력 아님, 중복 표현, 과한 형용사.
* **Response (Accept):**
    * 원고 전면 수정(Proofreading) 시 반영.

#### R3-7: 서론 축소 (Structure)
* **Comment:** R2-2와 동일.
* **Response:** R2-2와 동일하게 대응.

#### R3-9: 철근비 변환 설명 (Detailing)
* **Comment:** 철근비($\rho$) -> 실제 철근 개수 변환 과정 설명 요구.
* **Response (Clarify):**
    * `src/utils.py`의 로직(MATLAB 사전 계산 DB)을 바탕으로 문서 보강.

#### R3-10: 설계 변수 그림 (Visualization)
* **Comment:** 설계 변수 설명 그림 누락.
* **Response (Accept):**
    * 염색체(Chromosome) 구조 및 매핑 다이어그램 추가.

#### R3-11, R3-12: 변수 정의 (Notation)
* **Comment:** 가중치, DCR 정의 누락.
* **Response (Accept):**
    * 수식 설명 보강.

#### R3-13: 처짐 및 장기 거동 (SLS)
* **Comment:** 처짐 계산 단순화, 크리프 미고려.
* **Response (Defer):**
    * 초기 최적화 단계에서의 근사해법임을 인정하고, 상세 설계 단계에서의 보완 필요성을 한계점으로 기술.

#### R3-16: 통계적 신뢰성 (Statistics)
* **Comment:** R2-10과 동일.
* **Response:** R2-10과 동일하게 대응 (`batch_run_optimization.py`).

#### R3-17: 세대 수 적절성 (Parameters)
* **Comment:** 100세대가 충분한가?
* **Response (Clarify):**
    * 수렴 그래프(Convergence Plot)를 통해 100세대 이전에 수렴함을 증명.

#### R3-18: 결과 테이블 설명 (Results)
* **Comment:** Table 9의 0/1 의미 등 설명 부족.
* **Response (Accept):**
    * 표 주석(Footnote) 보강.

#### R3-19: 그림 분석 (Results)
* **Comment:** Figure 15 분석 텍스트 부족.
* **Response (Accept):**
    * 본문에 상세 분석 추가.

#### R3-20: 계산 시간 (Cost)
* **Comment:** R1-5와 동일.
* **Response:** R1-5와 동일하게 대응.

---

## 3. 코드 수정 이력 (Code Revision Log)

1.  **`scripts/batch_run_optimization.py` 생성:** 통계적 검증을 위한 다중 실행 스크립트.
2.  **`scripts/simpledata_2obj_NSGA2.py` 수정:**
    *   `argparse` 추가 (Seed, Population, Load 등 외부 제어).
    *   `src.utils` 임포트 추가 및 함수 호출 인자 오류 수정.
3.  **`src/modeling.py` 수정:**
    *   `geomTransf`를 `Linear`에서 `PDelta`로 변경.
4.  **`scripts/experiment_high_load.py` 생성:** 현실적인 지진 하중(400kN) 테스트용 스크립트.
5.  **문서화:** `docs/modeling_details.md` 작성 (모델링 상세 스펙).