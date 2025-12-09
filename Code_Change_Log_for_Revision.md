# 논문 수정을 위한 코드 변경 이력 (Code Change Log for Revision)

이 문서는 리뷰어들의 지적 사항을 반영하기 위해 수행된 코드 변경 내역을 요약한 것입니다. 논문의 "Methodology" 및 "Response Letter" 작성 시 참고 자료로 활용됩니다.

---

## 🏗️ 구조 모델링 및 해석 (Reviewer 2 & 3)

### R2-7: 구조 모델링 상세화 (Structural Modeling Details)
*   **수정 파일:** `src/structural_analysis.py`
*   **함수:** `build_model_for_section`
*   **변경 내용:**
    *   **강체 횡격막 (Rigid Diaphragm):** `ops.equalDOF`를 사용하여 층별 모든 절점의 횡방향(X, Y) 및 비틀림(Z회전) 자유도를 마스터 절점에 구속시키는 방식으로 구현 (기존 `rigidDiaphragm` 명령어 오류 수정).
    *   **P-Delta 효과:** 모든 기둥 요소에 `geomTransf('PDelta', ...)`를 적용하여 기하 비선형성 고려.
    *   **유효 강성 (Effective Stiffness):** ACI 318-19 기준에 따라 기둥($0.7I_g$)과 보($0.35I_g$)의 단면 2차 모멘트 저감 계수 적용.

### R2-8: 지진 하중 현실화 (Realistic Seismic Loading)
*   **수정 파일:** `src/structural_analysis.py`
*   **함수:** `evaluate`
*   **변경 내용:**
    *   **밑면 전단력 산정:** ASCE 7-16 등가정적해석법(ELF)에 따라 $V = C_s W$ 로직 구현.
    *   **지진 중량 ($W$):** 부재 자중 외에 슬래브 고정 하중(`DL_AREA_LOAD * FLOOR_AREA`)을 포함하도록 중량 산정 로직 수정.
    *   **수직 분포:** 층별 무게와 높이를 고려한 지진력 수직 분포 계수($C_{vx}$) 적용.

### R3-13: 장기 처짐 고려 (Long-term Deflection)
*   **수정 파일:** `src/structural_analysis.py`
*   **함수:** `evaluate`
*   **변경 내용:**
    *   **로직 구현:** ACI 318-19 (24.2.4.1)에 따른 장기 처짐 추가 처짐 계산 로직 구현.
    *   **계산 식:** $\Delta_{total} = \Delta_{immediate} + \lambda_\Delta \Delta_{sustained}$
    *   **시간 경과 계수:** $\lambda_\Delta = \frac{2.0}{1 + 50\rho'}$ (지속 기간 5년 이상, $\xi=2.0$ 가정).
    *   **DB 연동:** 데이터베이스의 압축 철근비(`rho_c`) 정보를 활용하여 정확한 $\lambda_\Delta$ 산정.
    *   **검토 기준:** 기존의 간이 검토($L/21$)를 삭제하고, 총 처짐을 허용 처짐($L/240$)과 비교하는 방식으로 고도화.

### R3-3: 거푸집 비용 반영 (Formwork Cost)
*   **수정 파일:** `src/structural_analysis.py`, `src/config.py`
*   **함수:** `evaluate`
*   **변경 내용:**
    *   **비용 모델:** 기둥 및 보의 표면적($2(b+h)L$)에 기반한 거푸집 시공 비용 항목을 목적함수(Cost)에 추가.
    *   **파라미터:** `config.py`에 단위 면적당 비용(`FORMWORK_UNIT_COST`) 정의.

---

## ⚙️ 최적화 및 검증 (Reviewer 1 & 2)

### R1-3: 성능 지표 강화 (Performance Metrics)
*   **수정 파일:** `src/optimization.py` (계획/확인됨)
*   **변경 내용:**
    *   **Hypervolume:** 다목적 최적화의 수렴 품질을 평가하기 위해 `deap.benchmarks.tools.hypervolume` 도입.
    *   **시간 측정:** 최적화 소요 시간(`time.time()`) 측정 및 로깅 기능 추가.

### R2-11: 일반 하중 조건 검증 (General Loading Validation)
*   **수정 파일:** `src/config.py`, `scripts/experiment_high_load.py`, `src/structural_analysis.py`
*   **변경 내용:**
    *   **대칭 하중:** 특정 비대칭 시나리오가 아닌, 일반적인 대칭 하중 조건($W_x \approx W_y$)에서 알고리즘을 검증하도록 설정 변경.
    *   **변위 계산 수정:** `evaluate` 함수 내 층간 변위 계산 시, 층별 지진력을 마스터 노드에 직접 재하(`ops.load`)하는 방식으로 수정하여 고하중 조건에서도 정확한 변위가 산출되도록 조치.

---

## 📝 문서화 및 파라미터

### R1-4: 설계 기준 연도 명시
*   **수정 파일:** 소스 코드 내 주석
*   **변경 내용:** 모든 코드 내 참조 기준을 `ACI 318` -> `ACI 318-19`, `ASCE 7` -> `ASCE 7-16`으로 구체화.
