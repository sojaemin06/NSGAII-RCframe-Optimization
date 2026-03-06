# Representative Optimized Designs for Section Detailing Analysis (4-Story)

본 문서는 4층 RC 모멘트 골조 예제의 최적화 결과 중, 선정된 두 가지 대표 사례(Solution ID 1, ID 20)의 전체 설계 변수 벡터(Design Variable Vector) 구성을 정리한 것입니다.

## 1. Design Comparison Overview
*   **Solution ID 1 (Economical Design):** 공사비와 탄소 배출량을 최소화한 설계안 (Pareto Front의 한쪽 극단)
*   **Solution ID 20 (High-Stiffness Design):** 층간변위 제어 성능을 극대화한 설계안 (Pareto Front의 반대쪽 극단)

## 2. Design Variable Vector Table (Column ID -> Column Rotation -> Beam ID)

각 솔루션은 실제 알고리즘의 탐색 순서인 기둥 정보(ID, 회전)와 보 정보(ID) 순으로 구성된 하나의 벡터입니다.

| Category | Floor | Group / Position | **Solution ID 1** | **Solution ID 20** |
|:---:|:---:|:---|:---:|:---:|
| **Column Section ID** | **1F** | Corner (Grp 0) | 4 | 676 |
| | | Edge (Grp 1) | 7 | 747 |
| | | Interior (Grp 2) | 15 | 402 |
| | **2F** | Corner (Grp 3) | 47 | 774 |
| | | Edge (Grp 4) | 28 | 692 |
| | | Interior (Grp 5) | 8 | 584 |
| | **3F** | Corner (Grp 6) | 170 | 754 |
| | | Edge (Grp 7) | 11 | 52 |
| | | Interior (Grp 8) | 11 | 716 |
| | **4F** | Corner (Grp 9) | 201 | 640 |
| | | Edge (Grp 10) | 157 | 473 |
| | | Interior (Grp 11) | 28 | 238 |
| **Column Rotation** | **1F** | Corner (Grp 0) | 1 | 1 |
| (0=0°, 1=90°) | | Edge (Grp 1) | 1 | 0 |
| | | Interior (Grp 2) | 1 | 1 |
| | **2F** | Corner (Grp 3) | 0 | 1 |
| | | Edge (Grp 4) | 1 | 0 |
| | | Interior (Grp 5) | 1 | 1 |
| | **3F** | Corner (Grp 6) | 1 | 0 |
| | | Edge (Grp 7) | 0 | 0 |
| | | Interior (Grp 8) | 1 | 0 |
| | **4F** | Corner (Grp 9) | 0 | 0 |
| | | Edge (Grp 10) | 0 | 1 |
| | | Interior (Grp 11) | 1 | 0 |
| **Beam Section ID** | **1F** | Exterior (Grp 0) | 71 | 436 |
| | | Interior (Grp 1) | 27 | 351 |
| | **2F** | Exterior (Grp 2) | 131 | 384 |
| | | Interior (Grp 3) | 11 | 472 |
| | **3F** | Exterior (Grp 4) | 90 | 344 |
| | | Interior (Grp 5) | 27 | 398 |
| | **4F** | Exterior (Grp 6) | 60 | 404 |
| | | Interior (Grp 7) | 6 | 436 |

---
*Note: 각 ID 값은 설계 변수 데이터베이스 내의 고유 인덱스이며, Rotation은 기둥의 강축 방향(0: 0도, 1: 90도)을 의미합니다.*
