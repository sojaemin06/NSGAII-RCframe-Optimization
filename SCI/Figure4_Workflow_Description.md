# Figure 4. Framework for Structural Section Detailing and Pareto-based DB Reduction

이 문서는 본 연구의 핵심 방법론인 '실무형 단면 데이터베이스 구축 및 지능형 감축 워크플로우'를 도식화한 Figure 4의 학술적 상세 내용입니다.

---

## 1. Figure 4 Flowchart (Academic Version)

```mermaid
flowchart TD
    %% Phase 1: Input & Seeding
    subgraph P1 ["<b>Phase 1: Structural Parameter Seeding</b>"]
        Start([<b>START</b>]) --> A[<b>Design Space Definition:</b><br/>Section Geometry ($b, h$)<br/>Material Strengths ($f_{ck}, f_y$)<br/>Reinforcement Diameters ($d_b, d_{sv}$)]
        A --> B{<b>Member<br/>Categorization</b>}
    end

    %% Phase 2: Detailing Engine (Column)
    subgraph P2C ["<b>Phase 2-C: Column Section Detailing</b>"]
        direction TB
        B -- "Column" --> C1["<b>Longitudinal Arrangement:</b><br/>Four-corner bars + Pairwise side bars<br/>(Biaxial symmetry strategy)"]
        C1 --> C2["<b>Spacing & Clearance Validation:</b><br/>Calculate clear spacing ($s_h, s_b$)<br/>Verify $s \ge s_{min}$ (ACI 318)"]
        C2 --> C3{"<b>Feasibility<br/>Check</b>"}
    end

    %% Phase 2: Detailing Engine (Beam)
    subgraph P2B ["<b>Phase 2-B: Beam Section Detailing</b>"]
        direction TB
        B -- "Beam" --> B1["<b>Flexural Detailing:</b><br/>Doubly reinforced layout ($\rho, \rho'$ )<br/>Multi-layer rebar arrangement"]
        B1 --> B2["<b>Effective Depth Mapping:</b><br/>Update effective depth ($d$) based on<br/>concrete cover and bar layers"]
        B2 --> B3["<b>Transverse Reinforcement:</b><br/>Stirrup spacing ($s_v$) per ACI 318 seismic codes"]
    end

    %% Phase 3: Capacity & Metric Evaluation
    subgraph P3 ["<b>Phase 3: High-Fidelity Performance Evaluation</b>"]
        direction TB
        C3 -- "Pass" --> C4["<b>Biaxial P-M Analysis:</b><br/>Strain Compatibility Method<br/>Quantify Interaction Volume ($V_{PM}$)"]
        
        B3 --> B4["<b>Mechanical Resistance:</b><br/>Design flexural strength ($\phi M_n$)<br/>Design shear strength ($\phi V_n$)"]
        
        B4 --> B5{"<b>$\phi M_n \ge 1.2 M_{cr}$</b>"}

        C4 --> D["<b>Quantitative Metrics:</b><br/>- Total Construction Cost ($Cost$)<br/>- Embodied Carbon ($CO_2$)<br/>- Unit Weight ($w$)"]
        B5 -- "Yes" --> D
    end

    %% Phase 4: DB Reduction
    subgraph P4 ["<b>Phase 4: Smart Search Space Reduction</b>"]
        D --> E["<b>Geometric Clustering:</b><br/>Grouping by identical dimensions ($b \times h$)"]
        E --> F["<b>Pareto Efficiency Filtering:</b><br/>$\min(Cost, CO_2)$ vs. $\max(Capacity)$"]
        F --> G[<b>Optimized Section Database</b><br/>(Ready for NSGA-II Execution)]
    end

    %% Error Loops
    C3 -- "Fail" --> A
    B5 -- "No" --> A
    G --> End([<b>END</b>])

    %% Styling
    style P1 fill:#ffffff,stroke:#333,stroke-width:2px
    style P2C fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style P2B fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style P3 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style P4 fill:#eceff1,stroke:#455a64,stroke-width:2px
    style G fill:#cfd8dc,stroke:#263238,stroke-width:3px
```

---

## 2. 학술적 워크플로우 상세 설명

### Phase 1: Structural Parameter Seeding
사용자가 정의한 이산적 범위 내에서 설계 변수를 샘플링합니다. $b, h$의 기하학적 치수와 $f_{ck}, f_y$의 재료 강도, 그리고 주철근($d_b$) 및 전단철근($d_{sv}$)의 직경이 포함됩니다.

### Phase 2: Detailing Engine (ACI 318-19)
*   **Column:** 모서리 철근 배치를 시작으로 전·후면 및 좌·우측 면에 대칭적으로 철근을 분배합니다. 계산된 중심 간격($s_h, s_b$)이 ACI 318에서 규정하는 최소 간격($s_{min}$)을 만족하는지 검토합니다.
*   **Beam:** 인장 및 압축 철근($\rho, \rho'$)을 배치하고, 다층 배근 시 발생하는 유효 깊이($d$)의 변화를 정밀하게 산출합니다. 내진 상세에 따른 스터럽 간격($s_v$)을 결정합니다.

### Phase 3: High-Fidelity Performance Evaluation
*   **Capacity Assessment:** 변형률 적합법(Strain Compatibility)을 사용하여 기둥의 2축 상관도 체적($V_{PM}$)과 보의 설계 휨강도($\phi M_n$)를 도출합니다. 보의 경우 취성 파괴 방지를 위해 균열 모멘트($M_{cr}$)의 1.2배 이상을 확보해야 합니다.
*   **Sustainability Metrics:** 단위 공사비와 내재탄소 배출계수(ECF)를 적용하여 경제성($Cost$)과 환경성($CO_2$)을 정량화합니다.

### Phase 4: Smart Search Space Reduction
동일한 외형 치수($b \times h$)를 가진 단면 그룹 내에서 파레토 지배(Pareto dominance) 원리를 적용합니다. 비용과 탄소 배출량은 최소화하면서 구조적 성능은 극대화하는 '비지배 단면(Non-dominated sections)'만을 선별하여 최적화 엔진의 탐색 효율을 높입니다.
