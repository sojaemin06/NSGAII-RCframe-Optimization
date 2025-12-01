# Structural Modeling & Analysis Details

This document provides a comprehensive description of the finite element modeling and structural analysis procedures implemented in this project, addressing the concerns raised by reviewers regarding reproducibility and methodological clarity.

## 1. Overview
The structural optimization framework utilizes **OpenSees (Open System for Earthquake Engineering Simulation)** via the `openseespy` Python interface. The model represents a 3D Reinforced Concrete (RC) Moment Resisting Frame (MRF).

**Key Modeling Decisions:**
- **Dimensionality:** 3D Space (6 Degrees of Freedom per node).
- **Element Type:** Elastic Beam-Column Elements.
- **Geometric Nonlinearity:** P-Delta effects are considered.
- **Boundary Conditions:** Fixed base.

## 2. Finite Element Formulation

### 2.1 Element Type
All structural members (beams and columns) are modeled using the `elasticBeamColumn` element.

*   **Rationale:** While nonlinear fiber sections (`nonlinearBeamColumn`) provide higher accuracy for post-yield behavior, they are computationally expensive for the tens of thousands of iterations required by the NSGA-II algorithm.
*   **Methodology:** The optimization process relies on **Linear Elastic Analysis with Stiffness Modifiers** (if applied) to determine member forces ($M_u, V_u, P_u$). These demands are then checked against capacity limits ($DCR \le 1.0$) calculated based on ACI 318-19, which implicitly accounts for material nonlinearity at the ultimate limit state.

### 2.2 Geometric Transformation (P-Delta)
To account for second-order effects (geometric nonlinearity), the **P-Delta geometric transformation** is applied to all vertical members (columns).

*   **Implementation:**
    ```python
    # src/modeling.py
    ops.geomTransf('PDelta', 1, 1, 0, 0) # For Columns
    ops.geomTransf('PDelta', 2, 0, 1, 0) # For Columns (Rotated)
    ops.geomTransf('Linear', 3, 0, 0, 1) # For Beams (Gravity only usually, but PDelta safe)
    ```
*   **Note:** Previous versions used `Linear` transformation. This has been upgraded to `PDelta` in the revised manuscript to strictly adhere to stability analysis requirements.

## 3. Material Properties
Since the analysis is elastic, material behavior is defined by elastic constants rather than stress-strain curves.

*   **Concrete:**
    *   Elastic Modulus ($E_c$): $2.5791 \times 10^7$ kPa (approx. 25.8 GPa, corresponding to $f_{ck} \approx 27$ MPa).
    *   Poisson's Ratio ($\nu$): 0.167.
    *   Shear Modulus ($G_c$): Calculated as $E_c / (2(1+\nu))$.

*   **Section Properties:**
    *   Cross-sectional area ($A$), Moments of Inertia ($I_z, I_y$), and Torsional Constant ($J$) are pre-calculated based on gross concrete sections ($I_g$).
    *   *Self-criticism/Future Work:* To fully comply with ACI 318 for drift checks, cracked section properties ($0.7I_g$ for columns, $0.35I_g$ for beams) should ideally be implemented. Currently, the code uses gross properties.

## 4. Loading & Analysis Procedure

### 4.1 Load Application
Loads are applied in static load patterns:
1.  **Gravity Loads:** Applied as `eleLoad` (uniformly distributed load) on beams.
    *   **Dead Load (DL):** Self-weight (calculated automatically) + Superimposed DL.
    *   **Live Load (LL):** Applied in checkerboard patterns (per floor) to induce maximum moments.
2.  **Lateral Loads:** Applied as `load` (nodal forces) at column-beam joints.
    *   **Wind (W):** ASCE 7 based distribution.
    *   **Seismic (E):** Equivalent Lateral Force (ELF) procedure. Vertical distribution follows an inverted triangle pattern.

### 4.2 Analysis Algorithm
For each individual in the population, the following sequence is executed:
1.  **Model Build:** Nodes and elements are generated based on the chromosome's section IDs.
2.  **System Setup:**
    ```python
    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ```
3.  **Execution:** `ops.analyze(1)` is called for each load combination.
4.  **Response Retrieval:** Nodal displacements and element forces are recorded.
5.  **Reset:** `ops.reset()` is used between load cases (or `ops.wipe()` for new individuals) to ensure independence.

## 5. Addressing Reviewer Comments

| Reviewer Comment | Original Model | Revised Model | Justification |
| :--- | :--- | :--- | :--- |
| **"Nonlinear analysis missing"** | Elastic | Elastic | Optimization speed trade-off; valid for DCR-based design. |
| **"P-Delta effects?"** | No (Linear) | **Yes (P-Delta)** | Included to capture stability effects accurately. |
| **"Boundary conditions?"** | Fixed | Fixed | Assumption of rigid foundation. |
| **"Element Type?"** | Not specified | `elasticBeamColumn` | Explicitly documented for reproducibility. |

---
**Code Reference:** `src/modeling.py`

```