# Methodology: The Integrated 3D RC Frame Optimization Framework

In this section, we present the architecture of the 3D RC frame optimization framework, a reference-driven agentic system for multi-objective structural design. As illustrated in Figure 2, the framework orchestrates a collaborative team of four specialized modules—Catalog Retriever, Model Planner, Performance Visualizer, and Multi-Objective Critic—to transform architectural requirements into Pareto-optimal structural solutions.

### Catalog Retriever (The Database "Retriever")

Given the user-defined constraints $S$ and the desired search space $C$, the Catalog Retriever identifies a subset of most efficient RC sections $\mathcal{D} = \{D_n\}_{n=1}^{N}$ from the initial candidate pool $\mathcal{P}$. To ensure computational tractability, we adopt a Pareto-based reduction approach:
$$
\mathcal{D} = \text{Module}_{\text{Ret}} \left( \mathcal{P}, \text{Cost}, \text{Performance} \right)
$$
Specifically, the Retriever is instructed to filter out dominated sections whose structural capacity (e.g., $P-M$ volume, $M_n$) is inferior to others with similar or lower costs. By explicitly selecting a refined catalog $\mathcal{D}$ that best matches the project’s economic and safety requirements, the Retriever provides a concrete foundation for the subsequent evolutionary search.

### Model Planner (The Decoding "Planner")

The Model Planner serves as the cognitive core for translating genetic information into physical reality. It takes the chromosome vector $\mathbf{X}$ and the retrieved catalog $\mathcal{D}$ as inputs. By performing in-context mapping from the indices in $\mathbf{X}$ to the attributes in $\mathcal{D}$, the Planner translates the abstract vector into a comprehensive 3D finite element model $M$:
$$
M = \text{Module}_{\text{Plan}}(\mathbf{X}, \mathcal{D}, \text{GroupingStrategy})
$$
The Planner ensures that section properties, material strengths, and column orientations are precisely assigned to each member group, while automatically generating the global topology including nodes and rigid diaphragms.

### Performance Visualizer (The Structural "Visualizer")

After receiving the 3D model $M$, the Performance Visualizer executes the automated structural analysis loop. The Visualizer leverages a high-fidelity finite element engine (OpenSees) to transform the model into visual and numerical structural responses. In each iteration $t$, given a model $M_t$, the Visualizer generates the structural response set $\mathcal{R}_t$:
$$
\mathcal{R}_t = \text{Engine}_{\text{Analysis}}(M_t, \{\text{LoadCombinations}\})
$$
where the analysis covers 38+ load cases to capture the full spectrum of gravity, seismic, and wind effects. This module converts the static design into dynamic performance metrics, providing the necessary data for objective evaluation.

### Multi-Objective Critic (The Selection "Critic")

The Multi-Objective Critic forms a closed-loop refinement mechanism with the Evolutionary Engine by closely examining the structural response $\mathcal{R}_t$ and providing fitness feedback. Upon receiving $\mathcal{R}_t$ at generation $t$, the Critic inspects it against the original safety criteria $(S, C)$ to identify factual misalignments or constraint violations. It calculates the objective vector $\mathbf{f} = [f_1, f_2]^T$ and the total violation $\Phi(\mathbf{X})$:
$$
\mathbf{f}, \Phi(\mathbf{X}) = \text{Module}_{\text{Critic}}(\mathcal{R}_t, \text{Criteria})
$$
Based on these findings, the Critic applies the *Constrained Dominance Principle* to prioritize feasible designs. The process iterates for $T=100$ generations, with the final output being the set of non-dominated Pareto-optimal solutions. This iterative refinement process ensures that the final design meets the high standards required for both economic efficiency and structural safety.

### Extension to Environmental Impact

The framework extends to environmental assessment by integrating a CO2 Emission Stylist within the Multi-Objective Critic. For sustainable precision, the Critic converts material volumes into carbon footprints: $E_{CO2} = \text{Module}_{\text{Eco}}(V_{conc}, W_{steel})$. This ensures that the Pareto front reflects the trade-off not only between cost and safety but also between structural performance and environmental sustainability.
