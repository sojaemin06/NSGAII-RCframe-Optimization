# Methodology: The Structural-to-Genetic Encoding Framework

In this section, we present the architecture of the encoding framework designed for 3D RC frame optimization. As illustrated in Figure 1, the framework orchestrates a collaborative sequence of four specialized modules—Spatial Analyzer, Attribute Mapper, Constraint Aligner, and Vector Synthesizer—to transform raw 3D structural data into an optimized genetic chromosome.

### Spatial Analyzer (The Structural "Retriever")

Given the global nodal coordinates $V$ and element connectivity $E$, the Spatial Analyzer identifies the most relevant structural roles for each member. As defined in Section 2.1, members are classified into groups $\mathcal{G} = \{G_n\}_{n=1}^{N}$ based on their boundary conditions and tributary areas. To leverage the geometric symmetry of the building, we adopt a spatial retrieval approach:
$$
\mathcal{G} = 	ext{Module}_{	ext{Space}} \left( V, E, 	ext{Floor}_{height} ight)
$$
Specifically, the analyzer ranks members by matching their planar coordinates (e.g., Corner, Edge, Interior) and vertical levels. By explicitly reasoning the structural category of each member $M_i$, the analyzer provides a concrete foundation for assigning discrete section properties in the subsequent stages.

### Attribute Mapper (The Design "Planner")

The Attribute Mapper serves as the cognitive core of the encoding process. It takes the classified groups $\mathcal{G}$ and the project-specific database $\mathcal{D}$ as inputs. By mapping structural demands to available catalog indices, the Mapper translates the raw physical requirements into a detailed set of discrete design variables $P$:
$$
P = 	ext{Module}_{	ext{Map}}(\mathcal{G}, \mathcal{D}, 	ext{SearchSpace})
$$
This module ensures that each column and beam group is assigned a unique index from the filtered Pareto-optimal catalog, providing the link between the continuous design space and the discrete search space.

### Constraint Aligner (The Design "Stylist")

To ensure the encoded design adheres to the practical standards of RC construction, the Constraint Aligner acts as a stylistic consultant. A primary challenge lies in ensuring directional stiffness, as traditional 2D optimization often neglects column orientation. The Aligner synthesizes an *Alignment Guideline* covering key dimensions such as principal axes and rotation binary flags. Armed with this guideline, the Aligner refines the initial mapping $P$ into a stylistically and structurally optimized version $P^*$:
$$
P^* = 	ext{Module}_{	ext{Align}}(P, \{R_{dir}\})
$$
This ensures that the final chromosome not only represents section sizes but also the optimal orientation of each column group to maximize lateral resistance.

### Vector Synthesizer (The "Visualizer")

After receiving the optimized attributes $P^*$, the Vector Synthesizer assembles the final genetic chromosome $\mathbf{X}$. This module leverages a vectorization process to transform the multi-dimensional design data into a unified, executable string:
$$
\mathbf{X} = 	ext{Module}_{	ext{Synth}}( \{C_{id,i}\}, \{R_{dir,i}\}, \{B_{id,j}\} )
$$
The final output $\mathbf{X}$ is a concatenated vector where the initial segments represent section indices and rotation flags for columns, followed by beam section indices. This structured representation ensures that the NSGA-II algorithm can effectively perform genetic operations while maintaining the physical integrity of the 3D frame.

### Extension to Optimization Evaluation (The "Critic")

The framework extends to the evaluation phase by invoking the "Critic" mechanism (Structural Analysis Engine). Upon receiving the generated chromosome $\mathbf{X}$, the Critic (OpenSees) inspects it against the source context (Load Combinations) to identify structural misalignments or constraint violations. It then provides feedback in the form of objective functions $f_1, f_2$ and violation penalties $\Phi(\mathbf{X})$, which drive the iterative refinement process of the genetic algorithm for $T$ generations.
