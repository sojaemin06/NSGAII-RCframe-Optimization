# Figure Captions and Summaries (Academic Style)

## Figure 1. Structural-to-Genetic Encoding Framework for 3D RC Frames

**Figure 1: Overview of the Structural-to-Genetic Encoding Framework.**
Given the physical building structure and the defined grouping strategy, we first apply a **Spatial Retrieval Phase** to classify structural members based on their planar positions (Corner, Edge, Interior) and vertical levels. We then synthesize a **Genetic Vector** ($\mathbf{X}$) consisting of discrete section indices, binary rotation flags ($R_{dir}$), and reinforcement attributes. As illustrated by the mapping arrows, each structural group is precisely encoded into a unified chromosome, providing a concrete foundation for the multi-objective optimization process. This encoding mechanism transforms raw 3D architectural data into an executable genetic string while maintaining structural integrity.

---

## Figure 2. Integrated 3D RC Frame Optimization Framework

**Figure 2: Overview of the Integrated 3D RC Frame Optimization Framework.**
Given the user-defined constraints and architectural intent, we first construct a **Project-Specific Section Catalog** through a Pareto-based reduction engine to ensure computational tractability. We then use an **Iterative Optimization Loop** (consisting of the Model Planner, Performance Visualizer, and Multi-Objective Critic) to decode genetic information into a high-fidelity 3D finite element model in OpenSees. The Critic agent evaluates the structural responses under 38+ load combinations, calculating factual misalignments (violations) and performance metrics ($f_1, f_2$). Through multi-generation refinements using the NSGA-II algorithm, the framework produces a set of non-dominated Pareto-optimal solutions for informed engineering decision-making.
