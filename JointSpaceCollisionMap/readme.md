# FR3 Planar Arm: Collision Mapping & CBF Control

This repository houses the research codebase for the 4-link planar FR3 manipulator. It bridges the gap between formal geometric theory and numerical safety validation.

## File Architecture
* **`joint_space_designer.py`**: The **Numerical Validation** suite. This script treats the robot as a physical system with link thicknesses. It performs dense point-cloud sampling to map the configuration space ($C_{space}$) and handles self-collision detection to provide a "ground truth" workspace.
* **`joint_space_designer_analytical.py`**: The **Theoretical Design** suite. This script implements the analytical collision mapping derived from **Capisani et al.**, providing the formal geometric boundary required for stability proofs in your thesis.

## Theoretical Background & Methodology

### 1. Theoretical Mapping (Capisani et al.)
The analytical approach relies on the geometric theorem proposed by **Capisani et al.** to define the "No-Go" zone. It models the robot as a kinematic chain of segments and calculates the forbidden angular intervals ($\delta_i$) based on the proximity to obstacles. 
* **Purpose:** This provides the rigorous mathematical foundation for the safety set $\mathcal{C}_{free}$.


### 2. Numerical Methods
To validate the theoretical model, we employ a **distance-to-segment metric**. The numerical method calculates the minimum distance $d(S_i, O)$ between the robot links ($S_i$) and the obstacle ($O$). 
* **Purpose:** This ensures that real-world physical constraints—such as link radius and potential self-intersections—are accounted for, effectively creating a "Safety Benchmark."

### 3. Control Synthesis (CBF)
Both files integrate a **differentiable distance function** $h(q) = \text{dist}(S, O) - R$. This analytical form is essential for the Control Barrier Function (CBF) controller, as it allows the optimizer to compute the gradient $\nabla_q h(q)$ in real-time to keep the arm within the safe manifold.

## Usage
1. **Data Generation:** Run `joint_space_designer.py` to generate the `thesis_data/` repository.
2. **Theoretical Analysis:** Run `joint_space_designer_analytical.py` to verify the geometric boundaries against the theoretical theorem.

---
*Developed for research on robotic manipulation safety and control barrier functions.*
