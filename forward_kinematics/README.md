# Universal Serial Manipulator Forward Kinematics Solver

A robust, self-contained Python implementation for calculating the forward kinematics of serial-chain robotic manipulators. This script provides a unified interface that reconciles the two most common kinematic frameworks used in industrial robotics and academic research: **Classic (Standard) Denavit-Hartenberg** and **Craig's Modified Denavit-Hartenberg (MDH)**.

The solver features automated joint coordinate offset handling and has been cross-verified analytically using the Product of Exponentials (PoE) framework.

## Features

- **Dual-Convention Engine:** Switch seamlessly between Standard DH and Modified DH algorithms using a single function call.
- **Dynamic Joint Offset Calibration:** Implements $\theta_i = q_i + \theta_{\text{offset}}$ structurally to handle non-flagpole zero configurations natively.
- **Robust Attribute Safeguards:** Uses defensive dictionary parsing (`.get()`) to allow optional keys (like missing offsets) to default gracefully without script crashes.
- **Array Dimension Verification:** Built-in shape checking prevents mismatched joint-vector/table inputs prior to matrix execution.

---

## Kinematic Convention Architecture

The repository handles the structural shift in frame tracking between distal and proximal topologies:

### 1. Standard DH (Classic / Distal)
Frame $i$ is attached to the *end* of link $i$ (at joint axis $i+1$). Transformation sequence:
$$A_i = R_z(\theta_i) \cdot T_z(d_i) \cdot T_x(a_i) \cdot R_x(\alpha_i)$$

### 2. Modified DH (Craig / Proximal)
Frame $i$ is attached to the *base* of link $i$ (at joint axis $i$). Transformation sequence:
$$A_i = R_x(\alpha_{i-1}) \cdot T_x(a_{i-1}) \cdot R_z(\theta_i) \cdot T_z(d_i)$$

---

## Code Structure

- `get_standard_dh_matrix(...)`: Computes individual classic $4 \times 4$ transformation matrices.
- `get_modified_dh_matrix(...)`: Computes individual Craig-convention $4 \times 4$ transformation matrices.
- `forward_kinematics(q, dh_params, DH_type, world_frame)`: Loops through the designated parameter array, extracts values, applies joint angle offsets, and post-multiplies matrices down the chain.

---

## Quick Start & Usage

### Prerequisites
- Python 3.x
- NumPy

### Run the Verification Example
The script contains a built-in verification sequence running a nominal **UR5 manipulator** through a non-trivial joint configuration ($q = [0, -\pi/2, -\pi/2, 0, 0, 0]$):

```bash
python3 Forward_Kinematics.py
