#!/usr/bin/env python3
"""
joint_space_designer_analytical.py  —  Script 1
FR3 Planar Configuration Space Mapper + CBF Safety Functions

Provides:
  FR3PlanarSystem  — robot model, kinematics, CBF barrier function, filter
  Standalone visualisation when run directly.

Paper references:
  [1] Capisani et al. "Obstacle Modelling Oriented to Safe Motion Planning
      and Control for Planar Rigid Robot Manipulators", JIRS 2013.
  [2] Ferraguti et al. "Safety and Efficiency in Robotics:
      The Control Barrier Functions Approach", IEEE RA-M 2022.

Notes:
  - Planar model in XZ vertical plane using FR3 joints 2, 4, 6
    (indices 1, 3, 5 in the 7-DOF joint vector) — the XZ flex joints.
  - Base column L[0]=0.333m is a fixed vertical segment (non-rotating).
  - Link lengths from Pinocchio FR3 frame origins at neutral pose:
      L[0]=0.333  fr3_link0 → fr3_link1  base height
      L[1]=0.316  fr3_link1 → fr3_link3  shoulder
      L[2]=0.384  fr3_link3 → fr3_link5  elbow
      L[3]=0.107  fr3_link5 → fr3_link8  wrist
  - Simple planar FK — all links assumed to point upward from joint origins.
    Known approximation: real FR3 has ~8cm x offset at EE due to wrist
    geometry. This model is used for C-space visualisation and Capisani
    verification only. Script 3 uses Pinocchio for exact CBF enforcement.
  - Link radii added to obstacle radius for collision margin (Capisani §4.2).
  - CBF uses kinematic single-integrator model: q̇ = u  (Ferraguti Eq.12).
  - Gradient is numerical (central differences) — equivalent to Ferraguti Eq.11.
  - Closed-form KKT projection — single constraint, no QP solver needed.
  - Known limitation: purely kinematic CBF — safety depends on conservative α
    (Ferraguti Fig.8). Energy-based extension is future work.
"""

import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  Robot model + CBF
# ══════════════════════════════════════════════════════════════════════════════

class FR3PlanarSystem:
    """
    Simple planar 3-link model of the FR3 in the XZ vertical plane.
    Used for C-space visualisation and Capisani analytical verification.
    Active joints: 2, 4, 6  (indices 1, 3, 5 in the 7-DOF joint vector).

    NOT used for online CBF safety enforcement in Script 3 —
    Script 3 uses Pinocchio directly for exact geometry.
    """

    def __init__(self, obs_center=None, obs_radius=0.085):
        # ── Link lengths [m] from Pinocchio frame origins ─────────────────────
        self.L = [0.333, 0.316, 0.384, 0.107]

        # ── Obstacle ──────────────────────────────────────────────────────────
        self.obs_center = np.array(obs_center if obs_center is not None
                                   else [0.45, 0.25])
        self.obs_radius = obs_radius

        # ── Physical link radii [m] (Capisani §4.2 enlargement) ──────────────
        self.link_radii = [0.05, 0.05, 0.02]   # shoulder, elbow, wrist

    # ── Forward kinematics ────────────────────────────────────────────────────
    def get_kinematics(self, q_deg):
        """
        Simple planar FK — all links point upward from joint origins.
        Input: q_deg = [q2, q4, q6] in degrees (XZ flex joints).

        Returns 5x2 array:
          [0]  base bottom   (0, 0)
          [1]  fr3_link1     (0, L[0])  — joint 2 origin
          [2]  fr3_link3     position   — joint 4 origin
          [3]  fr3_link5     position   — joint 6 origin
          [4]  fr3_link8     end effector
        """
        q = np.radians(q_deg)
        # Absolute angles: Q_i = sum q_j  (Capisani Eq.1)
        Q = np.array([q[0],
                      q[0] + q[1],
                      q[0] + q[1] + q[2]])

        pts  = [np.array([0.0, 0.0]),
                np.array([0.0, self.L[0]])]
        curr = pts[1].copy()
        for i in range(3):
            curr = curr + self.L[i + 1] * np.array([np.sin(Q[i]),
                                                      np.cos(Q[i])])
            pts.append(curr.copy())
        return np.array(pts)

    # ── CBF barrier function h(q) ─────────────────────────────────────────────
    def get_cbf_h(self, q_deg):
        """
        h(q) = min signed distance from any moving link to obstacle boundary.
        h > 0 safe,  h = 0 boundary,  h < 0 collision.
        Used for C-space visualisation and Capisani verification.
        Script 3 uses cbf_h_pinocchio() for safety-critical enforcement.
        """
        pts      = self.get_kinematics(q_deg)
        min_dist = float('inf')
        for i in range(3):
            p1, p2 = pts[i + 1], pts[i + 2]
            v    = p2 - p1
            w    = self.obs_center - p1
            t    = np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-9), 0.0, 1.0)
            dist = np.linalg.norm(w - t * v) - (self.obs_radius + self.link_radii[i])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # ── Numerical gradient ∇h(q) ──────────────────────────────────────────────
    def cbf_gradient(self, q_deg, eps=0.5):
        """Central differences gradient of h. Returns (h, grad_h)."""
        h0   = self.get_cbf_h(q_deg)
        grad = np.zeros(3)
        for i in range(3):
            qp = q_deg.copy(); qp[i] += eps
            qm = q_deg.copy(); qm[i] -= eps
            grad[i] = (self.get_cbf_h(qp) - self.get_cbf_h(qm)) / \
                      (2.0 * np.radians(eps))
        return h0, grad

    # ── CBF safety filter ─────────────────────────────────────────────────────
    def cbf_filter(self, q_deg, dq_des, alpha=1.0):
        """
        Minimum-intervention CBF filter (Ferraguti Eq.3, single constraint).
        Used by Script 2 for approximate waypoint planning.
        Script 3 uses cbf_filter_pinocchio() for exact safety enforcement.
        """
        h, grad_h = self.cbf_gradient(q_deg)
        cbf_val   = np.dot(grad_h, dq_des) + alpha * h
        if cbf_val >= 0.0:
            return dq_des, h
        correction = cbf_val / (np.dot(grad_h, grad_h) + 1e-9)
        dq_safe    = dq_des - correction * grad_h
        return dq_safe, h

    # ── Capisani analytical mapping (visualisation only) ──────────────────────
    def is_unsafe_paper_method(self, q_deg):
        """
        Capisani Theorem 3 / Eq.9 — analytical C-space forbidden region.
        Returns True if any link is in the forbidden angular interval.
        Used for visualisation and verification only — not for online CBF.
        """
        pts = self.get_kinematics(q_deg)
        q   = np.radians(q_deg)
        Q   = [q[0], q[0] + q[1], q[0] + q[1] + q[2]]

        for i in range(3):
            d = np.linalg.norm(self.obs_center - pts[i + 1])
            if d < 1e-9:
                return True

            Qw = np.arctan2(self.obs_center[0] - pts[i + 1][0],
                            self.obs_center[1] - pts[i + 1][1])
            l  = self.L[i + 1]
            R  = self.obs_radius+self.link_radii[i]

            if l ** 2 >= d ** 2 - R ** 2:                       # Case A
                delta = np.arcsin(np.clip(R / d, -1.0, 1.0))
            elif (d - R) ** 2 < l ** 2 <= d ** 2 - R ** 2:     # Case B
                delta = np.arccos(np.clip(
                    (l ** 2 + d ** 2 - R ** 2) / (2.0 * d * l), -1.0, 1.0))
            else:                                                 # Case C
                delta = 0.0

            angle_diff = abs((Q[i] - Qw + np.pi) % (2.0 * np.pi) - np.pi)
            if angle_diff < delta:
                return True

        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone visualisation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    robot = FR3PlanarSystem()

    # ── Plot 1: single configuration in workspace ─────────────────────────────
    pts = robot.get_kinematics([-45.0, 120.0, 90.0])
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(pts[:, 0], pts[:, 1], 'ko-', linewidth=3, markersize=6)
    ax1.add_patch(plt.Circle(robot.obs_center, robot.obs_radius,
                             color='red', alpha=0.5, label='Obstacle'))
    ax1.set_aspect('equal')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('z [m]')
    ax1.set_title('Operational Workspace — single configuration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Build C-space map ─────────────────────────────────────────────────────
    print("Building C-space map (res=50) ...")
    res     = 50
    q_range = np.linspace(-np.pi, np.pi, res)
    safe, unsafe = [], []

    for q1 in q_range:
        for q2 in q_range:
            for q3 in q_range:
                cfg = np.degrees([q1, q2, q3])
                if robot.is_unsafe_paper_method(cfg):
                    unsafe.append([q1, q2, q3])
                else:
                    safe.append([q1, q2, q3])

    safe   = np.array(safe)
    unsafe = np.array(unsafe)
    print(f"  Safe: {len(safe)}   Unsafe: {len(unsafe)}")

    # ── Plot 2: 3D C-space ────────────────────────────────────────────────────
    fig2 = plt.figure(figsize=(14, 6))

    ax2a = fig2.add_subplot(121, projection='3d')
    ax2a.scatter(safe[:, 0], safe[:, 1], safe[:, 2], s=0.5, c='#003399', alpha=0.2)
    ax2a.set_xlabel('$q_2$ [rad]')
    ax2a.set_ylabel('$q_4$ [rad]')
    ax2a.set_zlabel('$q_6$ [rad]')
    ax2a.set_title('Go Zone (Safe)')

    ax2b = fig2.add_subplot(122, projection='3d')
    ax2b.scatter(unsafe[:, 0], unsafe[:, 1], unsafe[:, 2], s=0.5, c='#CC0000', alpha=0.2)
    ax2b.set_xlabel('$q_2$ [rad]')
    ax2b.set_ylabel('$q_4$ [rad]')
    ax2b.set_zlabel('$q_6$ [rad]')
    ax2b.set_title('No-Go Zone (Forbidden)')

    # ── Plot 3: 2D slice q6=0 ─────────────────────────────────────────────────
    q6_fixed = 0.0
    tol      = (q_range[1] - q_range[0]) / 2.0
    mask_s   = np.abs(safe[:, 2]   - q6_fixed) < tol
    mask_u   = np.abs(unsafe[:, 2] - q6_fixed) < tol

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    if mask_s.any():
        ax3.scatter(safe[mask_s, 0], safe[mask_s, 1],
                    s=8, c='#003399', alpha=0.5, label='Safe')
    if mask_u.any():
        ax3.scatter(unsafe[mask_u, 0], unsafe[mask_u, 1],
                    s=8, c='#CC0000', alpha=0.5, label='Unsafe')
    ax3.set_xlabel('$q_2$ [rad]')
    ax3.set_ylabel('$q_4$ [rad]')
    ax3.set_title(f'C-Space Slice  (q6 = {np.degrees(q6_fixed):.0f}°)')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
