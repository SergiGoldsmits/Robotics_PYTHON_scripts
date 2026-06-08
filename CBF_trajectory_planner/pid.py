#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from joint_space_designer_analytical import FR3PlanarSystem


# ══════════════════════════════════════════════════════════════════════════════
#  Pinocchio-based CBF functions — exact geometry, no planar approximation
# ══════════════════════════════════════════════════════════════════════════════

def cbf_h_pinocchio(q_full, model, data, obs_center, obs_radius,
                    link_radii=None):
    """h(q) using exact Pinocchio FK. Checks shoulder, elbow, wrist in XZ."""
    if link_radii is None:
        link_radii = [0.05, 0.05, 0.02]

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)

    frame_pairs = [
        ('fr3_link1', 'fr3_link3'),
        ('fr3_link3', 'fr3_link5'),
        ('fr3_link5', 'fr3_link8'),
    ]

    obs      = np.array(obs_center)
    min_dist = float('inf')

    for idx, (f1, f2) in enumerate(frame_pairs):
        p1   = data.oMf[model.getFrameId(f1)].translation[[0, 2]]
        p2   = data.oMf[model.getFrameId(f2)].translation[[0, 2]]
        v    = p2 - p1
        w    = obs - p1
        t    = np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-9), 0.0, 1.0)
        dist = np.linalg.norm(w - t * v) - (obs_radius + link_radii[idx])
        if dist < min_dist:
            min_dist = dist

    return min_dist


def cbf_gradient_pinocchio(q_full, model, data, obs_center, obs_radius,
                            active_indices, link_radii=None, eps=0.01):
    """Numerical gradient of h w.r.t. active joints. Returns (h, grad_h)."""
    h0   = cbf_h_pinocchio(q_full, model, data, obs_center, obs_radius, link_radii)
    grad = np.zeros(len(active_indices))
    for i, idx in enumerate(active_indices):
        qp      = q_full.copy(); qp[idx] += eps
        qm      = q_full.copy(); qm[idx] -= eps
        h_plus  = cbf_h_pinocchio(qp, model, data, obs_center, obs_radius, link_radii)
        h_minus = cbf_h_pinocchio(qm, model, data, obs_center, obs_radius, link_radii)
        grad[i] = (h_plus - h_minus) / (2.0 * eps)
    return h0, grad


def cbf_filter_pinocchio(q_full, dq_des_planar, model, data,
                          obs_center, obs_radius, active_indices,
                          alpha=0.5, link_radii=None):
    """CBF filter — enforces grad_h . dq >= -alpha * h(q). KKT solution."""
    h, grad_h = cbf_gradient_pinocchio(
        q_full, model, data, obs_center, obs_radius,
        active_indices, link_radii)

    cbf_val = np.dot(grad_h, dq_des_planar) + alpha * h

    if cbf_val >= 0.0:
        return dq_des_planar, h

    correction = cbf_val / (np.dot(grad_h, grad_h) + 1e-9)
    dq_safe    = dq_des_planar - correction * grad_h
    return dq_safe, h


# ══════════════════════════════════════════════════════════════════════════════
#  Controller node
# ══════════════════════════════════════════════════════════════════════════════

class CartesianPIDController(Node):
    def __init__(self):
        print("NODE STARTED")
        super().__init__('cartesian_pid_controller')

        # ── Pinocchio model ───────────────────────────────────────────────────
        urdf_path = '/ros2_ws/src/libfranka/test/fr3.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()

        # ── End-effector frame ────────────────────────────────────────────────
        self.ee_frame_id = self.model.getFrameId('link8')
        if self.ee_frame_id < 0:
            raise RuntimeError("link8 not found in URDF")
        self.get_logger().info("Using EE frame: link8")

        # ── CBF parameters ────────────────────────────────────────────────────
        # UPDATE obs_center at the lab after measuring ball position.
        self.obs_center    = np.array([0.5, 0.5])   # [x, z] in metres
        self.obs_radius    = 0.13    # ball radius ~0.11m + 2cm safety margin
        self.link_radii    = [0.05, 0.05, 0.02]
        self.cbf_alpha     = 0.5     # start conservative
        self.active_joints = [1, 3, 5]   # fr3_joint2, fr3_joint4, fr3_joint6

        # ── Robot state ───────────────────────────────────────────────────────
        self.q            = np.zeros(7)
        self.q_prev       = np.zeros(7)
        self.q_dot        = np.zeros(7)
        self.state_received = False
        self.x_des        = None

        # ── Safety / filtering (unchanged from original) ──────────────────────
        self.dq_prev  = np.zeros(7)
        self.alpha    = 0.15     # low-pass filter coefficient
        self.max_dq   = 0.15     # rad/s velocity limit
        self.max_ddq  = 0.4      # rad/s² acceleration limit

        # ── Control gains (unchanged from original) ───────────────────────────
        self.Kp   = 0.8
        self.Kd   = 0.25
        self.Kp_r = 0.0
        self.Kd_r = 0.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/cartesian_pid/target_pose',
            self.target_callback, 10)
        self.create_subscription(
            JointState, '/franka/joint_states',
            self.joint_state_callback, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands', 10)

        # ── Loop timing ───────────────────────────────────────────────────────
        self.dt    = 0.01
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'CartesianPIDController started  (Pinocchio CBF active)\n'
            f'Obstacle: centre={self.obs_center.tolist()}  '
            f'radius={self.obs_radius} m  alpha={self.cbf_alpha}')

    # ── Target callback ───────────────────────────────────────────────────────
    def target_callback(self, msg):
        q = np.array([msg.pose.orientation.x,
                      msg.pose.orientation.y,
                      msg.pose.orientation.z,
                      msg.pose.orientation.w])
        q = q / np.linalg.norm(q)
        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
        t = np.array([msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z])
        self.x_des = pin.SE3(R, t)

    # ── Joint state callback ──────────────────────────────────────────────────
    def joint_state_callback(self, msg):
        joint_order = [f'fr3_joint{i}' for i in range(1, 8)]
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for i, j in enumerate(joint_order):
            if j in name_to_idx:
                self.q[i] = msg.position[name_to_idx[j]]
        self.state_received = True

    # ── Control loop — 100 Hz ─────────────────────────────────────────────────
    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        # Step 1: FK
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        x_current = self.data.oMf[self.ee_frame_id]

        # Step 2: Cartesian error
        e_t   = self.x_des.translation - x_current.translation
        R_err = self.x_des.rotation @ x_current.rotation.T
        e_r   = pin.log3(R_err)

        # Step 3: Jacobian
        J     = pin.computeFrameJacobian(
                    self.model, self.data, self.q,
                    self.ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos = J[:3, :]

        # Step 4: velocity estimate
        self.q_dot = (self.q - self.q_prev) / self.dt
        self.q_prev = self.q.copy()
        x_dot = J_pos @ self.q_dot

        # Step 5: PD control law (unchanged)
        v_t = self.Kp * e_t - self.Kd * x_dot
        v_r = self.Kp_r * e_r   # = 0

        # Step 6: IK (unchanged)
        lam   = 1e-4
        JJT   = J_pos @ J_pos.T + lam * np.eye(3)
        J_pinv = J_pos.T @ np.linalg.inv(JJT)
        dq_raw = J_pinv @ v_t

        # Step 7: CBF filter — Pinocchio exact geometry ── ADDED
        dq_planar           = dq_raw[self.active_joints]
        dq_safe_p, h        = cbf_filter_pinocchio(
                                  self.q, dq_planar,
                                  self.model, self.data,
                                  self.obs_center, self.obs_radius,
                                  self.active_joints, self.cbf_alpha,
                                  self.link_radii)
        dq_raw[self.active_joints] = dq_safe_p

        corr = np.linalg.norm(dq_safe_p - dq_planar)
        if corr > 1e-3:
            self.get_logger().warn(
                f'CBF active  h={h:.3f} m  correction={corr:.4f} rad/s')

        # Step 8: velocity saturation (unchanged)
        dq_raw = np.clip(dq_raw, -self.max_dq, self.max_dq)

        # Step 9: acceleration limiting (unchanged)
        dq_diff   = dq_raw - self.dq_prev
        max_step  = self.max_ddq * self.dt
        dq_diff   = np.clip(dq_diff, -max_step, max_step)
        dq_limited = self.dq_prev + dq_diff

        # Step 10: low-pass filter (unchanged)
        dq_cmd    = self.alpha * dq_limited + (1 - self.alpha) * self.dq_prev
        self.dq_prev = dq_cmd.copy()

        # Step 11: publish
        msg      = Float64MultiArray()
        msg.data = dq_cmd.tolist()
        self.vel_pub.publish(msg)


def main():
    rclpy.init()
    node = CartesianPIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop      = Float64MultiArray()
        stop.data = [0.0] * 7
        node.vel_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
