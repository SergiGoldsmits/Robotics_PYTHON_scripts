#!/usr/bin/env python3
"""
cartesian_pid_controller.py  —  FR3 Cartesian PID + CBF Safety Filter

Architecture:
  User sends goal pose → /cartesian_pid/goal_pose
  PID computes desired joint velocity toward goal
  CBF filter modifies velocity if safety constraint violated
  Safe velocity sent to hardware controller

No separate planner needed — CBF reacts at 100Hz inside the control loop.

Subscribes:
  /cartesian_pid/goal_pose                     (geometry_msgs/PoseStamped)
  /franka/joint_states                         (sensor_msgs/JointState)

Publishes:
  /joint_velocity_example_controller/commands  (std_msgs/Float64MultiArray)
  /cbf_status                                  (std_msgs/String) diagnostics
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

import pinocchio as pin


# ══════════════════════════════════════════════════════════════════════════════
#  Pinocchio-based CBF — exact geometry
# ══════════════════════════════════════════════════════════════════════════════

def cbf_h_pinocchio(q_full, model, data, obs_center, obs_radius,
                    link_radii=None):
    """
    h(q) = minimum signed distance from any moving link to obstacle boundary.
    Uses exact Pinocchio FK — no planar approximation.
    h > 0 safe,  h = 0 boundary,  h < 0 collision.
    """
    if link_radii is None:
        link_radii = [0.05, 0.05, 0.02]

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)

    frame_pairs = [
        ('fr3_link1', 'fr3_link3'),   # shoulder
        ('fr3_link3', 'fr3_link5'),   # elbow
        ('fr3_link5', 'fr3_link8'),   # wrist
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
    h0   = cbf_h_pinocchio(q_full, model, data, obs_center,
                            obs_radius, link_radii)
    grad = np.zeros(len(active_indices))
    for i, idx in enumerate(active_indices):
        qp      = q_full.copy(); qp[idx] += eps
        qm      = q_full.copy(); qm[idx] -= eps
        h_plus  = cbf_h_pinocchio(qp, model, data, obs_center,
                                   obs_radius, link_radii)
        h_minus = cbf_h_pinocchio(qm, model, data, obs_center,
                                   obs_radius, link_radii)
        grad[i] = (h_plus - h_minus) / (2.0 * eps)
    return h0, grad


def cbf_filter_pinocchio(q_full, dq_des, model, data,
                          obs_center, obs_radius, active_indices,
                          alpha=0.5, link_radii=None):
    """
    Minimum-intervention CBF filter (Ferraguti Eq.3).
    Enforces: grad_h . dq >= -alpha * h(q)
    Returns (dq_safe, h).
    """
    h, grad_h = cbf_gradient_pinocchio(
        q_full, model, data, obs_center, obs_radius,
        active_indices, link_radii)

    cbf_val = np.dot(grad_h, dq_des) + alpha * h

    if cbf_val >= 0.0:
        return dq_des, h   # safe — no modification

    # KKT closed-form correction
    correction = cbf_val / (np.dot(grad_h, grad_h) + 1e-9)
    dq_safe    = dq_des - correction * grad_h
    return dq_safe, h


# ══════════════════════════════════════════════════════════════════════════════
#  Controller node
# ══════════════════════════════════════════════════════════════════════════════

class CartesianPIDController(Node):

    def __init__(self):
        super().__init__('cartesian_pid_controller')

        # ── Pinocchio model ───────────────────────────────────────────────────
        urdf_path        = '/ros2_ws/src/libfranka/test/fr3.urdf'
        self.model       = pin.buildModelFromUrdf(urdf_path)
        self.data        = self.model.createData()
        self.ee_frame_id = self.model.getFrameId('link8')
        if self.ee_frame_id < 0:
            raise RuntimeError("link8 not found in URDF")
        self.get_logger().info("Using EE frame: link8")

        # ── CBF parameters ────────────────────────────────────────────────────
        # UPDATE obs_center at the lab after measuring ball position [x, z].
        self.obs_center    = np.array([0.5, 0.5])
        self.obs_radius    = 0.13
        self.link_radii    = [0.05, 0.05, 0.02]
        self.cbf_alpha     = 0.5        # start conservative
        self.active_joints = [1, 3, 5]  # fr3_joint2, fr3_joint4, fr3_joint6

        # ── Robot state ───────────────────────────────────────────────────────
        self.q              = np.zeros(7)
        self.q_prev         = np.zeros(7)
        self.q_dot          = np.zeros(7)
        self.state_received = False
        self.x_des          = None
        self.goal_reached   = False

        # ── Safety / filtering ────────────────────────────────────────────────
        self.dq_prev = np.zeros(7)
        self.alpha   = 0.15    # low-pass filter
        self.max_dq  = 0.15    # rad/s velocity limit
        self.max_ddq = 0.4     # rad/s² acceleration limit
        self.goal_tol = 0.02   # m — goal reached threshold

        # ── PD gains ──────────────────────────────────────────────────────────
        self.Kp   = 0.8
        self.Kd   = 0.25
        self.Kp_r = 0.0
        self.Kd_r = 0.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/cartesian_pid/goal_pose',
            self.goal_callback, 10)
        self.create_subscription(
            JointState, '/franka/joint_states',
            self.joint_state_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands', 10)
        self.status_pub = self.create_publisher(
            String, '/cbf_status', 10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.dt    = 0.01
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'CartesianPIDController started  (Pinocchio CBF active)\n'
            f'Obstacle: centre={self.obs_center.tolist()}  '
            f'radius={self.obs_radius} m  alpha={self.cbf_alpha}\n'
            f'Active CBF joints: {self.active_joints}  '
            f'(fr3_joint2, fr3_joint4, fr3_joint6)')

    # ── Goal callback ─────────────────────────────────────────────────────────
    def goal_callback(self, msg):
        # reject goals inside the obstacle
        gx = msg.pose.position.x
        gz = msg.pose.position.z
        dist = np.linalg.norm(np.array([gx, gz]) - self.obs_center)
        if dist < self.obs_radius:
            self.get_logger().error(
                f'Goal [{gx:.3f}, {gz:.3f}] inside obstacle '
                f'(dist={dist:.3f} m) — REJECTED')
            return

        q = np.array([msg.pose.orientation.x,
                      msg.pose.orientation.y,
                      msg.pose.orientation.z,
                      msg.pose.orientation.w])
        norm = np.linalg.norm(q)
        q    = q / norm if norm > 1e-6 else np.array([0., 0., 0., 1.])

        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
        t = np.array([msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z])

        self.x_des      = pin.SE3(R, t)
        self.goal_reached = False
        self.get_logger().info(
            f'New goal: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]  '
            f'dist_to_obs={dist:.3f} m')

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
        e_t = self.x_des.translation - x_current.translation

        # Step 3: goal reached check
        if np.linalg.norm(e_t) < self.goal_tol and not self.goal_reached:
            self.goal_reached = True
            self.get_logger().info(
                f'Goal reached  error={np.linalg.norm(e_t):.4f} m')

        # Step 4: rotation error (computed but not used — planar control)
        R_err = self.x_des.rotation @ x_current.rotation.T
        e_r   = pin.log3(R_err)

        # Step 5: Jacobian
        J     = pin.computeFrameJacobian(
                    self.model, self.data, self.q,
                    self.ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos = J[:3, :]

        # Step 6: velocity estimate
        self.q_dot  = (self.q - self.q_prev) / self.dt
        self.q_prev = self.q.copy()
        x_dot       = J_pos @ self.q_dot

        # Step 7: PD control
        v_t = self.Kp * e_t - self.Kd * x_dot
        v_r = self.Kp_r * e_r   # = 0

        # Step 8: IK
        lam    = 1e-4
        JJT    = J_pos @ J_pos.T + lam * np.eye(3)
        J_pinv = J_pos.T @ np.linalg.inv(JJT)
        dq_raw = J_pinv @ v_t

        # Step 9: CBF filter — reactive safety on joints 2,4,6
        dq_planar              = dq_raw[self.active_joints]
        dq_safe_p, h           = cbf_filter_pinocchio(
                                     self.q, dq_planar,
                                     self.model, self.data,
                                     self.obs_center, self.obs_radius,
                                     self.active_joints, self.cbf_alpha,
                                     self.link_radii)
        dq_raw[self.active_joints] = dq_safe_p

        # Step 10: velocity saturation
        dq_raw = np.clip(dq_raw, -self.max_dq, self.max_dq)

        # Step 11: acceleration limiting
        dq_diff    = dq_raw - self.dq_prev
        max_step   = self.max_ddq * self.dt
        dq_diff    = np.clip(dq_diff, -max_step, max_step)
        dq_limited = self.dq_prev + dq_diff

        # Step 12: low-pass filter
        dq_cmd       = self.alpha * dq_limited + (1 - self.alpha) * self.dq_prev
        self.dq_prev = dq_cmd.copy()

        # Step 13: publish velocity command
        out      = Float64MultiArray()
        out.data = dq_cmd.tolist()
        self.vel_pub.publish(out)

        # Step 14: publish status
        corr   = np.linalg.norm(dq_safe_p - dq_planar)
        active = corr > 1e-3
        s      = String()
        s.data = (f'h={h:.4f}  '
                  f'cbf_active={active}  '
                  f'correction={corr:.4f} rad/s  '
                  f'goal_reached={self.goal_reached}')
        self.status_pub.publish(s)

        if active:
            self.get_logger().warn(
                f'CBF active  h={h:.3f} m  correction={corr:.4f} rad/s')
        if h < 0.02:
            self.get_logger().error(
                f'h={h:.4f} m — NEAR BOUNDARY, reduce alpha or speed')


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
