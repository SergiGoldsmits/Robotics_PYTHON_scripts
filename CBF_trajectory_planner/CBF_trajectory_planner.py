#!/usr/bin/env python3
"""
cbf_trajectory_planner.py  —  Script 2
CBF-based Trajectory Planner ROS2 Node for FR3

Pipeline:
  /cbf_planner/goal_pose  →  Script 2  →  /cartesian_pid/target_pose
                                               ↓
                                           Script 3
                                               ↓
                                  /joint_velocity_example_controller/commands
                                               ↓
                                           FR3 robot

Subscribes:
  /cbf_planner/goal_pose   (geometry_msgs/PoseStamped)
  /joint_states            (sensor_msgs/JointState)

Publishes:
  /cartesian_pid/target_pose  (geometry_msgs/PoseStamped)  — waypoint stream
  /cbf_planner/status         (std_msgs/String)            — diagnostics
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from joint_space_designer_analytical import FR3PlanarSystem


class CBFTrajectoryPlanner(Node):

    def __init__(self):
        super().__init__('cbf_trajectory_planner')

        # ── CBF robot model ───────────────────────────────────────────────────
        # UPDATE obs_center at the lab after measuring ball position from
        # robot base origin with a tape measure.
        self.robot = FR3PlanarSystem(
            obs_center=[0.3, 0.6],
            obs_radius=0.13          # ball radius ~0.11m + 2cm safety margin
        )

        # ── Parameters ────────────────────────────────────────────────────────
        self.alpha    = 0.5    # CBF gain — start conservative, increase empirically
        self.Kp       = 0.5    # proportional gain: position error → Cartesian vel
        self.dq_max   = 0.1    # rad/s — velocity saturation
        self.dt       = 0.05   # s — planning rate (20 Hz)
        self.goal_tol = 0.02   # m — goal reached threshold

        # ── State ─────────────────────────────────────────────────────────────
        self.q              = np.zeros(7)
        self.state_received = False
        self.goal_xz        = None
        self.goal_pose_msg  = None
        self.goal_reached   = True    # start idle

        # ── ROS2 interfaces ───────────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/cbf_planner/goal_pose',
                                 self.goal_callback, 10)
        self.create_subscription(JointState, '/joint_states',
                                 self.joint_state_callback, 10)

        self.waypoint_pub = self.create_publisher(
            PoseStamped, '/cartesian_pid/target_pose', 10)
        self.status_pub   = self.create_publisher(
            String, '/cbf_planner/status', 10)

        self.create_timer(self.dt, self.planning_loop)

        self.get_logger().info('CBFTrajectoryPlanner started')
        self.get_logger().info(
            f'Obstacle: centre={self.robot.obs_center.tolist()}  '
            f'radius={self.robot.obs_radius} m')

    # ── Goal callback ─────────────────────────────────────────────────────────
    def goal_callback(self, msg):
        gx = msg.pose.position.x
        gz = msg.pose.position.z
        goal_candidate = np.array([gx, gz])

        # Reject goals inside the obstacle
        dist_to_obs = np.linalg.norm(goal_candidate - self.robot.obs_center)
        if dist_to_obs < self.robot.obs_radius:
            self.get_logger().error(
                f'Goal [{gx:.3f}, {gz:.3f}] is inside obstacle '
                f'(dist={dist_to_obs:.3f} m < radius={self.robot.obs_radius} m) '
                f'— REJECTED')
            return

        self.goal_xz      = goal_candidate
        self.goal_pose_msg = msg
        self.goal_reached  = False

        # Warn if current configuration is already unsafe
        if self.state_received:
            q_deg = np.degrees(self.q[[1, 3, 5]])
            h     = self.robot.get_cbf_h(q_deg)
            if h < 0.0:
                self.get_logger().warn(
                    f'Current configuration is unsafe! h={h:.4f} m')

        self.get_logger().info(
            f'New goal: x={gx:.3f}  z={gz:.3f}  '
            f'dist_to_obs={dist_to_obs:.3f} m')

    # ── Joint state callback ──────────────────────────────────────────────────
    def joint_state_callback(self, msg):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, joint in enumerate([f'fr3_joint{j}' for j in range(1, 8)]):
            if joint in name_to_idx:
                self.q[i] = msg.position[name_to_idx[joint]]
        self.state_received = True

    # ── Planar Jacobian (joints 1,3,5 → EE in XZ) ────────────────────────────
    def numerical_jacobian(self, q_planar_deg, eps=0.1):
        """2×3 Jacobian: joint velocities [rad/s] → EE velocity [m/s] in XZ."""
        ee0 = self.robot.get_kinematics(q_planar_deg)[-1]
        J   = np.zeros((2, 3))
        for i in range(3):
            qp      = q_planar_deg.copy(); qp[i] += eps
            ee_plus = self.robot.get_kinematics(qp)[-1]
            J[:, i] = (ee_plus - ee0) / np.radians(eps)
        return J

    # ── Planning loop — 20 Hz ─────────────────────────────────────────────────
    def planning_loop(self):
        if not self.state_received or self.goal_xz is None or self.goal_reached:
            return

        q_planar_deg = np.degrees(self.q[[1, 3, 5]])
        x_current    = self.robot.get_kinematics(q_planar_deg)[-1]

        # Goal reached?
        dist = np.linalg.norm(self.goal_xz - x_current)
        if dist < self.goal_tol:
            self.goal_reached = True
            self.get_logger().info(f'Goal reached  dist={dist:.4f} m')
            self.waypoint_pub.publish(self.goal_pose_msg)
            return

        # Step 1: desired Cartesian velocity toward goal
        e      = self.goal_xz - x_current
        dx_des = self.Kp * e

        # Step 2: IK — Cartesian vel → joint vel
        J      = self.numerical_jacobian(q_planar_deg)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
        dq_des = J_pinv @ dx_des

        # Step 3: CBF filter
        dq_safe, h = self.robot.cbf_filter(q_planar_deg, dq_des, self.alpha)
        dq_safe    = np.clip(dq_safe, -self.dq_max, self.dq_max)

        # Step 4: predict next joint config and EE position
        q_next_deg = q_planar_deg + np.degrees(dq_safe * self.dt)
        x_next     = self.robot.get_kinematics(q_next_deg)[-1]

        # Step 5: publish waypoint
        self._publish_waypoint(x_next)

        # Step 6: diagnostics
        cbf_active     = np.linalg.norm(dq_safe - dq_des) > 1e-3
        correction_mag = np.linalg.norm(dq_safe - dq_des)

        s = String()
        s.data = (f'h={h:.4f}  dist={dist:.4f}  '
                  f'cbf_active={cbf_active}  '
                  f'correction={correction_mag:.4f} rad/s')
        self.status_pub.publish(s)

        if cbf_active:
            self.get_logger().warn(
                f'CBF intervening  h={h:.3f}  correction={correction_mag:.4f} rad/s')

    # ── Publish waypoint to Script 3 ──────────────────────────────────────────
    def _publish_waypoint(self, x_next):
        msg                 = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.pose.position.x = float(x_next[0])
        msg.pose.position.y = float(
            self.goal_pose_msg.pose.position.y if self.goal_pose_msg else 0.0)
        msg.pose.position.z = float(x_next[1])   # z is second component of XZ

        if self.goal_pose_msg:
            msg.pose.orientation = self.goal_pose_msg.pose.orientation
        else:
            msg.pose.orientation.w = 1.0

        self.waypoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CBFTrajectoryPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()