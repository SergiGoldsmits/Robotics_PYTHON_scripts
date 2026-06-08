#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import pinocchio as pin


class CartesianPIDController(Node):
    def __init__(self):
        print("NODE STARTED")
        super().__init__('cartesian_pid_controller')

        # -------------------------
        # URDF + Pinocchio model
        # -------------------------
        urdf_path = '/ros2_ws/src/libfranka/test/fr3.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # -------------------------
        # End-effector frame
        # -------------------------
        self.ee_frame_id = self.model.getFrameId('link8')
        if self.ee_frame_id < 0:
            raise RuntimeError("link8 not found in URDF")

        self.get_logger().info("Using EE frame: link8")

        # -------------------------
        # Robot state
        # -------------------------
        self.q = np.zeros(7)
        self.q_prev = np.zeros(7)
        self.q_dot = np.zeros(7)

        self.state_received = False
        self.x_des = None

        # -------------------------
        # Safety / filtering
        # -------------------------
        self.dq_prev = np.zeros(7)

        self.alpha = 0.15
        self.max_dq = 0.15
        self.max_ddq = 0.4

        # -------------------------
        # Control gains (Cartesian)
        # -------------------------
        self.Kp = 0.8
        self.Kd = 0.25

        # >>> ADDED (rotation gains explicitly zero)
        self.Kp_r = 0.0
        self.Kd_r = 0.0

        # -------------------------
        # Subscribers
        # -------------------------
        self.create_subscription(
            PoseStamped,
            '/cartesian_pid/target_pose',
            self.target_callback,
            10
        )

        self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_state_callback,
            10
        )

        # -------------------------
        # Publisher
        # -------------------------
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10
        )

        # -------------------------
        # Loop timing
        # -------------------------
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("CartesianPIDController started")

    # =========================================================
    # Receive target pose
    # =========================================================
    def target_callback(self, msg):
        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        q = q / np.linalg.norm(q)

        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()

        t = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.x_des = pin.SE3(R, t)

    # =========================================================
    # Receive joint states
    # =========================================================
    def joint_state_callback(self, msg):
        joint_order = [f'fr3_joint{i}' for i in range(1, 8)]
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        for i, j in enumerate(joint_order):
            if j in name_to_idx:
                self.q[i] = msg.position[name_to_idx[j]]

        self.state_received = True

    # =========================================================
    # Control loop
    # =========================================================
    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        # -------------------------
        # Forward kinematics
        # -------------------------
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        x_current = self.data.oMf[self.ee_frame_id]

        # -------------------------
        # Cartesian error (translation)
        # -------------------------
        e_t = self.x_des.translation - x_current.translation

        # =====================================================
        # >>> ADDED: rotational error (theoretically correct)
        # =====================================================
        R_err = self.x_des.rotation @ x_current.rotation.T
        e_r = pin.log3(R_err)

        # -------------------------
        # Jacobian (6x7)
        # -------------------------
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        J_pos = J[:3, :]  # translation only (unchanged behavior)

        # -------------------------
        # Velocity estimate
        # -------------------------
        self.q_dot = (self.q - self.q_prev) / self.dt
        self.q_prev = self.q.copy()

        x_dot = J_pos @ self.q_dot

        # -------------------------
        # Control law (UNCHANGED EFFECT)
        # -------------------------
        v_t = self.Kp * e_t - self.Kd * x_dot

        # >>> rotation is computed but NOT used (2D/planar behavior preserved)
        v_r = self.Kp_r * e_r  # = 0

        dx = np.concatenate([v_t, v_r])

        # -------------------------
        # Jacobian inverse
        # -------------------------
        lam = 1e-4
        JJT = J_pos @ J_pos.T + lam * np.eye(3)
        J_pinv = J_pos.T @ np.linalg.inv(JJT)

        dq_raw = J_pinv @ v_t   # IMPORTANT: unchanged (translation only)

        # -------------------------
        # Velocity saturation
        # -------------------------
        dq_raw = np.clip(dq_raw, -self.max_dq, self.max_dq)

        # -------------------------
        # Acceleration limiting
        # -------------------------
        dq_diff = dq_raw - self.dq_prev
        max_step = self.max_ddq * self.dt
        dq_diff = np.clip(dq_diff, -max_step, max_step)
        dq_limited = self.dq_prev + dq_diff

        # -------------------------
        # Low-pass filtering
        # -------------------------
        dq_cmd = self.alpha * dq_limited + (1 - self.alpha) * self.dq_prev

        self.dq_prev = dq_cmd.copy()

        # -------------------------
        # Publish
        # -------------------------
        msg = Float64MultiArray()
        msg.data = dq_cmd.tolist()
        self.vel_pub.publish(msg)


# =========================================================
# MAIN
# =========================================================
def main():
    rclpy.init()
    node = CartesianPIDController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Float64MultiArray()
        stop.data = [0.0] * 7
        node.vel_pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()