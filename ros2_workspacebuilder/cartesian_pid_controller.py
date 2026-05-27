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
        super().__init__('cartesian_pid_controller')

        # ── Robot model — Pinocchio ───────────────────────────────────────────
        urdf_path = '/tmp/fr3_resolved.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        self.ee_frame_id = self.model.getFrameId('fr3_link8')

        # ── State ─────────────────────────────────────────────────────────────
        self.q      = np.zeros(7)
        self.dq     = np.zeros(7)
        self.x_des  = None
        self.state_received = False

        # ── PID gains ─────────────────────────────────────────────────────────
        self.Kp_t = np.array([1.0, 1.0, 1.0])
        self.Ki_t = np.array([0.0, 0.0, 0.0])
        self.Kd_t = np.array([0.1, 0.1, 0.1])

        self.Kp_r = np.array([0.5, 0.5, 0.5])
        self.Ki_r = np.array([0.0, 0.0, 0.0])
        self.Kd_r = np.array([0.05, 0.05, 0.05])

        self.integral_t = np.zeros(3)
        self.integral_r = np.zeros(3)
        self.prev_error = np.zeros(6)
        self.dt = 0.01  # 100 Hz

        # ── Subscribers ───────────────────────────────────────────────────────
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/cartesian_pid/target_pose',
            self.target_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10
        )

        # ── Control loop — 100 Hz ─────────────────────────────────────────────
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('CartesianPIDController started')

    # ── Callback: target pose ─────────────────────────────────────────────────
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
        self.get_logger().info(
            f'New target: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]')

    # ── Callback: joint states ────────────────────────────────────────────────
    def joint_state_callback(self, msg):
        # franka_ros2 publishes joints alphabetically — map by name
        joint_order = [f'fr3_joint{i}' for i in range(1, 8)]
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, joint in enumerate(joint_order):
            if joint in name_to_idx:
                idx = name_to_idx[joint]
                self.q[i]  = msg.position[idx]
                self.dq[i] = msg.velocity[idx] if len(msg.velocity) > idx else 0.0
        self.state_received = True

    # ── Control loop ──────────────────────────────────────────────────────────
    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        # ── Forward kinematics ────────────────────────────────────────────────
        q_pin = self.q.copy()
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        x_current = self.data.oMf[self.ee_frame_id]

        # ── Cartesian error ───────────────────────────────────────────────────
        e_t = self.x_des.translation - x_current.translation
        R_err = self.x_des.rotation @ x_current.rotation.T
        e_r   = pin.log3(R_err)
        error = np.concatenate([e_t, e_r])

        # ── PID ───────────────────────────────────────────────────────────────
        self.integral_t += e_t * self.dt
        self.integral_r += e_r * self.dt

        de_t = (e_t - self.prev_error[:3]) / self.dt
        de_r = (e_r - self.prev_error[3:]) / self.dt

        v_t = (self.Kp_t * e_t
               + self.Ki_t * self.integral_t
               + self.Kd_t * de_t)
        v_r = (self.Kp_r * e_r
               + self.Ki_r * self.integral_r
               + self.Kd_r * de_r)

        dx_cmd = np.concatenate([v_t, v_r])
        self.prev_error = error.copy()

        # ── IK — damped Jacobian pseudoinverse ────────────────────────────────
        J = pin.computeFrameJacobian(
            self.model, self.data, q_pin,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        lam   = 1e-4
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        dq_cmd = J_pinv @ dx_cmd

        # ── Velocity saturation ───────────────────────────────────────────────
        dq_max = 1.0  # rad/s
        dq_cmd = np.clip(dq_cmd, -dq_max, dq_max)

        # ── Publish ───────────────────────────────────────────────────────────
        msg_out = Float64MultiArray()
        msg_out.data = dq_cmd.tolist()
        self.vel_pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 7
        node.vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
