#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# pinocchio for IK
import pinocchio as pin


class CartesianPIDController(Node):
    def __init__(self):
        super().__init__('cartesian_pid_controller')

        # ── Robot model — Pinocchio ───────────────────────────────────────────
        # Load URDF — same file your C++ controller uses
        urdf_path = '/tmp/fr3_resolved.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        self.ee_frame_id = self.model.getFrameId('fr3_link8')

        # ── State ─────────────────────────────────────────────────────────────
        self.q       = np.zeros(7)   # joint positions
        self.dq      = np.zeros(7)   # joint velocities
        self.x_des   = None          # desired Cartesian pose (4x4 matrix)
        self.state_received = False

        # ── PID gains ─────────────────────────────────────────────────────────
        # Translational gains
        self.Kp_t = np.array([1.0, 1.0, 1.0])   # proportional
        self.Ki_t = np.array([0.0, 0.0, 0.0])   # integral
        self.Kd_t = np.array([0.1, 0.1, 0.1])   # derivative

        # Rotational gains
        self.Kp_r = np.array([0.5, 0.5, 0.5])
        self.Ki_r = np.array([0.0, 0.0, 0.0])
        self.Kd_r = np.array([0.05, 0.05, 0.05])

        self.integral_t  = np.zeros(3)
        self.integral_r  = np.zeros(3)
        self.prev_error  = np.zeros(6)
        self.dt = 0.01   # 100 Hz

        # ── Subscribers ───────────────────────────────────────────────────────
        # 1. Target pose from terminal
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/cartesian_pid/target_pose',
            self.target_callback,
            10
        )

        # 2. Joint states from robot
        self.joint_sub = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_state_callback,
            10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        # 6. Velocity commands to joint_velocity_example_controller
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10
        )

        # ── Control loop timer — 100 Hz ───────────────────────────────────────
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('CartesianPIDController started')

    # ── Callback 1: target pose ───────────────────────────────────────────────
    def target_callback(self, msg):
        # convert PoseStamped to 4x4 SE3 matrix
        from pinocchio import SE3, Quaternion
        import numpy as np
        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        q = q / np.linalg.norm(q)   # normalize
        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
        t = np.array([msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z])
        self.x_des = pin.SE3(R, t)

    # ── Callback 2: joint states ──────────────────────────────────────────────
    def joint_state_callback(self, msg):
        # franka_ros2 publishes joints in alphabetical order — reorder to 1-7
        joint_order = [f'fr3_joint{i}' for i in range(1, 8)]
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, joint in enumerate(joint_order):
            if joint in name_to_idx:
                idx = name_to_idx[joint]
                self.q[i]  = msg.position[idx]
                self.dq[i] = msg.velocity[idx]
        self.state_received = True

    # ── Control loop ──────────────────────────────────────────────────────────
    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        # ── Step 1: Forward kinematics ────────────────────────────────────────
        q_pin = self.q.copy()
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        x_current = self.data.oMf[self.ee_frame_id]

        # ── Step 2: Cartesian error ───────────────────────────────────────────
        # translation error
        e_t = self.x_des.translation - x_current.translation

        # orientation error — SO3 log map (same as your C++ controller)
        R_err = self.x_des.rotation @ x_current.rotation.T
        e_r   = pin.log3(R_err)

        error = np.concatenate([e_t, e_r])

        # ── Step 3: PID ───────────────────────────────────────────────────────
        self.integral_t += e_t * self.dt
        self.integral_r += e_r * self.dt

        de_t = (e_t - self.prev_error[:3]) / self.dt
        de_r = (e_r - self.prev_error[3:]) / self.dt

        # Cartesian velocity command
        v_t = self.Kp_t * e_t + self.Ki_t * self.integral_t + self.Kd_t * de_t
        v_r = self.Kp_r * e_r + self.Ki_r * self.integral_r + self.Kd_r * de_r

        dx_cmd = np.concatenate([v_t, v_r])   # 6D Cartesian velocity

        self.prev_error = error.copy()

        # ── Step 4: IK — Jacobian pseudoinverse ──────────────────────────────
        # compute Jacobian at current configuration
        J = pin.computeFrameJacobian(
            self.model, self.data, q_pin,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )   # 6x7

        # damped pseudoinverse
        lam = 1e-4
        JJT  = J @ J.T + lam * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)   # 7x6

        # joint velocity command
        dq_cmd = J_pinv @ dx_cmd   # 7x1

        # ── Step 5: velocity saturation ──────────────────────────────────────
        dq_max = 0.3   # rad/s — conservative limit
        dq_cmd = np.clip(dq_cmd, -dq_max, dq_max)

        # ── Step 6: publish ───────────────────────────────────────────────────
        msg = Float64MultiArray()
        msg.data = dq_cmd.tolist()
        self.vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # send zero velocities on shutdown
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 7
        node.vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()