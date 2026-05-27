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
        print("NODE STARTED")

        # -------------------------
        # MODEL
        # -------------------------
        urdf_path = '/ros2_ws/src/libfranka/test/fr3.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.ee_frame_id = self.model.getFrameId('link8')
        if self.ee_frame_id < 0:
            raise RuntimeError("link8 not found in URDF")

        self.get_logger().info("Using EE frame: link8")

        # -------------------------
        # STATE
        # -------------------------
        self.q = np.zeros(7)
        self.q_prev = np.zeros(7)
        self.x_des = None
        self.state_received = False

        # -------------------------
        # SAFETY / FILTERING
        # -------------------------
        self.dq_prev = np.zeros(7)

        self.alpha = 0.25          # smoothing (higher = smoother)
        self.max_dq = 0.15         # velocity limit

        # IMPORTANT: acceleration safety (very strict)
        self.max_dq_step = 0.02    # per control cycle change limit

        # Cartesian velocity limit (CRITICAL FIX)
        self.max_cart_v = 0.15

        # -------------------------
        # GAINS
        # -------------------------
        self.Kp_pos = 0.7
        self.Kp_rot = 0.3
        self.Kd = 0.2

        # -------------------------
        # ROS
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

        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10
        )

        # -------------------------
        # LOOP
        # -------------------------
        self.dt = 0.01
        self.t = 0.0
        self.timer = self.create_timer(self.dt, self.control_loop)

    # =========================================================
    # TARGET
    # =========================================================
    def target_callback(self, msg):
        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        q /= np.linalg.norm(q)

        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
        t = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.x_des = pin.SE3(R, t)

    # =========================================================
    # JOINT STATE
    # =========================================================
    def joint_state_callback(self, msg):
        joint_order = [f'fr3_joint{i}' for i in range(1, 8)]
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        for i, j in enumerate(joint_order):
            if j in name_to_idx:
                self.q[i] = msg.position[name_to_idx[j]]

        self.state_received = True

    # =========================================================
    # CONTROL LOOP
    # =========================================================
    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        self.t += self.dt

        # SLOW STARTUP RAMP (FIX #1)
        ramp = min(1.0, self.t / 4.0)

        # Forward kinematics
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        x = self.data.oMf[self.ee_frame_id]

        # -------------------------
        # POSITION ERROR
        # -------------------------
        e_pos = self.x_des.translation - x.translation

        # -------------------------
        # ORIENTATION ERROR
        # -------------------------
        R_err = self.x_des.rotation @ x.rotation.T
        e_rot = pin.log3(R_err)

        # -------------------------
        # VELOCITY ESTIMATE
        # -------------------------
        q_dot = (self.q - self.q_prev) / self.dt
        self.q_prev = self.q.copy()

        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL
        )

        x_dot = J @ q_dot

        # -------------------------
        # CARTESIAN CONTROL LAW
        # -------------------------
        v = np.concatenate([
            self.Kp_pos * e_pos - self.Kd * x_dot[:3],
            self.Kp_rot * e_rot - self.Kd * x_dot[3:]
        ])

        v *= ramp

        # -------------------------
        # FIX #2: LIMIT CARTESIAN VELOCITY NORM
        # -------------------------
        norm = np.linalg.norm(v)
        if norm > self.max_cart_v:
            v *= self.max_cart_v / norm

        # -------------------------
        # IK (DAMPED LEAST SQUARES)
        # -------------------------
        lam = 1e-3
        JJT = J @ J.T + lam * np.eye(6)
        dq = J.T @ np.linalg.inv(JJT) @ v

        # -------------------------
        # FIX #3: VELOCITY LIMIT
        # -------------------------
        dq = np.clip(dq, -self.max_dq, self.max_dq)

        # -------------------------
        # FIX #4: ACCELERATION LIMIT (IMPORTANT)
        # -------------------------
        dq_step = dq - self.dq_prev
        dq_step = np.clip(dq_step, -self.max_dq_step, self.max_dq_step)
        dq = self.dq_prev + dq_step

        # -------------------------
        # SMOOTHING FILTER
        # -------------------------
        dq = self.alpha * dq + (1 - self.alpha) * self.dq_prev
        self.dq_prev = dq.copy()

        # -------------------------
        # PUBLISH
        # -------------------------
        msg = Float64MultiArray()
        msg.data = dq.tolist()
        self.vel_pub.publish(msg)


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
