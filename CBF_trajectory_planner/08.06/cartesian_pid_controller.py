#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin

# ===================== CBF =====================

def cbf_h(q, model, data, obs_center, obs_radius):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    f1 = model.getFrameId('link3')
    f2 = model.getFrameId('link5')
    ee = model.getFrameId('link8')

    pairs = [(f1, f2), (f2, ee)]

    obs = np.array(obs_center)
    min_h = 1e9

    for a, b in pairs:
        p1 = data.oMf[a].translation[[0, 2]]
        p2 = data.oMf[b].translation[[0, 2]]

        v = p2 - p1
        w = obs - p1

        t = np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-9), 0, 1)
        dist = np.linalg.norm(w - t * v) - obs_radius

        min_h = min(min_h, dist)

    return min_h


def cbf_grad(q, model, data, obs_center, obs_radius, eps=1e-3):
    h0 = cbf_h(q, model, data, obs_center, obs_radius)
    g = np.zeros_like(q)

    for i in range(len(q)):
        qp = q.copy()
        qm = q.copy()
        qp[i] += eps
        qm[i] -= eps

        g[i] = (cbf_h(qp, model, data, obs_center, obs_radius) -
                cbf_h(qm, model, data, obs_center, obs_radius)) / (2 * eps)

    return h0, g


def cbf_project(q, dq, model, data, obs_center, obs_radius, alpha=0.5):
    h, g = cbf_grad(q, model, data, obs_center, obs_radius)

    cbf_val = np.dot(g, dq) + alpha * h

    correction = 0.0
    active = False

    if cbf_val < 0:
        correction = cbf_val / (np.dot(g, g) + 1e-9)
        dq = dq - correction * g
        active = True

    return dq, h, active, correction


# ===================== CONTROLLER =====================

class Controller(Node):

    def __init__(self):
        super().__init__('cartesian_pid_controller')

        urdf = '/ros2_ws/src/libfranka/test/fr3.urdf'
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()

        self.ee = self.model.getFrameId('link8')

        self.obs = np.array([0.45, 0.25])
        self.obs_r = 0.075

        self.q = np.zeros(7)
        self.q_prev = np.zeros(7)
        self.dq_prev = np.zeros(7)

        self.x_des = None
        self.state_ok = False

        self.dt = 0.01

        # SAFE GAINS
        self.Kp = 0.25
        self.Kd = 0.12

        self.max_dq = 0.03   # ⭐ VERY IMPORTANT (Franka-safe)
        self.max_step = 0.01 # velocity smoothness limit

        self.create_subscription(PoseStamped,
                                 '/cartesian_pid/goal_pose',
                                 self.goal_cb, 10)

        self.create_subscription(JointState,
                                 '/franka/joint_states',
                                 self.state_cb, 10)

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10)

        self.create_timer(self.dt, self.loop)

        self.get_logger().info("FRANKA SAFE PID+CBF STARTED")

    # ---------- goal ----------
    def goal_cb(self, msg):
        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        q = q / (np.linalg.norm(q) + 1e-9)

        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
        t = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.x_des = pin.SE3(R, t)

    # ---------- state ----------
    def state_cb(self, msg):
        idx = {n:i for i,n in enumerate(msg.name)}
        for i in range(7):
            name = f'fr3_joint{i+1}'
            if name in idx:
                self.q[i] = msg.position[idx[name]]
        self.state_ok = True

    # ---------- STOP SAFE ----------
    def send_zero(self):
        msg = Float64MultiArray()
        msg.data = [0.0]*7
        self.pub.publish(msg)

    # ---------- MAIN LOOP ----------
    def loop(self):

        if not self.state_ok or self.x_des is None:
            self.send_zero()
            return

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        x = self.data.oMf[self.ee]

        e = self.x_des.translation - x.translation

        J = pin.computeFrameJacobian(
            self.model, self.data, self.q,
            self.ee,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        Jp = J[:3]

        qdot = (self.q - self.q_prev) / self.dt
        self.q_prev = self.q.copy()

        xdot = Jp @ qdot

        # PID
        v = self.Kp * e - self.Kd * xdot

        dq = Jp.T @ np.linalg.inv(Jp @ Jp.T + 1e-4*np.eye(3)) @ v

        # CBF
        dq, h, active, corr = cbf_project(
            self.q, dq,
            self.model, self.data,
            self.obs, self.obs_r
        )

        # HARD SAFETY LIMIT
        dq = np.clip(dq, -self.max_dq, self.max_dq)

        # ⭐ SMOOTHING (VERY IMPORTANT FIX)
        dq = np.clip(dq, self.dq_prev - self.max_step,
                          self.dq_prev + self.max_step)

        self.dq_prev = dq.copy()

        msg = Float64MultiArray()
        msg.data = dq.tolist()
        self.pub.publish(msg)

        self.get_logger().info(
            f"h={h:.3f} | cbf={active} | corr={corr:.4f}"
        )


def main():
    rclpy.init()
    node = Controller()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("STOPPING SAFELY")
    finally:
        node.send_zero()
        node.send_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()