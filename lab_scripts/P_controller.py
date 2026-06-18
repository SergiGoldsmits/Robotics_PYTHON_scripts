#!/usr/bin/env python3
"""
Modified FRANKA FR3 Controller.
Bypasses the CBF filter and escape logic to act as a normal P controller.
Maintains calculations and logging of the barrier function (h) to observe negative values.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import pinocchio as pin
import csv
import time

ACTIVE = [1, 3, 5]                 # planar joints (fr3_joint2, 4, 6) -> XZ motion
TASK_ROWS = [0, 2]                 # X, Z rows of the 6xN Jacobian
LINK_SEGMENTS = [('fr3_link3', 'fr3_link4'), ('fr3_link4', 'fr3_link5'),
                 ('fr3_link5', 'fr3_link6'), ('fr3_link6', 'fr3_link7'),
                 ('fr3_link7', 'fr3_link8')]


# =============================================================================
#  Capisani angular barrier
# =============================================================================
def capisani_delta(p_base, p_end, obs_center, r_eff):
    x_b, z_b = p_base[0], p_base[2]
    x_e, z_e = p_end[0], p_end[2]
    l_i = np.hypot(x_e - x_b, z_e - z_b)
    d_i = np.hypot(obs_center[0] - x_b, obs_center[1] - z_b)

    if d_i <= r_eff:                                                   
        delta_i = np.pi / 2.0
    elif l_i * l_i >= (d_i * d_i - r_eff * r_eff):                     
        delta_i = np.arcsin(np.clip(r_eff / d_i, -1.0, 1.0))
    else:                                                             
        denom = 2.0 * d_i * l_i
        val = (l_i * l_i + d_i * d_i - r_eff * r_eff) / denom if denom > 1e-9 else 1.0
        if val <= 1.0:
            delta_i = np.arccos(np.clip(val, -1.0, 1.0))              
        else:
            delta_i = -np.arccosh(val)                                

    q_w = np.arctan2(obs_center[1] - z_b, obs_center[0] - x_b)         
    q_l = np.arctan2(z_e - z_b, x_e - x_b)                             
    diff = np.arctan2(np.sin(q_l - q_w), np.cos(q_l - q_w))            
    return diff, delta_i


def _link_barriers(q_full, model, data, obs_center, r_eff):
    pin.forwardKinematics(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)
    hs = []
    for bf, ef in LINK_SEGMENTS:
        pb = data.oMf[model.getFrameId(bf)].translation
        pe = data.oMf[model.getFrameId(ef)].translation
        diff, delta = capisani_delta(pb, pe, obs_center, r_eff)
        hs.append(diff * diff - np.copysign(delta * delta, delta))
    return np.array(hs) if hs else np.array([1e3])


def _soft_min(h, beta):
    m = float(np.min(h))
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (h - m))))


def h_and_grad(q_full, model, obs_center, r_eff, beta=15.0, eps=1e-4):
    h0 = _soft_min(_link_barriers(q_full, model, model.createData(),
                                  obs_center, r_eff), beta)
    g = np.zeros(len(ACTIVE))
    for k, j in enumerate(ACTIVE):
        qp = q_full.copy(); qp[j] += eps
        qm = q_full.copy(); qm[j] -= eps
        hp = _soft_min(_link_barriers(qp, model, model.createData(), obs_center, r_eff), beta)
        hm = _soft_min(_link_barriers(qm, model, model.createData(), obs_center, r_eff), beta)
        g[k] = (hp - hm) / (2.0 * eps)
    return h0, g


# =============================================================================
#  Controller node
# =============================================================================
class CartesianCBFController(Node):
    def __init__(self):
        super().__init__('cartesian_cbf_controller')
        self.model = pin.buildModelFromUrdf('/tmp/fr3_resolved.urdf')
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId('fr3_link8')

        # --- obstacle ---
        self.obs_center = np.array([0.4, 0.20])
        self.obs_radius = 0.07
        self.link_buffer = 0.00
        self.r_eff = self.obs_radius + self.link_buffer

        # --- state ---
        self.q = np.zeros(7)
        self.state_received = False
        self.x_des = None
        self.dq_prev = np.zeros(3)

        # --- nominal (PID) controller ---
        self.dt = 0.01
        self.Kp = 2.0
        self.v_max_task = 0.20
        self.max_dq = 1.0                              

        # --- CBF tracking (For monitoring purposes only) ---
        self.lambda_cbf = 10.0                          
        self.beta = 15.0
        self.grad_clip = 20.0

        # --- data logging ---
        self.start_time = time.time()
        self.csv_filename = '/tmp/cbf_trajectory_data.csv'
        with open(self.csv_filename, mode='w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'x_ee', 'z_ee', 'h', 'cstr', 'escaping', 'feasible',
                'correction_mag', 'q0', 'q1', 'q2', 'dq0', 'dq1', 'dq2',
                'ddq0', 'ddq1', 'ddq2'])

        # --- ROS I/O ---
        self.vel_pub = self.create_publisher(
            Float64MultiArray, '/joint_velocity_example_controller/commands', 10)
        self.status_pub = self.create_publisher(String, '/cbf_status', 10)
        self.create_subscription(PoseStamped, '/cartesian_pid/goal_pose',
                                 self.goal_callback, 10)
        self.create_subscription(JointState, '/franka/joint_states',
                                 self.joint_state_callback, 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

    def goal_callback(self, msg):
        t = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        o = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w])
        n = np.linalg.norm(o)
        o = o / n if n > 1e-6 else np.array([0., 0., 0., 1.])
        self.x_des = pin.SE3(pin.Quaternion(o[3], o[0], o[1], o[2]).toRotationMatrix(), t)

    def joint_state_callback(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for i, j in enumerate([f'fr3_joint{k}' for k in range(1, 8)]):
            if j in name_to_idx:
                self.q[i] = msg.position[name_to_idx[j]]
        self.state_received = True

    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.framesForwardKinematics(self.model, self.data, self.q)

        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id,
                                     pin.LOCAL_WORLD_ALIGNED)[np.ix_(TASK_ROWS, ACTIVE)]
        p_ee = self.data.oMf[self.ee_frame_id].translation
        x_ee = np.array([p_ee[0], p_ee[2]])
        x_goal = np.array([self.x_des.translation[0], self.x_des.translation[2]])

        # 1) Nominal task-space P control -> joint velocity
        e = x_goal - x_ee
        v_pid = self.Kp * e
        sp = np.linalg.norm(v_pid)
        if sp > self.v_max_task:
            v_pid = v_pid * (self.v_max_task / sp)
        ld = 0.05
        Jinv = J.T @ np.linalg.inv(J @ J.T + ld * ld * np.eye(2))
        dq_nom = Jinv @ v_pid

        # 2) Basic joint velocity saturation box constraint
        dq = np.clip(dq_nom, -self.max_dq, self.max_dq)

        # 3) Calculate barrier value + gradient purely for monitoring/logging
        h, g = h_and_grad(self.q, self.model, self.obs_center, self.r_eff, self.beta)
        gn = np.linalg.norm(g)
        if gn > self.grad_clip:
            g = g * (self.grad_clip / gn)

        # Theoretical constraint margin calculation 
        cstr_pub = float(g @ dq + self.lambda_cbf * h)

        # 4) Publish unmitigated P-control command
        ddq = (dq - self.dq_prev) / self.dt
        self.dq_prev = dq

        final = np.zeros(7)
        final[ACTIVE] = dq
        self.vel_pub.publish(Float64MultiArray(data=final.tolist()))

        # 5) Log metrics to watch h drop deep into the negative territory
        elapsed = time.time() - self.start_time
        with open(self.csv_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([
                elapsed, x_ee[0], x_ee[1], h, cstr_pub, 0, 1,
                0.0, self.q[ACTIVE[0]], self.q[ACTIVE[1]], self.q[ACTIVE[2]],
                dq[0], dq[1], dq[2], ddq[0], ddq[1], ddq[2]])

        self.status_pub.publish(String(
            data=(f"x={x_ee[0]:+.3f} z={x_ee[1]:+.3f} h={h:+.4f} cstr={cstr_pub:+.4f} "
                  f"esc=0 feas=1 |e|={np.linalg.norm(e):.3f} |dq|={np.linalg.norm(dq):.3f}")))


def main():
    rclpy.init()
    rclpy.spin(CartesianCBFController())


if __name__ == '__main__':
    main()
