#!/usr/bin/env python3
"""
Modified Capisani-CBF velocity-level safety filter for the FRANKA FR3  --  FULL 3D / 7-DOF.
Bypasses the CBF filter and circulation field logic to act as a normal 3D P controller.
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

ACTIVE = list(range(7))            # all 7 joints
TASK_ROWS = [0, 1, 2]              # X, Y, Z rows of the 6xN Jacobian
LINK_SEGMENTS = [('fr3_link3', 'fr3_link4'), ('fr3_link4', 'fr3_link5'),
                 ('fr3_link5', 'fr3_link6'), ('fr3_link6', 'fr3_link7'),
                 ('fr3_link7', 'fr3_link8')]


# =============================================================================
#  Capisani angular barrier (3D)
# =============================================================================
def capisani_delta(p_base, p_end, obs_center, r_eff):
    """Cone half-width delta_i (scalar, unchanged) and UNSIGNED angle phi_i between the
    link direction and the base->obstacle direction."""
    u_link = np.asarray(p_end) - np.asarray(p_base)
    d_vec = np.asarray(obs_center) - np.asarray(p_base)
    l_i = np.linalg.norm(u_link)
    d_i = np.linalg.norm(d_vec)

    if d_i <= r_eff:                                                  # base inside obstacle
        delta_i = np.pi / 2.0
    elif l_i * l_i >= (d_i * d_i - r_eff * r_eff):                    # Case A : tangent
        delta_i = np.arcsin(np.clip(r_eff / d_i, -1.0, 1.0))
    else:                                                            # Case B / Case C
        denom = 2.0 * d_i * l_i
        val = (l_i * l_i + d_i * d_i - r_eff * r_eff) / denom if denom > 1e-9 else 1.0
        if val <= 1.0:
            delta_i = np.arccos(np.clip(val, -1.0, 1.0))             # Case B
        else:
            delta_i = -np.arccosh(val)                               # Case C (empty cone)

    if l_i < 1e-9 or d_i < 1e-9:                                     # degenerate -> far from cone
        phi_i = np.pi
    else:
        cphi = np.clip(np.dot(u_link, d_vec) / (l_i * d_i), -1.0, 1.0)
        phi_i = np.arccos(cphi)                                      # in [0, pi]
    return phi_i, delta_i


def _link_barriers(q_full, model, data, obs_center, r_eff):
    """Per-link  h_i = phi_i^2 - sign(delta_i)*delta_i^2  (>0 = safe)."""
    pin.forwardKinematics(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)
    hs = []
    for bf, ef in LINK_SEGMENTS:
        pb = data.oMf[model.getFrameId(bf)].translation
        pe = data.oMf[model.getFrameId(ef)].translation
        phi, delta = capisani_delta(pb, pe, obs_center, r_eff)
        hs.append(phi * phi - np.copysign(delta * delta, delta))
    return np.array(hs) if hs else np.array([1e3])


def _soft_min(h, beta):
    """Smooth (C^inf) under-estimate of min(h)."""
    m = float(np.min(h))
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (h - m))))


def h_and_grad(q_full, model, obs_center, r_eff, beta=15.0, eps=1e-4):
    """Barrier value h(q) and gradient grad(h) w.r.t. the 7 active joints (central diff)."""
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
        super().__init__('cartesian_cbf_controller_3d')
        self.model = pin.buildModelFromUrdf('/tmp/fr3_resolved.urdf')
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId('fr3_link8')

        # --- obstacle (3D: X, Y, Z) ---
        self.obs_center = np.array([0.45, 0.0, 0.2])
        self.obs_radius = 0.10
        self.link_buffer = 0.06
        self.r_eff = self.obs_radius + self.link_buffer

        # --- state ---
        self.q = np.zeros(7)
        self.state_received = False
        self.x_des = None
        self.dq_prev = np.zeros(7)
        self.dq_filt = np.zeros(7)                     # LPF state for published command
        self.lpf_alpha = 0.3                           # 0<a<=1; lower = smoother, more lag

        # --- nominal (P) controller ---
        self.dt = 0.01
        self.Kp = 0.8
        self.v_max_task = 0.15
        self.max_dq = 1.0                              # Direct joint velocity max bounds

        # --- null-space posture (uses the 4-D redundancy) ---
        self.q_rest = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
        self.k_post = 1.0

        # --- CBF Tracking (For monitoring purposes only) ---
        self.lambda_cbf = 2.0                          
        self.beta = 15.0
        self.grad_clip = 40.0

        # --- data logging ---
        self.start_time = time.time()
        self.csv_filename = '/tmp/cbf_trajectory_data_3d.csv'
        header = (['timestamp', 'x_ee', 'y_ee', 'z_ee', 'h', 'cstr', 'feasible',
                   'gn_raw', 'circ_mag', 'correction_mag']
                  + [f'q{i}' for i in range(7)]
                  + [f'dq{i}' for i in range(7)]
                  + [f'ddq{i}' for i in range(7)])
        with open(self.csv_filename, mode='w', newline='') as f:
            csv.writer(f).writerow(header)

        # --- ROS I/O ---
        self.vel_pub = self.create_publisher(
            Float64MultiArray, '/joint_velocity_example_controller/commands', 10)
        self.status_pub = self.create_publisher(String, '/cbf_status', 10)
        self.create_subscription(PoseStamped, '/cartesian_pid/goal_pose',
                                 self.goal_callback, 10)
        self.create_subscription(JointState, '/franka/joint_states',
                                 self.joint_state_callback, 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

    # ----- callbacks ----------------------------------------------------------
    def goal_callback(self, msg):
        self.x_des = np.array([msg.pose.position.x,
                               msg.pose.position.y,
                               msg.pose.position.z])

    def joint_state_callback(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for i, j in enumerate([f'fr3_joint{k}' for k in range(1, 8)]):
            if j in name_to_idx:
                self.q[i] = msg.position[name_to_idx[j]]
        self.state_received = True

    # ----- main loop -----------------------------------------------------------
    def control_loop(self):
        if not self.state_received or self.x_des is None:
            return

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.framesForwardKinematics(self.model, self.data, self.q)

        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id,
                                     pin.LOCAL_WORLD_ALIGNED)[np.ix_(TASK_ROWS, ACTIVE)]  # 3x7
        x_ee = np.array(self.data.oMf[self.ee_frame_id].translation)
        x_goal = self.x_des

        # 1) nominal task-space P control
        e = x_goal - x_ee
        v_pid = self.Kp * e
        sp = np.linalg.norm(v_pid)
        if sp > self.v_max_task:
            v_pid = v_pid * (self.v_max_task / sp)

        # 2) resolved-rate kinematics with null-space posture task
        Jpinv_true = np.linalg.pinv(J)                               # true pinv, projector only
        N = np.eye(7) - Jpinv_true @ J                               # Null space projector
        dq_posture = -self.k_post * (self.q[ACTIVE] - self.q_rest)
        
        # We use a slight task damping for singularity management near operational limits
        ld = 0.05
        Jinv = J.T @ np.linalg.inv(J @ J.T + ld * ld * np.eye(3))     
        dq_nom = Jinv @ v_pid + N @ dq_posture

        # 3) Basic joint velocity saturation box constraint
        dq = np.clip(dq_nom, -self.max_dq, self.max_dq)

        # 4) Calculate barrier value + gradient purely for monitoring/logging
        h, g = h_and_grad(self.q, self.model, self.obs_center, self.r_eff, self.beta)
        gn_raw = float(np.linalg.norm(g))
        if gn_raw > self.grad_clip:
            g = g * (self.grad_clip / gn_raw)

        # Theoretical constraint margin calculation
        cstr_pub = float(g @ dq + self.lambda_cbf * h)

        # 5) Publish unmitigated P-control command
        ddq = (dq - self.dq_prev) / self.dt
        self.dq_prev = dq.copy()

        self.dq_filt = self.lpf_alpha * dq + (1.0 - self.lpf_alpha) * self.dq_filt
        final = np.zeros(7)
        final[ACTIVE] = self.dq_filt
        self.vel_pub.publish(Float64MultiArray(data=final.tolist()))

        # 6) Log metrics to watch h drop deep into negative territory
        elapsed = time.time() - self.start_time
        row = ([elapsed, x_ee[0], x_ee[1], x_ee[2], h, cstr_pub, 1,
                gn_raw, 0.0, 0.0]
               + list(self.q) + list(dq) + list(ddq))
        with open(self.csv_filename, mode='a', newline='') as f:
            csv.writer(f).writerow(row)

        self.status_pub.publish(String(
            data=(f"ee=({x_ee[0]:+.3f},{x_ee[1]:+.3f},{x_ee[2]:+.3f}) "
                  f"h={h:+.4f} cstr={cstr_pub:+.4f} circ=0.000 "
                  f"gn={gn_raw:.1f} feas=1 "
                  f"|e|={np.linalg.norm(e):.3f} |dq|={np.linalg.norm(dq):.3f}")))


def main():
    rclpy.init()
    rclpy.spin(CartesianCBFController())


if __name__ == '__main__':
    main()
