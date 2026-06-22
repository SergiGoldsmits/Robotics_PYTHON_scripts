#!/usr/bin/env python3
"""
Capisani-CBF velocity-level safety filter for the FRANKA FR3  --  FULL 3D / 7-DOF.

3D generalisation of the validated planar controller. Single CBF row (softmin over
links) projected by a box-constrained QP; box carries velocity + acceleration + jerk.

CHANGES vs the planar version
  * Barrier: signed planar angle diff -> UNSIGNED 3D angle phi = arccos(u_link . u_obs).
    Enters h squared, so dropping the sign is exact. h_i = phi^2 - sign(delta)*delta^2.
    Cases A/B/C of delta are scalar (l_i, d_i, r_eff) and unchanged. New arccos
    (anti)parallel singularity is absorbed by the gradient-norm clip.
  * ACTIVE = all 7 joints, TASK_ROWS = X,Y,Z. 4-D null space -> posture task.
  * Latch -> CIRCULATION field: smooth swirl around the obstacle, axis-selectable, sign
    toward the goal, gated by exp(-h/h_act) (proximity to barrier) AND by tanh(|e|/e0)
    (proximity to GOAL). The goal fade is what stops the EE orbiting / sign-chattering at
    the goal: circulation should exist only while you are being BLOCKED on the way
    somewhere, and be exactly zero once you have arrived. Enters NOMINAL only -> QP still
    certifies safety unchanged.
  * Null-space projector built from the TRUE pinv(J), not the damped Jinv, so the posture
    term does not leak into task-space motion (the damped-Jinv projector had J N != 0,
    which showed up as residual jitter near the goal). DLS is kept for the task term only.

OPERATIONAL NOTE (read before running)
  The published command must EQUAL the certified command. Any velocity/accel caps in the
  C++ velocity_example_controller MUST be strictly LOOSER than this QP box (or removed).
  If the low level clamps tighter, realised dq lags certified dq, g.dq_realised < -lambda*h,
  and h dives below 0 even though the logged `cstr` stays >= 0. That cstr>=0-but-h<0
  signature is exactly the C++ gap. The QP must be the only limiter.

DIAGNOSING h < 0 from the CSV: look at `cstr`, `feasible`, `gn_raw`.
   cstr>=0 & h<0          -> realised != commanded (C++ caps). Loosen/remove C++ caps.
   cstr<0 & feasible==0   -> box too tight for the CBF. Lower lambda / raise r_eff.
   cstr>=0 & gn_raw==clip -> gradient clip firing at the arccos singularity. Raise clip.
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

ACTIVE = list(range(7))            # all 7 joints help avoid in 3D
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
#  Circulation (vortex) field  --  replaces the planar latch
# =============================================================================
def circulation_velocity(x_ee, x_goal, obs_center, h, k_circ, h_act, axis, w_cap=3.0):
    """Smooth swirl around the obstacle. Sign chosen toward the goal; gated by
    exp(-h/h_act) (barrier proximity). The separate GOAL fade is applied in the loop."""
    r_vec = np.asarray(x_ee) - np.asarray(obs_center)
    v_swirl = np.cross(axis, r_vec)
    nv = np.linalg.norm(v_swirl)
    if nv < 1e-9:                                  # EE on the circulation axis -> undefined
        return np.zeros(3)
    v_swirl /= nv
    if np.dot(v_swirl, np.asarray(x_goal) - np.asarray(x_ee)) < 0.0:
        v_swirl = -v_swirl
    w = k_circ * min(float(np.exp(-h / h_act)), w_cap)
    return w * v_swirl


# =============================================================================
#  Actuator box: velocity AND acceleration AND jerk as a single box (keeps the
#  QP closed-form). accel_prev clipped into the accel band first => lb <= ub.
# =============================================================================
def actuator_box(dq_prev, dq_prev2, dt, ddq_max, j_max, v_max):
    accel_prev = np.clip((dq_prev - dq_prev2) / dt, -ddq_max, ddq_max)
    a_lo = np.maximum(-ddq_max, accel_prev - j_max * dt)
    a_hi = np.minimum( ddq_max, accel_prev + j_max * dt)
    lb = np.maximum(-v_max, dq_prev + dt * a_lo)
    ub = np.minimum( v_max, dq_prev + dt * a_hi)
    lb = np.minimum(lb, ub)
    return lb, ub


# =============================================================================
#  Safety QP:  min 1/2||dq-dq_nom||^2  s.t.  a.dq >= b,  lb <= dq <= ub
#  Box + ONE linear inequality via the scalar dual. Dimension-agnostic.
# =============================================================================
def safety_qp(dq_nom, a, b, lb, ub, n_iter=60):
    """Returns (dq, active, feasible)."""
    dq = np.clip(dq_nom, lb, ub)
    if float(a @ dq) >= b:
        return dq, False, True
    if float(a @ a) < 1e-12:
        return dq, False, False

    def adq(mu):
        return float(a @ np.clip(dq_nom + mu * a, lb, ub))

    lo, hi, it = 0.0, 1.0, 0
    while adq(hi) < b and hi < 1e9:
        hi *= 2.0; it += 1
        if it > 80:
            break
    feasible = adq(hi) >= b
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if adq(mid) < b:
            lo = mid
        else:
            hi = mid
    dq = np.clip(dq_nom + hi * a, lb, ub)
    return dq, True, feasible


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
        self.obs_center = np.array([0.45, 0.0, 0.25])
        self.obs_radius = 0.06
        self.link_buffer = 0.025
        self.r_eff = self.obs_radius + self.link_buffer

        # --- state ---
        self.q = np.zeros(7)
        self.state_received = False
        self.x_des = None
        self.dq_prev = np.zeros(7)
        self.dq_prev2 = np.zeros(7)                    # second-order memory for jerk

        # --- nominal (P) controller ---
        self.dt = 0.01
        self.Kp = 1.0
        self.v_max_task = 0.25

        # --- null-space posture (uses the 4-D redundancy) ---
        self.q_rest = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
        self.k_post = 1.0

        # --- per-joint actuator limits [j1..j7]  (TUNE to your URDF / safety factor) ---
        sf = 1.1
        self.v_max   = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]) / sf
        self.ddq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / sf
        self.j_max   = np.array([40., 40., 40., 40., 40., 40., 40.]) / sf

        # --- CBF filter ---
        self.lambda_cbf = 1.50                          # lower = brakes earlier, keeps h higher
        self.beta = 15.0
        self.grad_clip = 40.0

        # --- circulation field (replaces the latch) ---
        self.k_circ = 0.20                             # swirl speed scale [m/s]
        self.h_act = 0.10                              # barrier-proximity gate width (rad^2)
        self.e0 = 0.06                                 # GOAL fade scale [m] -> kills orbit/chatter
        axis = np.array([0.0, 0.0, 1.0])               # gravity-up -> "go around the side"
        self.circ_axis = axis / np.linalg.norm(axis)

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
                               msg.pose.position.z])   # all 3 used; no latch to reset

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

        # 2) barrier value + gradient (record raw norm before clipping for diagnostics)
        h, g = h_and_grad(self.q, self.model, self.obs_center, self.r_eff, self.beta)
        gn_raw = float(np.linalg.norm(g))
        if gn_raw > self.grad_clip:
            g = g * (self.grad_clip / gn_raw)

        # 3) circulation -> task velocity (NOMINAL only). Fades to 0 at the goal.
        v_circ = circulation_velocity(x_ee, x_goal, self.obs_center, h,
                                      self.k_circ, self.h_act, self.circ_axis)
        goal_fade = float(np.tanh(np.linalg.norm(e) / self.e0))
        v_circ = v_circ * goal_fade
        v_task = v_pid + v_circ

        # 4) resolved-rate: DLS for the task term, TRUE pinv for the null-space projector
        ld = 0.05
        Jinv = J.T @ np.linalg.inv(J @ J.T + ld * ld * np.eye(3))     # damped, task term
        Jpinv_true = np.linalg.pinv(J)                               # true, projector only
        N = np.eye(7) - Jpinv_true @ J                               # J N ~= 0 -> no leak
        dq_posture = -self.k_post * (self.q[ACTIVE] - self.q_rest)
        dq_nom = Jinv @ v_task + N @ dq_posture
        dq_nom_pure = (Jinv @ v_pid + N @ dq_posture).copy()         # for correction_mag

        # 5) SINGLE actuator-limited safety QP (vel + accel + jerk box, ONE CBF row)
        lb, ub = actuator_box(self.dq_prev, self.dq_prev2, self.dt,
                              self.ddq_max, self.j_max, self.v_max)
        a = g
        b = -self.lambda_cbf * h
        dq, active, feasible = safety_qp(dq_nom, a, b, lb, ub)

        # 6) publish
        ddq = (dq - self.dq_prev) / self.dt
        cstr_pub = float(g @ dq + self.lambda_cbf * h)
        circ_mag = float(np.linalg.norm(v_circ))
        correction_mag = float(np.linalg.norm(dq - dq_nom_pure))
        self.dq_prev2 = self.dq_prev.copy()
        self.dq_prev = dq.copy()

        final = np.zeros(7)
        final[ACTIVE] = dq
        self.vel_pub.publish(Float64MultiArray(data=final.tolist()))

        elapsed = time.time() - self.start_time
        row = ([elapsed, x_ee[0], x_ee[1], x_ee[2], h, cstr_pub, int(feasible),
                gn_raw, circ_mag, correction_mag]
               + list(self.q) + list(dq) + list(ddq))
        with open(self.csv_filename, mode='a', newline='') as f:
            csv.writer(f).writerow(row)

        self.status_pub.publish(String(
            data=(f"ee=({x_ee[0]:+.3f},{x_ee[1]:+.3f},{x_ee[2]:+.3f}) "
                  f"h={h:+.4f} cstr={cstr_pub:+.4f} circ={circ_mag:.3f} "
                  f"gn={gn_raw:.1f} feas={int(feasible)} "
                  f"|e|={np.linalg.norm(e):.3f} |dq|={np.linalg.norm(dq):.3f}")))


def main():
    rclpy.init()
    rclpy.spin(CartesianCBFController())


if __name__ == '__main__':
    main()
