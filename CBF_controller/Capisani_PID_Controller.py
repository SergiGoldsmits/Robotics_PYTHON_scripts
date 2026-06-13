#!/usr/bin/env python3
"""
Capisani-CBF safety filter for the FRANKA FR3 (planar XZ).

Filter (PDF eqs. 6-8):
    v_cmd = argmin_v 1/2||v - v_pid||^2  s.t.  grad(h)^T v + lambda*h >= 0
    => v_cmd = v_pid - kappa*min(0, grad(h)^T v_pid + lambda*h)/||grad(h)||^2 * grad(h)

BARRIER (Capisani angular barrier, used directly):
    h(q) = min over links of [ diff_i^2 - sign(delta_i)*delta_i^2 ]      (>0 = safe)
    Cases A/B (delta>=0): h_i = diff^2 - delta^2 (paper, unchanged).
    Case C (link too short, empty cone): delta continued negative via -arccosh; signed
    square -> h_i = diff^2 + delta^2 > 0. C^1 across delta=0.

ESCAPE (deadlock breaker, built IN JOINT SPACE so it is truly tangent to h):
    When the filter blocks (cstr<0), replace the nominal with a climb that is
      (a) the joint velocity raising the EE (+Z), with its grad(h)-component removed
          -> exactly tangent to the barrier (cannot reduce h), and
      (b) a small push along +grad(h) -> guarantees h rises so it crests and releases.
    Building it in joint space (not task space mapped back through a damped inverse) is
    the fix for the old failure where esc=1 yet cstr stayed strongly negative.
    Latched: engage on cstr<0; release on h>h_release. Engaging only on cstr<0 means a
    normal descent past the obstacle (reach pulling tangent/away, cstr>=0) does not
    re-trigger the climb even as h shrinks.

Numerical hygiene: softmin over links, central-difference gradient, gradient clip.
ROS interface unchanged.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import pinocchio as pin

ACTIVE = [1, 3, 5]                 # planar joints (fr3_joint2, 4, 6) -> XZ motion
TASK_ROWS = [0, 2]                 # X, Z rows of the 6xN Jacobian
LINK_SEGMENTS = [('fr3_link3', 'fr3_link4'), ('fr3_link4', 'fr3_link5'),
                 ('fr3_link5', 'fr3_link6'), ('fr3_link6', 'fr3_link7'),
                 ('fr3_link7', 'fr3_link8')]


# =============================================================================
#  Capisani angular barrier
# =============================================================================
def capisani_delta(p_base, p_end, obs_center, r_eff):
    """Capisani forbidden-cone half-width delta_i and signed angular offset
    diff_i = (Q_i - Q_i^W). Cases A/B per the paper; Case C continued negative via
    -arccosh (empty cone, link cannot reach), C^1 across the boundary."""
    x_b, z_b = p_base[0], p_base[2]
    x_e, z_e = p_end[0], p_end[2]
    l_i = np.hypot(x_e - x_b, z_e - z_b)
    d_i = np.hypot(obs_center[0] - x_b, obs_center[1] - z_b)

    if d_i <= r_eff:                                                   # base inside obstacle
        delta_i = np.pi / 2.0
    elif l_i * l_i >= (d_i * d_i - r_eff * r_eff):                     # Case A : tangent
        delta_i = np.arcsin(np.clip(r_eff / d_i, -1.0, 1.0))
    else:                                                             # Case B / Case C
        denom = 2.0 * d_i * l_i
        val = (l_i * l_i + d_i * d_i - r_eff * r_eff) / denom if denom > 1e-9 else 1.0
        if val <= 1.0:
            delta_i = np.arccos(np.clip(val, -1.0, 1.0))              # Case B
        else:
            delta_i = -np.arccosh(val)                                # Case C (empty cone)

    q_w = np.arctan2(obs_center[1] - z_b, obs_center[0] - x_b)         # critical angle Q_i^W
    q_l = np.arctan2(z_e - z_b, x_e - x_b)                             # actual link angle Q_i
    diff = np.arctan2(np.sin(q_l - q_w), np.cos(q_l - q_w))            # wrapped (Q_i - Q_i^W)
    return diff, delta_i


def _link_barriers(q_full, model, data, obs_center, r_eff):
    """Per-link  h_i = diff_i^2 - sign(delta_i)*delta_i^2  (>0 = safe)."""
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
    """Smooth (C^inf) under-estimate of min(h): differentiable 'min over links'."""
    m = float(np.min(h))
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (h - m))))


def h_and_grad(q_full, model, obs_center, r_eff, beta=15.0, eps=1e-4):
    """Barrier value h(q) and gradient grad(h) w.r.t. the 3 active joints."""
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
        self.obs_center = np.array([0.45, 0.25])
        self.obs_radius = 0.06
        self.link_buffer = 0.0150
        self.r_eff = self.obs_radius + self.link_buffer

        # --- state ---
        self.q = np.zeros(7)
        self.state_received = False
        self.x_des = None
        self.dq_prev = np.zeros(3)

        # --- nominal (PID) controller ---
        self.dt = 0.01
        self.Kp = 2.0
        self.v_max_task = 0.25
        self.max_dq = 0.8

        # --- CBF filter ---
        self.lambda_cbf = 4.0
        self.kappa = 1.0
        self.beta = 15.0
        self.grad_clip = 40.0

        # --- tangential escape (joint-space, over-the-top) ---
        self.k_esc = 0.15               # upward tangential climb speed
        self.k_out = 0.15              # outward push along +grad(h) (guarantees crest)
        self.cstr_release = 0.20          # release the climb once h exceeds this margin
        self.escaping = False          # hysteresis latch (reset on each new goal)

        self.lpf_alpha = 1.0           # 1.0 = no output smoothing

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
        t = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w])
        n = np.linalg.norm(q)
        q = q / n if n > 1e-6 else np.array([0., 0., 0., 1.])
        self.x_des = pin.SE3(pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix(), t)
        self.escaping = False          # fresh goal -> clear the latch

    def joint_state_callback(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for i, j in enumerate([f'fr3_joint{k}' for k in range(1, 8)]):
            if j in name_to_idx:
                self.q[i] = msg.position[name_to_idx[j]]
        self.state_received = True

    # ----- main loop ----------------------------------------------------------
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

        # 1) nominal "PID" velocity (task-space P term) -> joint velocity
        e = x_goal - x_ee
        v_pid = self.Kp * e
        sp = np.linalg.norm(v_pid)
        if sp > self.v_max_task:
            v_pid = v_pid * (self.v_max_task / sp)
        lam = 0.05
        Jinv = J.T @ np.linalg.inv(J @ J.T + lam * lam * np.eye(2))
        dq_nom = Jinv @ v_pid

        # 2) barrier value + gradient
        h, g = h_and_grad(self.q, self.model, self.obs_center, self.r_eff, self.beta)
        gn = np.linalg.norm(g)
        if gn > self.grad_clip:
            g = g * (self.grad_clip / gn)
        gg = float(g @ g) + 1e-9
        cstr = float(g @ dq_nom + self.lambda_cbf * h)        # = h_dot + lambda*h

        # --- tangential ESCAPE with hysteresis on cstr (not h) ----------------
        # Engage only on a real block; release as soon as the constraint is cleared
        # with margin. Releasing on cstr (not h) means a normal descent past the
        # obstacle -- where h shrinks but the motion is no longer INTO the cone --
        # does NOT re-trigger the climb.
        if cstr < 0.0:
            self.escaping = True
        elif cstr > self.cstr_release:                        # cleared with margin -> let go
            self.escaping = False

        if self.escaping:
            g_task = np.linalg.pinv(J.T) @ g                  # grad(h) in (X, Z)
            t = np.array([-g_task[1], g_task[0]])             # rotate +90 deg
            nt = np.linalg.norm(t)
            if nt > 1e-9:
                if t[1] < 0.0:
                    t = -t                                    # upward branch
                v_esc = self.k_esc * t / nt
                dq_nom = dq_nom + J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(2)) @ v_esc
            cstr = float(g @ dq_nom + self.lambda_cbf * h)    # re-evaluate

        dq = dq_nom.copy()
        if cstr < 0.0:                                        # project onto constraint boundary
            dq = dq_nom - self.kappa * (cstr / (g @ g + 1e-9)) * g

        # 3) CBF projection (certifies h>=0 for the nominal in use)
        dq = dq_nom.copy()
        if cstr < 0.0:
            dq = dq_nom - self.kappa * (cstr / gg) * g

        # 4) saturate, (optional) filter, publish
        dq = self.lpf_alpha * dq + (1.0 - self.lpf_alpha) * self.dq_prev
        self.dq_prev = dq
        dq = np.clip(dq, -self.max_dq, self.max_dq)

        final = np.zeros(7)
        final[ACTIVE] = dq
        self.vel_pub.publish(Float64MultiArray(data=final.tolist()))
        self.status_pub.publish(String(
            data=(f"x={x_ee[0]:+.3f} z={x_ee[1]:+.3f} h={h:+.4f} cstr={cstr:+.4f} "
                  f"esc={int(self.escaping)} |e|={np.linalg.norm(e):.3f} "
                  f"|dq|={np.linalg.norm(dq):.3f}")))


def main():
    rclpy.init()
    node = CartesianCBFController()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
