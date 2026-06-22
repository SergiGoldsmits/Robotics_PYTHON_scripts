#!/usr/bin/env python3
"""
Thesis plotting for the Capisani-CBF controller.

SCHEMA-AWARE: works with BOTH logs and tells you which one it loaded.
  * planar / latch  CSV  (cbf_trajectory_data.csv):     columns include `escaping`,
                                                        3 joints, x_ee/z_ee.
  * 3D / circulation CSV (cbf_trajectory_data_3d.csv):  columns include `h_hardmin`,
                                                        `cstr`, `feasible`, `circ_mag`,
                                                        7 joints, x_ee/y_ee/z_ee.

WHY THIS MATTERS: the original script hard-coded /tmp/cbf_trajectory_data.csv (the planar
file). If you ran the 3D controller (which writes *_3d.csv) but plotted the old path, you
were plotting a stale latch run -- that is exactly why an "Escape Latch Engaged" panel
appears even though the 3D code never logs `escaping`. This script auto-detects the file,
prints its path + modification time + schema, and adapts every figure.

DIAGNOSTIC ADDITIONS (3D schema):
  * Fig 1 overlays h_hardmin (the TRUE min link barrier) on h (the softmin). The softmin
    sits up to ln(n_links)/beta below the hard min, so this pair tells you instantly
    whether a sub-zero dip is a real violation (h_hardmin<0, shaded red) or just softmin
    bias (h<0 but h_hardmin>=0).
  * Fig 1 overlays cstr (the realised constraint g.dq + lambda*h). cstr>=0 while
    h_hardmin<0 is the certified-vs-executed gap (e.g. tighter caps in the C++ layer):
    the QP did its job on the command, but the robot didn't execute it.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  LOAD  --  auto-detect the newest available log unless DATA_PATH is forced
# =============================================================================
DATA_PATH = None  # set a string to force a specific file; None = auto-detect
CANDIDATES = ['/tmp/cbf_trajectory_data_3d.csv', '/tmp/cbf_trajectory_data.csv']

if DATA_PATH is None:
    existing = [(c, os.path.getmtime(c)) for c in CANDIDATES if os.path.exists(c)]
    if not existing:
        raise FileNotFoundError(f"None of the candidate logs exist: {CANDIDATES}")
    DATA_PATH = max(existing, key=lambda kv: kv[1])[0]   # newest by mtime

df = pd.read_csv(DATA_PATH)
cols = set(df.columns)

# --- schema introspection ----------------------------------------------------
N_JOINTS   = sum(1 for i in range(20) if f'q{i}' in cols)
HAS_HARDMIN = 'h_hardmin' in cols
HAS_CSTR    = 'cstr' in cols
HAS_FEAS    = 'feasible' in cols
HAS_CIRC    = 'circ_mag' in cols
HAS_ESCAPE  = 'escaping' in cols
IS_3D       = 'y_ee' in cols
SCHEMA      = '3D / circulation' if (HAS_CIRC or IS_3D or HAS_HARDMIN) else 'planar / latch'

import time as _time
print("=" * 70)
print(f"[load] file   : {DATA_PATH}")
print(f"[load] modified: {_time.ctime(os.path.getmtime(DATA_PATH))}")
print(f"[load] schema : {SCHEMA}   (joints={N_JOINTS}, 3D={IS_3D})")
print(f"[load] extras : hardmin={HAS_HARDMIN} cstr={HAS_CSTR} feasible={HAS_FEAS} "
      f"circ={HAS_CIRC} escaping={HAS_ESCAPE}")
print("=" * 70)

# =============================================================================
#  CROP  --  the apex-crossing window (tune to your scenario)
# =============================================================================
condition = (df['x_ee'] > 0.301) & (df['x_ee'] < 0.599)
active_df = df[condition].copy()
if active_df.empty:
    print("[crop] condition matched no rows; using the entire dataframe.")
    active_df = df.copy()

time_raw = active_df['timestamp'].values
time_axis = time_raw - time_raw[0] if len(time_raw) > 0 else time_raw

# --- common arrays ------------------------------------------------------------
h_vals = active_df['h'].values
hardmin_vals = active_df['h_hardmin'].values if HAS_HARDMIN else None
cstr_vals = active_df['cstr'].values if HAS_CSTR else None
correction_vals = active_df['correction_mag'].values if 'correction_mag' in cols else \
                  np.zeros(len(active_df))
circ_vals = active_df['circ_mag'].values if HAS_CIRC else None
feas_vals = pd.to_numeric(active_df['feasible'], errors='coerce').fillna(1).values \
            if HAS_FEAS else None

# Generic "filter active" mask: latch if present, else circulation/correction activity.
if HAS_ESCAPE:
    active_mask = (active_df['escaping'].astype(str).str.strip().str.lower()
                   .isin(['1', '1.0', 'true', 't', 'yes']).values)
elif HAS_CIRC:
    active_mask = (circ_vals > 1e-3)
else:
    active_mask = (correction_vals > 1e-3)

# Real-violation mask: prefer the true hard min; fall back to softmin with a warning.
if HAS_HARDMIN:
    violation_mask = (hardmin_vals < 0.0)
else:
    violation_mask = (h_vals < 0.0)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# =============================================================================
#  PINOCCHIO INERTIA CONFIG (Fig 4 Panel C; planar 3-joint schema only)
# =============================================================================
URDF_PATH = '/home/sergi/ros2_ws/src/1.0_Thesis/arm_description/urdf/expanded_fr3.urdf'
ACTIVE_Q_MODEL_INDICES = [1, 3, 5]                       # planar logged joints -> model idx
HELD_JOINT_POSE = {0: 0.0, 2: 0.0, 4: 0.0, 6: 0.7854}    # planar held joints (replace!)

# =============================================================================
#  FIGURE 1: SAFETY MANIFOLD + DIAGNOSTICS
# =============================================================================
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 1. Barrier: softmin h, true hard-min, unsafe boundary, real-violation shading
ax1.plot(time_axis, h_vals, color='darkblue', linewidth=2, label=r'$h(q)$ (softmin)')
if HAS_HARDMIN:
    ax1.plot(time_axis, hardmin_vals, color='slateblue', linewidth=1.3, linestyle='-.',
             label=r'$\min_i h_i$ (true)')
ax1.axhline(0, color='red', linestyle='--', alpha=0.7, label='Unsafe Boundary')
if violation_mask.any():
    ax1.fill_between(time_axis, ax1.get_ylim()[0], 0.0, where=violation_mask, step='post',
                     color='red', alpha=0.15,
                     label=('TRUE violation' if HAS_HARDMIN else 'softmin < 0'))
ax1.set_ylabel('Barrier Value')
ax1.set_title('Multi-Body Safety Manifold Evolution Throughout Apex Crossing')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=8)

# 2. Correction norm (left) + realised constraint cstr (right twin) for the gap diagnosis
ax2.plot(time_axis, correction_vals, color='crimson', linewidth=2,
         label=r'$\|\dot{q}-\dot{q}_{nom}\|$')
ax2.set_ylabel('Correction Norm [rad/s]')
ax2.set_title('Minimal-Invasiveness Projection + Certified Constraint')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', fontsize=8)
if HAS_CSTR:
    ax2c = ax2.twinx()
    ax2c.plot(time_axis, cstr_vals, color='darkorange', linewidth=1.3, alpha=0.85,
              label=r'$g\cdot\dot{q}+\lambda h$ (cstr)')
    ax2c.axhline(0, color='orange', linestyle=':', alpha=0.6)
    ax2c.set_ylabel(r'Certified constraint', color='darkorange')
    ax2c.tick_params(axis='y', labelcolor='darkorange')
    ax2c.legend(loc='upper right', fontsize=8)

# 3. Escape state: latch step (planar) OR circulation magnitude (3D) OR infeasibility
if HAS_ESCAPE:
    esc_num = pd.to_numeric(active_df['escaping'], errors='coerce').fillna(0).values
    ax3.step(time_axis, esc_num, color='orange', alpha=0.8, where='post',
             label='Escape Latch Engaged')
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(['0 (Nominal)', '1 (Escape)'])
    ax3.set_ylabel('Discrete State')
    ax3.set_title('Active Joint-Space Tangential Escape States')
elif HAS_CIRC:
    ax3.plot(time_axis, circ_vals, color='orange', linewidth=2, label=r'$\|v_{circ}\|$')
    if HAS_FEAS and (feas_vals < 0.5).any():
        ax3.fill_between(time_axis, 0.0, np.nanmax(circ_vals) if len(circ_vals) else 1.0,
                         where=(feas_vals < 0.5), step='post', color='red', alpha=0.18,
                         label='QP infeasible')
    ax3.set_ylabel('Circulation [m/s]')
    ax3.set_title('Smooth Circulation Magnitude (latch replaced)')
else:
    ax3.text(0.5, 0.5, 'no escape/circulation column in this log',
             ha='center', va='center', transform=ax3.transAxes)
ax3.set_xlabel('Normalized Profile Time [s]')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right', fontsize=8)

fig1.tight_layout()
fig1.savefig('/tmp/thesis_cbf_performance_metrics.png', dpi=300)
print("Figure 1 (Safety + diagnostics) -> /tmp/thesis_cbf_performance_metrics.png")

# =============================================================================
#  FIGURE 2: JOINT KINEMATICS (generalised to N_JOINTS)
# =============================================================================
if N_JOINTS == 3:
    joint_labels = ['Joint 2', 'Joint 4', 'Joint 6']         # planar active set
else:
    joint_labels = [f'Joint {i+1}' for i in range(N_JOINTS)]
cmap = plt.get_cmap('tab10' if N_JOINTS <= 10 else 'tab20')
colors = [cmap(i % cmap.N) for i in range(N_JOINTS)]

fig2, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
for i, (label, col) in enumerate(zip(joint_labels, colors)):
    axs[0].plot(time_axis, active_df[f'q{i}'].values, color=col, linewidth=1.8, label=label)
axs[0].set_ylabel('Position [rad]')
axs[0].set_title(f'Active Joint Coordinate Tracking ({N_JOINTS} joints)')
axs[0].grid(True, linestyle=':', alpha=0.6)
axs[0].legend(loc='best', fontsize=8, ncol=2)
for i, (label, col) in enumerate(zip(joint_labels, colors)):
    axs[1].plot(time_axis, active_df[f'dq{i}'].values, color=col, linewidth=1.8, label=label)
axs[1].set_ylabel('Velocity [rad/s]')
axs[1].set_title('Filtered Joint Velocity Commands')
axs[1].grid(True, linestyle=':', alpha=0.6)
for i, (label, col) in enumerate(zip(joint_labels, colors)):
    axs[2].plot(time_axis, active_df[f'ddq{i}'].values, color=col, linewidth=1.3, alpha=0.85,
                label=label)
axs[2].set_xlabel('Normalized Profile Time [s]')
axs[2].set_ylabel('Acceleration [rad/s²]')
axs[2].set_title('Derived Numerical Joint Accelerations')
axs[2].grid(True, linestyle=':', alpha=0.6)

fig2.tight_layout()
fig2.savefig('/tmp/thesis_joint_kinematics.png', dpi=300)
print("Figure 2 (Joint kinematics) -> /tmp/thesis_joint_kinematics.png")

# =============================================================================
#  FIGURE 3: EE TRAJECTORY  --  correct projection(s) for the schema
#  Vertical circulation axis => avoidance happens in X-Y. An X-Z view alone would
#  look like the EE drives through the obstacle, so for 3D we show X-Y (top-down,
#  the swirl plane) AND X-Z (side).
# =============================================================================
OBS_3D = (0.45, 0.0, 0.25)        # X, Y, Z (match controller obs_center)
OBS_RADIUS = 0.06
LINK_BUFFER = 0.025

x_full = df['x_ee'].values
z_full = df['z_ee'].values
y_full = df['y_ee'].values if IS_3D else None


def _draw_obstacle(ax, cx, cy):
    ax.add_patch(plt.Circle((cx, cy), OBS_RADIUS, color='red', alpha=0.45, label='Obstacle (R)'))
    ax.add_patch(plt.Circle((cx, cy), OBS_RADIUS + LINK_BUFFER, color='red', alpha=0.9,
                            fill=False, linestyle='--', linewidth=1.5,
                            label=r'$r_{eff}=R+$buffer'))


if IS_3D and y_full is not None and (np.nanmax(y_full) - np.nanmin(y_full)) > 1e-3:
    fig3, (axxy, axxz) = plt.subplots(1, 2, figsize=(12, 5.5))
    # X-Y: the plane the vertical-axis circulation actually goes around in
    axxy.plot(x_full, y_full, color='darkblue', linewidth=2, label='EE path')
    _draw_obstacle(axxy, OBS_3D[0], OBS_3D[1])
    axxy.scatter(x_full[0], y_full[0], c='green', s=70, zorder=5, label='start')
    axxy.scatter(x_full[-1], y_full[-1], c='black', s=70, zorder=5, label='goal')
    axxy.set_xlabel('x [m]'); axxy.set_ylabel('y [m]'); axxy.set_aspect('equal')
    axxy.set_title('Top-down X-Y  (circulation/swirl plane)')
    axxy.grid(True, linestyle=':', alpha=0.6); axxy.legend(loc='best', fontsize=8)
    # X-Z: side view
    axxz.plot(x_full, z_full, color='darkblue', linewidth=2, label='EE path')
    _draw_obstacle(axxz, OBS_3D[0], OBS_3D[2])
    axxz.scatter(x_full[0], z_full[0], c='green', s=70, zorder=5, label='start')
    axxz.scatter(x_full[-1], z_full[-1], c='black', s=70, zorder=5, label='goal')
    axxz.set_xlabel('x [m]'); axxz.set_ylabel('z [m]'); axxz.set_aspect('equal')
    axxz.set_title('Side X-Z')
    axxz.grid(True, linestyle=':', alpha=0.6); axxz.legend(loc='best', fontsize=8)
else:
    fig3, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_full, z_full, color='darkblue', linewidth=2, label='EE path')
    _draw_obstacle(ax, OBS_3D[0], OBS_3D[2])
    if len(x_full):
        ax.scatter(x_full[0], z_full[0], c='green', s=70, zorder=5, label='start')
        ax.scatter(x_full[-1], z_full[-1], c='black', s=70, zorder=5, label='goal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]'); ax.set_aspect('equal')
    ax.set_title('End-Effector Trajectory in the X-Z Plane')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='best', fontsize=8)

fig3.tight_layout()
fig3.savefig('/tmp/thesis_ee_trajectory.png', dpi=300)
print("Figure 3 (EE trajectory) -> /tmp/thesis_ee_trajectory.png")

# =============================================================================
#  FIGURE 4: LYAPUNOV / ENERGY  (V uses full spatial error; KE generalised)
# =============================================================================
x_ee = active_df['x_ee'].values
z_ee = active_df['z_ee'].values
y_ee = active_df['y_ee'].values if IS_3D else None

x_goal = x_full[-1] if len(x_full) else 0.0
z_goal = z_full[-1] if len(z_full) else 0.0
y_goal = (y_full[-1] if (IS_3D and y_full is not None and len(y_full)) else 0.0)

ex = x_goal - x_ee
ez = z_goal - z_ee
ey = (y_goal - y_ee) if IS_3D else np.zeros_like(x_ee)
V = 0.5 * (ex**2 + ey**2 + ez**2)

if len(time_axis) > 1:
    xdot = np.gradient(x_ee, time_axis)
    zdot = np.gradient(z_ee, time_axis)
    ydot = np.gradient(y_ee, time_axis) if IS_3D else np.zeros_like(x_ee)
else:
    xdot = np.zeros_like(x_ee); zdot = np.zeros_like(z_ee); ydot = np.zeros_like(x_ee)
Vdot = -(ex * xdot + ey * ydot + ez * zdot)

dq_cols = active_df[[f'dq{i}' for i in range(N_JOINTS)]].values
q_cols = active_df[[f'q{i}' for i in range(N_JOINTS)]].values


def compute_kinetic_energy(q_active, dq_active, n_joints):
    """True 1/2 qdot^T M(q) qdot via pinocchio if possible, else 1/2||qdot||^2 proxy."""
    try:
        import pinocchio as pin
    except Exception as e:
        print(f"[KE] pinocchio not importable ({e}); using 1/2||qdot||^2 proxy.")
        return 0.5 * np.sum(dq_active**2, axis=1), 'proxy'
    try:
        model = pin.buildModelFromUrdf(URDF_PATH)
        data = model.createData()
        q_base = pin.neutral(model)
        if n_joints == model.nv:                         # all joints logged (3D case)
            active_idx = list(range(n_joints))
        else:                                            # planar: held pose + active subset
            for idx, val in HELD_JOINT_POSE.items():
                if idx < model.nq:
                    q_base[idx] = val
            active_idx = ACTIVE_Q_MODEL_INDICES
        if max(active_idx) >= model.nv:
            raise IndexError(f"active idx {active_idx} exceed model.nv={model.nv}")
        ke = np.empty(len(q_active))
        for t in range(len(q_active)):
            q = q_base.copy(); v = np.zeros(model.nv)
            for k, mi in enumerate(active_idx):
                q[mi] = q_active[t, k]; v[mi] = dq_active[t, k]
            pin.crba(model, data, q)
            M = np.array(data.M); M = np.triu(M) + np.triu(M, 1).T
            ke[t] = 0.5 * float(v @ M @ v)
        print(f"[KE] true 1/2 qdot^T M(q) qdot from {URDF_PATH} (nv={model.nv}).")
        return ke, 'true'
    except Exception as e:
        print(f"[KE] pinocchio path failed ({e}); using 1/2||qdot||^2 proxy.")
        return 0.5 * np.sum(dq_active**2, axis=1), 'proxy'


KE, KE_MODE = compute_kinetic_energy(q_cols, dq_cols, N_JOINTS)
if KE_MODE == 'true':
    KE_LABEL = r'$\frac{1}{2}\,\dot{q}^\top M(q)\,\dot{q}$'
    KE_YLABEL = 'Kinetic Energy [J]'
    KE_TITLE = r'Joint-Space Kinetic Energy $\frac{1}{2}\dot{q}^\top M(q)\dot{q}$ (bounded transient)'
else:
    KE_LABEL = r'$\frac{1}{2}\|\dot{q}\|^2$'
    KE_YLABEL = r'Joint KE [rad$^2$/s$^2$]'
    KE_TITLE = (r'Joint KE proxy $\frac{1}{2}\|\dot{q}\|^2$ '
                '(M(q) unavailable — control effort, not a stability proof)')

fig4, (axA, axB, axC) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
axA.plot(time_axis, V, color='darkgreen', linewidth=2,
         label=r'$V=\frac{1}{2}\|x_{goal}-x\|^2$')
if active_mask.any():
    axA.fill_between(time_axis, 0.0, float(np.max(V)) if len(V) else 1.0,
                     where=active_mask, step='post', color='orange', alpha=0.15,
                     label='filter active')
axA.set_ylabel(r'$V$ [m$^2$]')
axA.set_title(r'Task-Space Lyapunov Function $V=\frac{1}{2}\|x_{goal}-x\|^2$'
              + ('  (full 3D error)' if IS_3D else ''))
axA.grid(True, linestyle=':', alpha=0.6); axA.legend(loc='upper right', fontsize=8)

axB.plot(time_axis, Vdot, color='purple', linewidth=2, label=r'$\dot{V}=-e^\top\dot{x}$')
axB.axhline(0, color='red', linestyle='--', alpha=0.7, label=r'$\dot{V}=0$')
if active_mask.any() and len(Vdot):
    axB.fill_between(time_axis, float(np.min(Vdot)), float(np.max(Vdot)),
                     where=active_mask, step='post', color='orange', alpha=0.15,
                     label='filter active')
axB.set_ylabel(r'$\dot{V}$ [m$^2$/s]')
axB.set_title(r'Lyapunov Decrease Rate ($\dot{V}<0\Rightarrow$ locally asymptotically stable)')
axB.grid(True, linestyle=':', alpha=0.6); axB.legend(loc='upper right', fontsize=8)

axC.plot(time_axis, KE, color='teal', linewidth=2, label=KE_LABEL)
if active_mask.any():
    axC.fill_between(time_axis, 0.0, float(np.max(KE)) if len(KE) else 1.0,
                     where=active_mask, step='post', color='orange', alpha=0.15,
                     label='filter active')
axC.set_xlabel('Normalized Profile Time [s]')
axC.set_ylabel(KE_YLABEL); axC.set_title(KE_TITLE)
axC.grid(True, linestyle=':', alpha=0.6); axC.legend(loc='upper right', fontsize=8)

fig4.tight_layout()
fig4.savefig('/tmp/thesis_lyapunov_energy.png', dpi=300)
print("Figure 4 (Lyapunov + energy) -> /tmp/thesis_lyapunov_energy.png")

# --- console summary: the one-line verdict on the safety question -------------
if HAS_HARDMIN:
    worst = float(np.min(hardmin_vals)) if len(hardmin_vals) else float('nan')
    print(f"[verdict] worst TRUE barrier min_i h_i = {worst:+.4f}  "
          f"({'VIOLATION' if worst < 0 else 'safe'})")
    if HAS_CSTR and worst < 0:
        gap = (cstr_vals[violation_mask] >= 0).mean() if violation_mask.any() else 0.0
        print(f"[verdict] of the violating ticks, {100*gap:.0f}% had cstr>=0 "
              f"-> certified-vs-executed gap (check C++ caps).")
else:
    worst = float(np.min(h_vals)) if len(h_vals) else float('nan')
    print(f"[verdict] worst softmin h = {worst:+.4f} (no h_hardmin column; "
          f"true min is up to ln(n)/beta higher).")

plt.show()
