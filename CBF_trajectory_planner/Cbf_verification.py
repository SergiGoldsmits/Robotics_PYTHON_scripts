#!/usr/bin/env python3
"""
cbf_verification.py
Verifies that the geometric CBF barrier h(q) zero crossing matches
the analytical Capisani forbidden region boundary.

Run from the scripts directory:
    python3 cbf_verification.py

Produces 2 figures:
  Figure 1 — three panels (q5=0):
    - h(q) heatmap with zero crossing
    - Capisani analytical unsafe region
    - Overlay — boundary comparison (key verification plot)

  Figure 2 — q1 vs q3 at three q5 values:
    - Shows how the forbidden region changes with wrist pose
    - Top row: h(q) heatmap
    - Bottom row: Capisani + h=0 overlay
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_space_designer_analytical import FR3PlanarSystem

# ── Parameters ────────────────────────────────────────────────────────────────
OBS_CENTER = [0.3, 0.6]
OBS_RADIUS = 0.13
Q5_FIXED   = 0.0     # radians — fixed slice for Figure 1
RES        = 80      # grid resolution

robot      = FR3PlanarSystem(obs_center=OBS_CENTER, obs_radius=OBS_RADIUS)
q_range    = np.linspace(-np.pi, np.pi, RES)
Q1g, Q3g   = np.meshgrid(q_range, q_range, indexing='ij')

def build_grid(q5_val):
    """Compute h(q) and Capisani unsafe flag over q1-q3 grid."""
    H      = np.zeros((RES, RES))
    unsafe = np.zeros((RES, RES), dtype=bool)
    for i, q1 in enumerate(q_range):
        for j, q3 in enumerate(q_range):
            q            = np.degrees([q1, q3, q5_val])
            H[i, j]      = robot.get_cbf_h(q)
            unsafe[i, j] = robot.is_unsafe_paper_method(q)
    return H, unsafe

def overlay_legend(ax):
    ax.legend(handles=[
        Patch(facecolor='#CC0000', alpha=0.5, label='Capisani unsafe'),
        Patch(facecolor='#003399', alpha=0.3, label='Capisani safe'),
        Line2D([0], [0], color='black', lw=2.5,
               ls='--', label='h(q) = 0'),
    ], fontsize=8, loc='upper right')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — single slice q5=0, three panels
# ══════════════════════════════════════════════════════════════════════════════
print(f"Building Figure 1  q5={np.degrees(Q5_FIXED):.0f} deg ...")
H2, unsafe2 = build_grid(Q5_FIXED)

fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle(
    f'CBF verification — q1 vs q3  (q5 = {np.degrees(Q5_FIXED):.0f} deg)\n'
    f'Obstacle: centre={OBS_CENTER}  radius={OBS_RADIUS} m',
    fontsize=12)

# Panel 1: h(q) heatmap
ax = axes[0]
cf = ax.contourf(Q1g, Q3g, H2, levels=50, cmap='RdYlGn')
ax.contour(Q1g, Q3g, H2, levels=[0.0],
           colors='black', linewidths=2.5)
plt.colorbar(cf, ax=ax, label='h(q) [m]')
ax.set_title('h(q) — geometric CBF barrier\nblack line = h=0')
ax.set_xlabel('q1 [rad]'); ax.set_ylabel('q3 [rad]')
ax.set_aspect('equal')

# Panel 2: Capisani analytical region
ax = axes[1]
ax.contourf(Q1g, Q3g, unsafe2.astype(float),
            levels=[0.5, 1.5], colors=['#CC0000'], alpha=0.6)
ax.contourf(Q1g, Q3g, (~unsafe2).astype(float),
            levels=[0.5, 1.5], colors=['#003399'], alpha=0.3)
ax.set_title('Capisani analytical mapping\nred = unsafe,  blue = safe')
ax.set_xlabel('q1 [rad]'); ax.set_ylabel('q3 [rad]')
ax.set_aspect('equal')

# Panel 3: overlay — key verification
ax = axes[2]
ax.contourf(Q1g, Q3g, unsafe2.astype(float),
            levels=[0.5, 1.5], colors=['#CC0000'], alpha=0.5)
ax.contourf(Q1g, Q3g, (~unsafe2).astype(float),
            levels=[0.5, 1.5], colors=['#003399'], alpha=0.2)
cs = ax.contour(Q1g, Q3g, H2, levels=[0.0],
                colors='black', linewidths=3, linestyles='--')
ax.clabel(cs, fmt='h=0', fontsize=9)
ax.set_title('Verification: Capisani region\nvs h=0 boundary (black dashed)\n'
             'offset = link radius enlargement')
ax.set_xlabel('q1 [rad]'); ax.set_ylabel('q3 [rad]')
ax.set_aspect('equal')
overlay_legend(ax)

plt.tight_layout()
plt.savefig('/tmp/cbf_verification_fig1.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cbf_verification_fig1.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — q1 vs q3 at three q5 values
# ══════════════════════════════════════════════════════════════════════════════
q5_slices = [
    (-1.0, 'q5 = -57 deg  (wrist back)'),
    ( 0.0, 'q5 =   0 deg  (wrist neutral)'),
    ( 1.0, 'q5 = +57 deg  (wrist forward)'),
]

for fignum, (q5_val, q5_label) in enumerate(q5_slices):
    deg   = int(round(np.degrees(q5_val)))
    wrist = 'back' if q5_val < 0 else ('forward' if q5_val > 0 else 'neutral')

    print(f"Building Figure {fignum+2}  q5={deg} deg ({wrist}) ...")
    H_sl, unsafe_sl = build_grid(q5_val)

    pct = 100 * unsafe_sl.sum() / (RES * RES)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f'C-space — q1 vs q3   (q5 = {deg}°, wrist {wrist})\n'
        f'Obstacle: centre={OBS_CENTER}   radius={OBS_RADIUS} m',
        fontsize=13)

    # left: h(q) heatmap
    cf = ax_l.contourf(Q1g, Q3g, H_sl, levels=50, cmap='RdYlGn')
    ax_l.contour(Q1g, Q3g, H_sl, levels=[0.0],
                 colors='black', linewidths=2.5)
    plt.colorbar(cf, ax=ax_l, label='h(q) [m]')
    ax_l.set_title('h(q) — CBF barrier function\nblack line = h=0 boundary',
                   fontsize=12)
    ax_l.set_xlabel('q1 [rad]', fontsize=11)
    ax_l.set_ylabel('q3 [rad]', fontsize=11)
    ax_l.set_aspect('equal')
    ax_l.text(0.03, 0.04, f'forbidden: {pct:.1f}%',
              transform=ax_l.transAxes, fontsize=10, color='white',
              bbox=dict(boxstyle='round', facecolor='#CC0000', alpha=0.85))

    # right: overlay
    ax_r.contourf(Q1g, Q3g, unsafe_sl.astype(float),
                  levels=[0.5, 1.5], colors=['#CC0000'], alpha=0.5)
    ax_r.contourf(Q1g, Q3g, (~unsafe_sl).astype(float),
                  levels=[0.5, 1.5], colors=['#003399'], alpha=0.2)
    ax_r.contour(Q1g, Q3g, H_sl, levels=[0.0],
                 colors='black', linewidths=2.5, linestyles='--')
    ax_r.set_title('Capisani analytical region\nvs h=0 boundary (black dashed)',
                   fontsize=12)
    ax_r.set_xlabel('q1 [rad]', fontsize=11)
    ax_r.set_ylabel('q3 [rad]', fontsize=11)
    ax_r.set_aspect('equal')
    overlay_legend(ax_r)

    plt.tight_layout()
    fname = f'/tmp/cbf_verification_q5_{deg}deg.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved: {fname}")

plt.show()
print("\nDone.")