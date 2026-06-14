#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the recorded data from the original /tmp directory
data_path = '/tmp/cbf_trajectory_data.csv'
df = pd.read_csv(data_path)

# --- Clean/Crop Data for your Thesis ---
# Captures the window where the robot actually started crossing the obstacle (x > 0.301)
# and before it fully settles on the other side
condition = (df['x_ee'] > 0.301) & (df['x_ee'] < 0.599)
active_df = df[condition].copy()

if active_df.empty:
    print("Warning: Crop condition returned no data. Using entire dataframe instead.")
    active_df = df.copy()

# Zero out the time axis
time_raw = active_df['timestamp'].values
time_axis = time_raw - time_raw[0] if len(time_raw) > 0 else time_raw

# Convert all critical data columns to flat 1D numpy arrays to bypass the Pandas 2.0 indexing bug
h_vals = active_df['h'].values
correction_vals = active_df['correction_mag'].values
escaping_vals = active_df['escaping'].values

# Academic paper font configurations
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# =============================================================================
#  FIGURE 1: SAFETY MANIFOLD PERFORMANCE METRICS
# =============================================================================
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 1. Plot the Composite Control Barrier Function h(q)
ax1.plot(time_axis, h_vals, color='darkblue', linewidth=2, label=r'$h(q)$')
ax1.axhline(0, color='red', linestyle='--', alpha=0.7, label='Unsafe Boundary')
ax1.set_ylabel('Barrier Value')
ax1.set_title('Multi-Body Safety Manifold Evolution Throughout Apex Crossing')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# 2. Plot the Control Invasiveness / Correction Magnitude
ax2.plot(time_axis, correction_vals, color='crimson', linewidth=2, label=r'$\|\dot{q} - \dot{q}_{nom}\|$')
ax2.set_ylabel('Correction Norm [rad/s]')
ax2.set_title('Minimal-Invasiveness Projection Correction Profiles')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# 3. Plot the Hysteresis Escape Engine Flags
ax3.step(time_axis, escaping_vals, color='orange', alpha=0.7, label='Escape Latch Engaged')
ax3.set_xlabel('Normalized Profile Time [s]')
ax3.set_ylabel('Discrete State')
ax3.set_title('Active Joint-Space Tangential Escape States')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['0 (Nominal Tracking)', '1 (Active Escape)'])
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')

fig1.tight_layout()
fig1.savefig('/tmp/thesis_cbf_performance_metrics.png', dpi=300)
print("Figure 1 (Safety Metrics) saved to: /tmp/thesis_cbf_performance_metrics.png")


# =============================================================================
#  FIGURE 2: ACTIVE JOINT KINEMATICS (POS, VEL, ACCEL)
# =============================================================================
fig2, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

joint_labels = ['Joint 2', 'Joint 4', 'Joint 6']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 1. Plot Active Joint Positions (q)
for i, (label, col) in enumerate(zip(joint_labels, colors)):
    axs[0].plot(time_axis, active_df[f'q{i}'].values, color=col, linewidth=2, label=label)
axs[0].set_ylabel('Position [rad]')
axs[0].set_title('Active Joint Coordinate Tracking Profiles')
axs[0].grid(True, linestyle=':', alpha=0.6)
axs[0].legend(loc='lower left')

# 2. Plot Command/Filtered Joint Velocities (dq)
for i, (label, col) in enumerate(zip(joint_labels, colors)):
    axs[1].plot(time_axis, active_df[f'dq{i}'].values, color=col, linewidth=2, label=label)
axs[1].set_ylabel('Velocity [rad/s]')
axs[1].set_title('Filtered Joint Velocity Control Commands')
axs[1].grid(True, linestyle=':', alpha=0.6)

# 3. Plot Derived Joint Accelerations (ddq)
for i, (label, col) in enumerate(zip(joint_labels, colors)):
    axs[2].plot(time_axis, active_df[f'ddq{i}'].values, color=col, linewidth=1.5, alpha=0.8, label=label)
axs[2].set_xlabel('Normalized Profile Time [s]')
axs[2].set_ylabel('Acceleration [rad/s²]')
axs[2].set_title('Derived Numerical Joint Accelerations')
axs[2].grid(True, linestyle=':', alpha=0.6)

fig2.tight_layout()
fig2.savefig('/tmp/thesis_joint_kinematics.png', dpi=300)
print("Figure 2 (Joint Kinematics) saved to: /tmp/thesis_joint_kinematics.png")

# Display both figure windows simultaneously on screen
plt.show()
