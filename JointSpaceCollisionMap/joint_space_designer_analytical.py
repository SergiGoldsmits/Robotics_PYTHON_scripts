import numpy as np
import matplotlib.pyplot as plt

class FR3PlanarSystem:
    def __init__(self):
        # Physical parameters: Base, Shoulder, Elbow, Wrist
        self.L = [0.333, 0.316, 0.384, 0.107]
        self.obs_center = np.array([0.2, 0.6])
        self.obs_radius = 0.10
        self.link_radii = [0.05, 0.05, 0.05, 0.02]

    def get_kinematics(self, q_deg):
        q = np.radians(q_deg)
        Q = [q[0], q[0]+q[1], q[0]+q[1]+q[2]]
        pts = [np.array([0.0, 0.0]), np.array([0.0, self.L[0]])]
        curr = pts[1].copy()
        for i in range(3):
            curr += self.L[i+1] * np.array([np.sin(Q[i]), np.cos(Q[i])])
            pts.append(curr.copy())
        return np.array(pts)

    def get_cbf_h(self, q_deg):
        """Analytical Distance Function for CBF Controller (Gradient-ready)."""
        pts = self.get_kinematics(q_deg)
        min_dist = float('inf')
        for i in range(3):
            p1, p2 = pts[i+1], pts[i+2]
            v = p2 - p1
            w = self.obs_center - p1
            t = np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-6), 0, 1)
            dist = np.linalg.norm(w - t * v) - (self.obs_radius + self.link_radii[i])
            min_dist = min(min_dist, dist)
        return min_dist

    def is_unsafe_paper_method(self, q_deg):
        """Analytical Mapping per Capisani et al. (for Thesis Visualization)."""
        pts = self.get_kinematics(q_deg)
        q = np.radians(q_deg)
        Q = [q[0], q[0]+q[1], q[0]+q[1]+q[2]]
        for i in range(3):
            d = np.linalg.norm(self.obs_center - pts[i+1])
            Qw = np.arctan2(self.obs_center[0] - pts[i+1][0], self.obs_center[1] - pts[i+1][1])
            if self.L[i+1]**2 >= d**2 - self.obs_radius**2:
                delta = np.arcsin(np.clip(self.obs_radius / d, -1, 1))
            elif (d - self.obs_radius)**2 < self.L[i+1]**2 <= d**2 - self.obs_radius**2:
                delta = np.arccos(np.clip((self.L[i+1]**2 + d**2 - self.obs_radius**2)/(2*d*self.L[i+1]), -1, 1))
            else: delta = 0
            if abs((Q[i] - Qw + np.pi) % (2 * np.pi) - np.pi) < delta: return True
        return False

# --- Visualization ---
mapper = FR3PlanarSystem()

# 1. Workspace
pts = mapper.get_kinematics([-45, 120, 90])
plt.figure(figsize=(6,6))
plt.plot(pts[:,0], pts[:,1], 'ko-', linewidth=3)
plt.gca().add_patch(plt.Circle(mapper.obs_center, mapper.obs_radius, color='red', alpha=0.5))
plt.title("Operational Workspace")

# 2. Configuration Space Mapping (Using Paper Method)
res = 50
q_range = np.linspace(-np.pi, np.pi, res)
safe, unsafe = [], []
for q1 in q_range:
    for q2 in q_range:
        for q3 in q_range:
            if mapper.is_unsafe_paper_method(np.degrees([q1, q2, q3])): 
                unsafe.append([q1, q2, q3])
            else: safe.append([q1, q2, q3])

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
s = np.array(safe)
ax1.scatter(s[:,0], s[:,1], s[:,2], s=0.5, c='blue', alpha=0.2)
ax1.set_title("Go Zone (Safe)")
ax2 = fig.add_subplot(122, projection='3d')
u = np.array(unsafe)
ax2.scatter(u[:,0], u[:,1], u[:,2], s=0.5, c='red', alpha=0.2)
ax2.set_title("No-Go Zone (Forbidden)")
plt.show()