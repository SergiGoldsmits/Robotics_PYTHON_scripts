import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Planar4LinkFR3:
    def __init__(self, safety_buffer=0.0):
        # 4 link lengths: Base, Shoulder, Elbow, Wrist
        self.L = [0.333, 0.316, 0.384, 0.107]
        self.buffer = safety_buffer
        
        # LINK RADII: Base (0.10m), Links 2 & 3 (0.05m), Wrist (0.0m)
        self.link_radii = [0.10, 0.05, 0.05, 0.0]
        
        self.obs_center = np.array([0.20, 0.45])
        self.obs_radius = 0.10
        self.config_dir = "thesis_data/"
        os.makedirs(self.config_dir, exist_ok=True)

    def get_coords(self, q):
        # q = [q1, q3, q5]
        th1 = q[0]
        th2 = q[0] - q[1]
        th3 = q[0] - q[1] + q[2]
        
        x = [0, 0, self.L[1]*np.sin(th1), 
             self.L[1]*np.sin(th1)+self.L[2]*np.sin(th2), 
             self.L[1]*np.sin(th1)+self.L[2]*np.sin(th2)+self.L[3]*np.sin(th3)]
        z = [0, self.L[0], self.L[0]+self.L[1]*np.cos(th1), 
             self.L[0]+self.L[1]*np.cos(th1)+self.L[2]*np.cos(th2), 
             self.L[0]+self.L[1]*np.cos(th1)+self.L[2]*np.cos(th2)+self.L[3]*np.cos(th3)]
        return np.vstack((x, z)).T
    
    def segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def is_safe(self, q):
        coords = self.get_coords(q)
        for i in range(4):
            p1, p2 = coords[i], coords[i+1]
            v, w = p2 - p1, self.obs_center - p1
            t = np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-6), 0, 1)
            
            # Distance check with link radii + obstacle radius
            if np.linalg.norm(w - t * v) < (self.obs_radius + self.link_radii[i]): 
                return False
            
        for i in range(4):
            for j in range(i + 2, 4):
                if self.segments_intersect(coords[i], coords[i+1], coords[j], coords[j+1]):
                    return False
        return True

    def run_analysis(self, res=25):
        q_range = np.linspace(-np.pi, np.pi, res)
        safe, unsafe = [], []
        for q1 in q_range:
            for q3 in q_range:
                for q5 in q_range:
                    config = [q1, q3, q5]
                    if self.is_safe(config): safe.append(config)
                    else: unsafe.append(config)
        pd.DataFrame(safe, columns=['q1','q3','q5']).to_csv(self.config_dir+'safe.csv', index=False)
        pd.DataFrame(unsafe, columns=['q1','q3','q5']).to_csv(self.config_dir+'unsafe.csv', index=False)

    def plot_all(self, samples=20, cloud_samples=3000):
        df_u = pd.read_csv(self.config_dir + 'unsafe.csv')
        df_s = pd.read_csv(self.config_dir + 'safe.csv')
        
        fig = plt.figure(figsize=(20, 6))
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(df_u['q1'], df_u['q3'], df_u['q5'], c=df_u['q5'], cmap='Reds', s=5, alpha=0.5)
        ax1.set_xlabel('$q_1$'); ax1.set_ylabel('$q_3$'); ax1.set_zlabel('$q_5$')
        ax1.set_title("Unsafe Joint Configurations")
        
        ax2 = fig.add_subplot(132)
        ax2.add_patch(plt.Circle(self.obs_center, self.obs_radius, color='red', alpha=0.3))
        bias = np.array([-0.785, -2.356, 1.571])
        for _ in range(samples):
            q = np.random.normal(loc=bias, scale=0.2, size=3)
            pts = self.get_coords(q)
            ax2.plot(pts[:,0], pts[:,1], 'k-', alpha=0.2)
        ax2.set_aspect('equal')
        ax2.set_title("Cartesian Workspace (Ready Pose Bias)")
        
        ax3 = fig.add_subplot(133)
        ax3.add_patch(plt.Circle(self.obs_center, self.obs_radius, color='red', alpha=0.3))
        
        s_sample = df_s.sample(n=min(len(df_s), cloud_samples))
        for _, row in s_sample.iterrows():
            pts = self.get_coords([row['q1'], row['q3'], row['q5']])
            ax3.scatter(pts[-1, 0], pts[-1, 1], color='#003399', s=6, alpha=0.1)
            
        u_sample = df_u.sample(n=min(len(df_u), cloud_samples))
        for _, row in u_sample.iterrows():
            pts = self.get_coords([row['q1'], row['q3'], row['q5']])
            ax3.scatter(pts[-1, 0], pts[-1, 1], color='#CC0000', s=6, alpha=0.2)
            
        ax3.set_aspect('equal')
        ax3.set_title("Total Reachable Workspace (Density Map)")
        plt.show()

if __name__ == "__main__":
    mapper = Planar4LinkFR3()
    mapper.run_analysis()
    mapper.plot_all()