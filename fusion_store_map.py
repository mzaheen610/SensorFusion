"""
Real-Time 2D LiDAR SLAM on Raspberry Pi (Headless / Save-to-Disk Version)
Fixes: 
1. Use 'Agg' backend for headless environments (SSH/No Monitor).
2. RPLidarException auto-recovery.
3. get_health() monkey-patch for "too many values" bug.
4. Optimized periodic map saving to file.
"""

import math
import time
import threading
import numpy as np
import os

# ── Dependencies ─────────────────────────────────────────────────────────────
try:
    from rplidar import RPLidar
    HAS_RPLIDAR = True
except ImportError:
    HAS_RPLIDAR = False
    print("[WARN] rplidar not installed. Lidar disabled.")

try:
    import board, busio
    from adafruit_icm20x import ICM20948
    HAS_IMU = True
except ImportError:
    HAS_IMU = False
    print("[WARN] adafruit-circuitpython-icm20x not installed. IMU disabled.")

import matplotlib
matplotlib.use("Agg")  # CRITICAL: Use non-interactive backend for headless save
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LIDAR_PORT    = "/dev/ttyUSB0"   
LIDAR_BAUD    = 256000
IMU_ADDR      = 0x69             

IMU_HZ        = 100              
LIDAR_HZ      = 10               
DT_IMU        = 1.0 / IMU_HZ

MAX_RANGE_M   = 8.0              
MIN_RANGE_M   = 0.15             
ICP_ITER      = 15               
ICP_THRESH    = 0.05             
MAP_CELL_M    = 0.05             
MAP_CELLS     = 400              

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS & MATH
# ═══════════════════════════════════════════════════════════════════════════════

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def rot2(theta):
    #2D rotation matrix for angle theta
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def scan_to_xy(angles_deg, ranges_m):
    rad = np.deg2rad(angles_deg)
    return np.column_stack([ranges_m * np.cos(rad), ranges_m * np.sin(rad)])

def filter_scan(angles, ranges):
    mask = (ranges > MIN_RANGE_M) & (ranges < MAX_RANGE_M)
    return angles[mask], ranges[mask]

def icp_2d(src, ref, max_iter=ICP_ITER, tol=ICP_THRESH):
    #tiny ICP implementation for 2D point clouds. Returns transformation matrix and fitness score.
    if len(src) < 5 or len(ref) < 5: return np.eye(3), float("inf")
    T = np.eye(3)
    pts = src.copy()
    for _ in range(max_iter):
        diffs = ref[:, None, :] - pts[None, :, :]
        dists2 = (diffs ** 2).sum(axis=-1)
        idx = dists2.argmin(axis=0)
        matched_ref = ref[idx]
        mu_s, mu_r = pts.mean(axis=0), matched_ref.mean(axis=0)
        S = (pts - mu_s).T @ (matched_ref - mu_r)
        U, _, Vt = np.linalg.svd(S)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0: Vt[-1] *= -1; R_step = Vt.T @ U.T
        t_step = mu_r - R_step @ mu_s
        dT = np.eye(3); dT[:2, :2] = R_step; dT[:2, 2] = t_step
        T = dT @ T
        pts = pts @ R_step.T + t_step
        if np.linalg.norm(t_step) < tol: break
    fitness = np.sqrt(dists2.min(axis=0)).mean()
    return T, fitness

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class EKF2D:
    def __init__(self):
        self.x = np.zeros(5) 
        self.P = np.diag([0.5, 0.5, 0.1, 1.0, 0.05])
        self.Q = np.diag([0.002, 0.002, 0.001, 0.10, 0.05])
        self.R = np.diag([0.05, 0.05, 0.02])

    def predict(self, ax, gz, dt):
        x, y, th, v, om = self.x #state space: x, y, theta, velocity, angular velocity
        v_new = max(0.0, v + ax * dt) #the new velocity using the acceleration from the IMU
        th_new = wrap(th + gz * dt) #the new angle using the gyro z from the IMU
        if abs(gz) > 1e-4: #if the angular velocity is significant, we use the bicycle model for prediction
            x_new = x + (v / gz) * (math.sin(th + gz * dt) - math.sin(th))
            y_new = y + (v / gz) * (-math.cos(th + gz * dt) + math.cos(th))
        else: #
            x_new = x + v * math.cos(th) * dt #update position using the velocity component in the x direction
            y_new = y + v * math.sin(th) * dt #update position using the velocity component in the y direction
        self.x = np.array([x_new, y_new, th_new, v_new, gz])
        self.P = self.P + self.Q 

    def correct(self, z):
        H = np.zeros((3, 5)); H[0,0] = H[1,1] = H[2,2] = 1.0 #measurement matrix that maps the state to the measurement space (x, y, theta)
        innov = z - H @ self.x; innov[2] = wrap(innov[2])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innov; self.x[2] = wrap(self.x[2])
        self.P = (np.eye(5) - K @ H) @ self.P

    @property
    def pose(self): return self.x[:3].copy()

class OccupancyGrid:
    def __init__(self):
        self.grid = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.float32)
    def update(self, robot_xy, pts_world):
        for pt in pts_world:
            cx = int(pt[0] / MAP_CELL_M + MAP_CELLS/2)
            cy = int(pt[1] / MAP_CELL_M + MAP_CELLS/2)
            if 0 <= cx < MAP_CELLS and 0 <= cy < MAP_CELLS:
                self.grid[cy, cx] = np.clip(self.grid[cy, cx] + 0.8, -5, 5)
    def prob(self): return 1.0 / (1.0 + np.exp(-self.grid))

class IMUReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.ax = 0.0; self.gz = 0.0
        self._lock = threading.Lock(); self._stop = threading.Event()
    def run(self):
        if not HAS_IMU: return
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            imu = ICM20948(i2c, address=IMU_ADDR)
            while not self._stop.is_set():
                acc_x, _, _ = imu.acceleration
                _, _, gyro_z = imu.gyro
                with self._lock: self.ax, self.gz = acc_x, gyro_z
                time.sleep(DT_IMU)
        except Exception as e:
            print(f"[IMU] Error: {e}")
    def read(self):
        with self._lock: return self.ax, self.gz

class LidarReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._angles = np.array([])
        self._ranges = np.array([])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._new = threading.Event()
        self._scan_count = 0

    def run(self):
        if not HAS_RPLIDAR: return
        while not self._stop.is_set():
            lidar = None
            try:
                lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=3)
                lidar.get_health = lambda: ('Good', 0) # Monkey-patch fix
                lidar.connect()
                lidar.start_motor()
                time.sleep(0.5)
                lidar.clean_input()
                for scan in lidar.iter_scans(max_buf_meas=5000):
                    if self._stop.is_set(): break
                    a = np.array([m[1] for m in scan if m[0] > 0])  # Quality check
                    r = np.array([m[2] / 1000.0 for m in scan if m[0] > 0])
                    a, r = filter_scan(a, r)
                    if len(a) > 0:
                        with self._lock: 
                            self._angles = a.copy()
                            self._ranges = r.copy()
                            self._scan_count += 1
                        self._new.set()
            except Exception as e:
                print(f"[Lidar] Error: {e}, attempts: {self._scan_count}")
                if lidar: 
                    try:
                        lidar.stop()
                        lidar.stop_motor()
                        lidar.disconnect()
                    except:
                        pass
                time.sleep(1)

    def get_scan(self):
        self._new.clear()
        with self._lock: 
            return self._angles.copy(), self._ranges.copy()

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def save_map_to_disk(grid, filename, fig, ax):
    """Updates the figure and saves to file without closing/reopening."""
    ax.clear()
    prob_map = grid.prob()
    ax.imshow(prob_map, cmap="bone", extent=[-10, 10, -10, 10], origin='lower')
    ax.set_title(f"SLAM Map - {time.strftime('%H:%M:%S')} | Grid Min: {grid.grid.min():.2f}, Max: {grid.grid.max():.2f}")
    fig.savefig(filename, dpi=80)
    # Debug: Print grid statistics
    if grid.grid.max() > 0:
        print(f"[MAP] Grid stats - Min: {grid.grid.min():.2f}, Max: {grid.grid.max():.2f}, Points added: YES")

def main():
    imu = IMUReader()
    lidar = LidarReader()
    imu.start()
    lidar.start()
    
    ekf = EKF2D()
    grid = OccupancyGrid()
    
    # Initialize background figure for saving
    fig, ax = plt.subplots(figsize=(10, 10))
    
    t_prev = time.monotonic()
    t_last_save = time.monotonic()
    ref_points = None
    scan_data_received = False

    print(f"[SYSTEM] Headless mode active. Saving map every 5s to 'map_output.png'.")
    print(f"[SYSTEM] Waiting for LiDAR scans...")
    
    try:
        while True:
            t_now = time.monotonic()
            dt = t_now - t_prev
            t_prev = t_now
            
            # 1. Prediction
            ax_im, gz_im = imu.read()
            ekf.predict(ax_im, gz_im, dt)

            # 2. Correction
            if lidar._new.is_set():
                angles, ranges = lidar.get_scan()
                if len(angles) > 0:
                    scan_data_received = True
                    scan_xy = scan_to_xy(angles, ranges)
                    pose = ekf.pose
                    if ref_points is not None:
                        R = rot2(pose[2])
                        scan_world = (R @ scan_xy.T).T + pose[:2]
                        T_icp, fitness = icp_2d(scan_world, ref_points)
                        if fitness < 0.3:
                            theta_icp = math.atan2(T_icp[1,0], T_icp[0,0])
                            z = np.array([T_icp[0,2]+pose[0], T_icp[1,2]+pose[1], wrap(theta_icp+pose[2])])
                            ekf.correct(z)
                    else:
                        print(f"[SLAM] Initializing with {len(scan_xy)} scan points from {len(angles)} angles")
                    R = rot2(ekf.pose[2])
                    ref_points = (R @ scan_xy.T).T + ekf.pose[:2]
                    grid.update(ekf.pose[:2], ref_points)

            # 3. Periodic Disk Save (Every 5 Seconds)
            if t_now - t_last_save > 1.0:
                save_map_to_disk(grid, "map_output.png", fig, ax)
                t_last_save = t_now
                if grid.grid.max() > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Map updated on disk.")
                elif not scan_data_received:
                    print(f"[{time.strftime('%H:%M:%S')}] Waiting for LiDAR data...")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Saving final map and exiting...")
        save_map_to_disk(grid, "final_map.png", fig, ax)
        lidar._stop.set()
        lidar.stop_motor()
        imu._stop.set()
        

if __name__ == "__main__":
    main()
