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

# ── ZUPT (Zero Velocity Update) thresholds ────────────────────────────────────
# Loosened from original (0.3 / 0.08) to reliably detect stillness on real sensors
ZUPT_ACC_TOL   = 0.8    # m/s² tolerance around 1G
ZUPT_GYRO_TOL  = 0.15   # rad/s
ZUPT_VEL_NOISE = 1e-4   # covariance pinned to this when ZUPT fires

# ── ICP anchor gating ─────────────────────────────────────────────────────────
# ref_points only updated when ICP fitness is below this threshold,
# preventing a bad match from corrupting the scan anchor
ICP_MAX_FITNESS_FOR_REF = 0.15

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS & MATH
# ═══════════════════════════════════════════════════════════════════════════════

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def skew_symmetric(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])

def scan_to_xy(angles_deg, ranges_m):
    rad = np.deg2rad(angles_deg)
    return np.column_stack([ranges_m * np.cos(rad), ranges_m * np.sin(rad)])

def filter_scan(angles, ranges):
    mask = (ranges > MIN_RANGE_M) & (ranges < MAX_RANGE_M)
    return angles[mask], ranges[mask]

def icp_2d(src, ref, max_iter=ICP_ITER, tol=ICP_THRESH):
    """
    Point-to-point ICP in 2-D.
    Returns the 3×3 homogeneous transform T and a fitness score
    (mean nearest-neighbour distance; lower = better match).
    """
    if len(src) < 5 or len(ref) < 5:
        return np.eye(3), float("inf")
    T = np.eye(3)
    pts = src.copy()
    dists2 = None
    for _ in range(max_iter):
        diffs  = ref[:, None, :] - pts[None, :, :]
        dists2 = (diffs ** 2).sum(axis=-1)
        idx    = dists2.argmin(axis=0)
        matched_ref = ref[idx]

        mu_s, mu_r = pts.mean(axis=0), matched_ref.mean(axis=0)
        S = (pts - mu_s).T @ (matched_ref - mu_r)
        U, _, Vt = np.linalg.svd(S)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[-1] *= -1
            R_step = Vt.T @ U.T
        t_step = mu_r - R_step @ mu_s

        dT = np.eye(3)
        dT[:2, :2] = R_step
        dT[:2, 2]  = t_step
        T   = dT @ T
        pts = pts @ R_step.T + t_step

        if np.linalg.norm(t_step) < tol:
            break

    fitness = np.sqrt(dists2.min(axis=0)).mean() if dists2 is not None else float("inf")
    return T, fitness

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class EKF2D:
    """
    15-state EKF: [px, py, pz, vx, vy, vz, roll, pitch, yaw, bax, bay, baz, bgx, bgy, bgz]

    Key changes vs. original:
      • Process noise Q reduced by ~10× — EKF trusts its own kinematic model more,
        so velocity/bias states don't grow freely when the robot is still.
      • Velocity covariance is explicitly collapsed by ZUPT (done in main()).
    """
    def __init__(self):
        self.x = np.zeros(15)
        self.P = np.eye(15) * 0.5
        self.P[6:9, 6:9] *= 0.2

        # ── Tightened process noise ──────────────────────────────────────────
        self.Q = np.eye(15) * 0.001          # was 0.01  — position / attitude
        self.Q[3:6,  3:6]  *= 0.05           # velocity:  5e-5  (was 1e-3)
        self.Q[9:15, 9:15] *= 0.0001         # bias drift: 1e-7 (was 1e-5)
        # ─────────────────────────────────────────────────────────────────────

        self.R = np.diag([0.05, 0.05, 0.02])

    # ── internal helpers ──────────────────────────────────────────────────────

    def get_rotation_matrix(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c[2], -s[2], 0], [s[2],  c[2], 0], [0, 0, 1]])
        Ry = np.array([[c[1],  0, s[1]], [0,      1,   0], [-s[1], 0, c[1]]])
        Rx = np.array([[1,     0,    0], [0,  c[0], -s[0]], [0, s[0], c[0]]])
        return Rz @ Ry @ Rx

    def calculate_jacobian(self, x, a, dt):
        theta = x[6:9]
        R = self.get_rotation_matrix(theta)
        F = np.eye(15)
        F[0:3,  3:6]  = np.eye(3) * dt
        F[3:6,  6:9]  = -R @ skew_symmetric(a) * dt
        F[3:6,  9:12] = -R * dt
        F[3:6, 12:15] =  R * dt
        return F

    # ── public interface ──────────────────────────────────────────────────────

    def predict(self, ax, ay, az, gx, gy, gz, dt):
        p     = self.x[0:3]
        v     = self.x[3:6]
        theta = self.x[6:9]
        ba    = self.x[9:12]
        bg    = self.x[12:15]

        unbiased_a = np.array([ax, ay, az]) - ba
        unbiased_g = np.array([gx, gy, gz]) - bg

        R            = self.get_rotation_matrix(theta)
        accel_global = R.dot(unbiased_a) + np.array([0, 0, -9.81])

        new_p     = p + v * dt + 0.5 * accel_global * (dt ** 2)
        new_v     = v + accel_global * dt
        new_theta = wrap(theta + unbiased_g * dt)

        self.x = np.concatenate([new_p, new_v, new_theta, ba, bg])

        F      = self.calculate_jacobian(self.x, unbiased_a, dt)
        self.P = F @ self.P @ F.T + self.Q

    def correct(self, z):
        """
        Measurement update: z = [x, y, yaw] from ICP-corrected scan matching.
        """
        H = np.zeros((3, 15))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 8] = 1.0

        innov    = z - H @ self.x
        innov[2] = wrap(innov[2])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x    += K @ innov
        self.x[8]  = wrap(self.x[8])
        self.P     = (np.eye(15) - K @ H) @ self.P

    @property
    def pose(self):
        """2-D pose for mapping: [x, y, yaw]."""
        return np.array([self.x[0], self.x[1], self.x[8]])


class OccupancyGrid:
    """
    Log-odds occupancy grid.

    Key change vs. original:
      • Map decay (forgetting factor) is now conditional on `is_moving`.
        When the robot is stationary the grid is NOT decayed, so clean
        observations accumulated at rest are not eroded.
    """
    def __init__(self):
        self.grid = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.float32)

    def update(self, robot_xy, pts_world, is_moving: bool):
        # ── Conditional decay ────────────────────────────────────────────────
        # Only decay while the robot is moving; stationary observations persist.
        if is_moving:
            self.grid -= 0.02
        # ─────────────────────────────────────────────────────────────────────

        for pt in pts_world:
            cx = int(pt[0] / MAP_CELL_M + MAP_CELLS / 2)
            cy = int(pt[1] / MAP_CELL_M + MAP_CELLS / 2)
            if 0 <= cx < MAP_CELLS and 0 <= cy < MAP_CELLS:
                self.grid[cy, cx] += 0.4

        self.grid = np.clip(self.grid, -5.0, 5.0)

    def prob(self):
        return 1.0 / (1.0 + np.exp(-self.grid))


class IMUReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.acc  = (0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        if not HAS_IMU:
            return
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            imu = ICM20948(i2c, address=IMU_ADDR)
            while not self._stop.is_set():
                a_x, a_y, a_z = imu.acceleration
                g_x, g_y, g_z = imu.gyro
                with self._lock:
                    self.acc  = (a_x, a_y, a_z)
                    self.gyro = (g_x, g_y, g_z)
                time.sleep(DT_IMU)
        except Exception as e:
            print(f"[IMU] Error: {e}")

    def read(self):
        with self._lock:
            return self.acc, self.gyro


class LidarReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._angles      = np.array([])
        self._ranges      = np.array([])
        self._lock        = threading.Lock()
        self._stop        = threading.Event()
        self._new         = threading.Event()
        self._scan_count  = 0

    def run(self):
        if not HAS_RPLIDAR:
            return
        while not self._stop.is_set():
            lidar = None
            try:
                lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=3)
                lidar.get_health = lambda: ('Good', 0)
                lidar.connect()
                lidar.start_motor()
                time.sleep(0.5)
                lidar.clean_input()
                for scan in lidar.iter_scans(max_buf_meas=1000):
                    if self._stop.is_set():
                        break
                    a = np.array([m[1] for m in scan if m[0] > 0])
                    r = np.array([m[2] / 1000.0 for m in scan if m[0] > 0])
                    a, r = filter_scan(a, r)
                    if len(a) > 0:
                        with self._lock:
                            self._angles      = a.copy()
                            self._ranges      = r.copy()
                            self._scan_count += 1
                        self._new.set()
            except Exception as e:
                print(f"[Lidar] Error: {e}, scans so far: {self._scan_count}")
                if lidar:
                    try:
                        lidar.stop()
                        lidar.stop_motor()
                        lidar.disconnect()
                    except Exception:
                        pass
                time.sleep(1)

    def get_scan(self):
        self._new.clear()
        with self._lock:
            return self._angles.copy(), self._ranges.copy()

# ═══════════════════════════════════════════════════════════════════════════════
#  MAP SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def save_map_to_disk(grid, filename, fig, ax, traj_x, traj_y):
    ax.clear()
    prob_map       = grid.prob()
    map_size_meters = MAP_CELLS * MAP_CELL_M
    extent         = [-map_size_meters / 2, map_size_meters / 2,
                      -map_size_meters / 2, map_size_meters / 2]

    ax.imshow(prob_map, cmap="bone_r", extent=extent, origin='lower')

    if len(traj_x) > 0:
        ax.plot(traj_x,    traj_y,    'r-', linewidth=2,   label='Robot Path')
        ax.plot(traj_x[-1], traj_y[-1], 'go', markersize=8, label='Current Pos')
        ax.plot(traj_x[0],  traj_y[0],  'bo', markersize=6, label='Start Pos')

    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Distance (Meters)")
    ax.set_ylabel("Distance (Meters)")
    ax.set_title(f"SLAM Output — {time.strftime('%H:%M:%S')}")
    ax.legend(loc='upper right')

    fig.savefig(filename, dpi=120)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    imu   = IMUReader()
    lidar = LidarReader()
    imu.start()
    lidar.start()

    ekf  = EKF2D()
    grid = OccupancyGrid()

    fig, ax = plt.subplots(figsize=(10, 10))

    t_prev      = time.monotonic()
    t_last_save = time.monotonic()
    ref_points  = None

    trajectory_x = []
    trajectory_y = []

    print("[SYSTEM] Headless mode active. Saving map every 5 s to 'map_output.png'.")

    try:
        while True:
            t_now  = time.monotonic()
            dt     = t_now - t_prev
            t_prev = t_now

            acc_im, gyro_im = imu.read()

            # ── Motion detection ─────────────────────────────────────────────
            acc_mag  = math.sqrt(acc_im[0] ** 2 + acc_im[1] ** 2 + acc_im[2] ** 2)
            gyro_mag = math.sqrt(gyro_im[0] ** 2 + gyro_im[1] ** 2 + gyro_im[2] ** 2)

            is_stationary = (abs(acc_mag - 9.81) < ZUPT_ACC_TOL and
                             gyro_mag < ZUPT_GYRO_TOL)
            is_moving     = not is_stationary
            # ─────────────────────────────────────────────────────────────────

            # ── ZUPT (Zero Velocity Update) ───────────────────────────────────
            # When stationary: zero velocity state AND collapse velocity
            # covariance so the next predict() cannot re-inflate it from noise.
            if is_stationary:
                ekf.x[3:6]       = 0.0
                ekf.P[3:6, 3:6]  = np.eye(3) * ZUPT_VEL_NOISE
            # ─────────────────────────────────────────────────────────────────

            # EKF predict — full 6-DOF IMU propagation
            ekf.predict(
                acc_im[0],  acc_im[1],  acc_im[2],
                gyro_im[0], gyro_im[1], gyro_im[2],
                dt
            )

            # ── Lidar / ICP update ────────────────────────────────────────────
            if lidar._new.is_set():
                angles, ranges = lidar.get_scan()
                if len(angles) > 0:
                    scan_xy = scan_to_xy(angles, ranges)
                    pose    = ekf.pose

                    if ref_points is not None:
                        # Transform current scan to world frame using EKF pose
                        R_mat      = rot2(pose[2])
                        scan_world = (R_mat @ scan_xy.T).T + pose[:2]

                        T_icp, fitness = icp_2d(scan_world, ref_points)

                        if fitness < 0.3:
                            theta_icp = math.atan2(T_icp[1, 0], T_icp[0, 0])
                            z = np.array([
                                T_icp[0, 2] + pose[0],
                                T_icp[1, 2] + pose[1],
                                wrap(theta_icp + pose[2])
                            ])
                            ekf.correct(z)

                        trajectory_x.append(ekf.pose[0])
                        trajectory_y.append(ekf.pose[1])

                    else:
                        # First scan — initialise
                        print(f"[SLAM] Initialising with {len(scan_xy)} scan points.")
                        fitness = float("inf")   # sentinel: skip anchor update gate
                        trajectory_x.append(ekf.pose[0])
                        trajectory_y.append(ekf.pose[1])

                    # ── Anchor update — gated on ICP quality ─────────────────
                    # Only replace ref_points when the match was clean enough.
                    # A bad match (high fitness) means the scan drifted; keeping
                    # the old anchor prevents that drift from propagating.
                    new_world_pts = (rot2(ekf.pose[2]) @ scan_xy.T).T + ekf.pose[:2]

                    if ref_points is None or fitness < ICP_MAX_FITNESS_FOR_REF:
                        ref_points = new_world_pts
                    # else: keep previous ref_points
                    # ─────────────────────────────────────────────────────────

                    # Update occupancy grid (decay only if moving)
                    grid.update(ekf.pose[:2], new_world_pts, is_moving=is_moving)
            # ─────────────────────────────────────────────────────────────────

            # Periodic map save
            if t_now - t_last_save > 5.0:
                save_map_to_disk(grid, "map_output.png", fig, ax,
                                 trajectory_x, trajectory_y)
                t_last_save = t_now
                print(f"[{time.strftime('%H:%M:%S')}] Map updated on disk.")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Saving final map and exiting...")
        save_map_to_disk(grid, "final_map.png", fig, ax,
                         trajectory_x, trajectory_y)
        lidar._stop.set()
        imu._stop.set()


if __name__ == "__main__":
    main()