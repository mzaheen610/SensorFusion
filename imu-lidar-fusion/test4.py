import argparse
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from rplidar import RPLidar
    HAS_RPLIDAR = True
except ImportError:
    HAS_RPLIDAR = False
    print("[WARN] rplidar not installed. LiDAR disabled.")

try:
    import board
    import busio
    from adafruit_icm20x import ICM20948
    HAS_IMU = True
except ImportError:
    HAS_IMU = False
    print("[WARN] adafruit-circuitpython-icm20x not installed. IMU disabled.")


LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 256000
IMU_ADDR = 0x69

IMU_HZ = 100
DT_IMU = 1.0 / IMU_HZ

GRAVITY = 9.81
MAX_RANGE_M = 8.0
MIN_RANGE_M = 0.15
SCAN_VOXEL_M = 0.08
SUBMAP_VOXEL_M = 0.10
MAX_SCAN_POINTS = 220

MAP_CELL_M = 0.05
MAP_CELLS = 500
MAP_FREE_LOGODDS = -0.08
MAP_OCC_LOGODDS = 0.28
MAP_MIN_LOGODDS = -5.0
MAP_MAX_LOGODDS = 5.0

CALIBRATION_SECONDS = 2.5
STATIONARY_ACC_TOL = 0.18
STATIONARY_GYRO_TOL = 0.04
STATIONARY_BIAS_GAIN = 0.02

ICP_MAX_ITER = 20
ICP_TOL_M = 1e-3
ICP_MAX_PAIR_DIST = 0.45
ICP_MIN_PAIRS = 35
ICP_MAX_FITNESS = 0.18
ICP_MAX_TRANSLATION_STEP = 0.60
ICP_MAX_YAW_STEP = math.radians(12.0)

SUBMAP_MAX_SCANS = 20


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot2(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotation_matrix_rpy(roll, pitch, yaw):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def filter_scan(angles_deg, ranges_m):
    mask = np.isfinite(ranges_m)
    mask &= ranges_m > MIN_RANGE_M
    mask &= ranges_m < MAX_RANGE_M
    return angles_deg[mask], ranges_m[mask]


def scan_to_xy(angles_deg, ranges_m):
    angles_rad = np.deg2rad(angles_deg)
    return np.column_stack((ranges_m * np.cos(angles_rad), ranges_m * np.sin(angles_rad)))


def voxel_downsample(points, voxel_size, max_points=None):
    if len(points) == 0:
        return points
    voxels = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxels, axis=0, return_index=True)
    unique_idx.sort()
    sampled = points[unique_idx]
    if max_points is not None and len(sampled) > max_points:
        step = max(1, len(sampled) // max_points)
        sampled = sampled[::step][:max_points]
    return sampled


def transform_points(points, pose):
    if len(points) == 0:
        return points
    return (rot2(pose[2]) @ points.T).T + pose[:2]


def apply_transform_to_pose(pose, transform):
    rot = math.atan2(transform[1, 0], transform[0, 0])
    xy = transform[:2, :2] @ pose[:2] + transform[:2, 2]
    return np.array([xy[0], xy[1], wrap(pose[2] + rot)], dtype=float)


def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x = x0
    y = y0
    cells = []
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


def accel_to_roll_pitch(acc_body):
    ax, ay, az = acc_body
    roll = math.atan2(ay, az if abs(az) > 1e-6 else 1e-6)
    pitch = math.atan2(-ax, max(1e-6, math.sqrt(ay * ay + az * az)))
    return roll, pitch


def level_acceleration(acc_body, roll, pitch):
    r_level_from_body = rotation_matrix_rpy(roll, pitch, 0.0)
    acc_level = r_level_from_body @ acc_body - np.array([0.0, 0.0, GRAVITY])
    return acc_level[:2]


def compute_expected_gravity_body(roll, pitch):
    r_world_from_body = rotation_matrix_rpy(roll, pitch, 0.0)
    gravity_world = np.array([0.0, 0.0, GRAVITY])
    return r_world_from_body.T @ gravity_world


@dataclass
class CalibrationResult:
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    initial_roll: float
    initial_pitch: float


class TiltEstimator:
    def __init__(self, roll=0.0, pitch=0.0):
        self.roll = roll
        self.pitch = pitch

    def update(self, acc_body, gyro_body, dt, stationary=False):
        self.roll += gyro_body[0] * dt
        self.pitch += gyro_body[1] * dt

        roll_acc, pitch_acc = accel_to_roll_pitch(acc_body)
        alpha = 0.10 if stationary else 0.03
        self.roll = (1.0 - alpha) * self.roll + alpha * roll_acc
        self.pitch = (1.0 - alpha) * self.pitch + alpha * pitch_acc
        return self.roll, self.pitch


class PlanarEKF:
    def __init__(self):
        self.x = np.zeros(8, dtype=float)
        self.P = np.diag([0.15, 0.15, 0.20, 0.20, 0.10, 0.08, 0.08, 0.04])
        self.q_acc = 0.45
        self.q_yaw = 0.10
        self.q_bias_acc = 0.0008
        self.q_bias_gyro = 0.00015

    @property
    def pose(self):
        return self.x[[0, 1, 4]].copy()

    @property
    def speed(self):
        return float(np.linalg.norm(self.x[2:4]))

    def predict(self, acc_body_xy, gyro_z, dt):
        if dt <= 0.0:
            return

        px, py, vx, vy, yaw, bax, bay, bgz = self.x
        corrected_acc_body = np.array([acc_body_xy[0] - bax, acc_body_xy[1] - bay])
        rotation = rot2(yaw)
        acc_world = rotation @ corrected_acc_body
        yaw_rate = gyro_z - bgz

        new_px = px + vx * dt + 0.5 * acc_world[0] * dt * dt
        new_py = py + vy * dt + 0.5 * acc_world[1] * dt * dt
        new_vx = vx + acc_world[0] * dt
        new_vy = vy + acc_world[1] * dt
        new_yaw = wrap(yaw + yaw_rate * dt)
        self.x = np.array([new_px, new_py, new_vx, new_vy, new_yaw, bax, bay, bgz], dtype=float)

        d_r_d_yaw = np.array([[-math.sin(yaw), -math.cos(yaw)], [math.cos(yaw), -math.sin(yaw)]], dtype=float)
        d_acc_d_yaw = d_r_d_yaw @ corrected_acc_body

        f = np.eye(8)
        f[0, 2] = dt
        f[1, 3] = dt
        f[0, 4] = 0.5 * d_acc_d_yaw[0] * dt * dt
        f[1, 4] = 0.5 * d_acc_d_yaw[1] * dt * dt
        f[2, 4] = d_acc_d_yaw[0] * dt
        f[3, 4] = d_acc_d_yaw[1] * dt
        f[0:2, 5:7] = -0.5 * rotation * dt * dt
        f[2:4, 5:7] = -rotation * dt
        f[4, 7] = -dt

        q = np.zeros((8, 8), dtype=float)
        pos_q = 0.25 * self.q_acc * dt ** 4
        vel_q = self.q_acc * dt ** 2
        q[0:2, 0:2] = np.eye(2) * pos_q
        q[2:4, 2:4] = np.eye(2) * vel_q
        q[4, 4] = self.q_yaw * dt * dt
        q[5:7, 5:7] = np.eye(2) * self.q_bias_acc * dt
        q[7, 7] = self.q_bias_gyro * dt

        self.P = f @ self.P @ f.T + q
        self.P = 0.5 * (self.P + self.P.T)

    def correct_pose(self, pose_measurement, measurement_cov):
        h = np.zeros((3, 8), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        h[2, 4] = 1.0
        innovation = pose_measurement - h @ self.x
        innovation[2] = wrap(innovation[2])
        s = h @ self.P @ h.T + measurement_cov
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x += k @ innovation
        self.x[4] = wrap(self.x[4])
        self.P = (np.eye(8) - k @ h) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def zero_velocity_update(self, noise=2e-4):
        h = np.zeros((2, 8), dtype=float)
        h[0, 2] = 1.0
        h[1, 3] = 1.0
        z = np.zeros(2, dtype=float)
        innovation = z - h @ self.x
        r = np.eye(2) * noise
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x += k @ innovation
        self.P = (np.eye(8) - k @ h) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def adapt_biases_when_stationary(self, planar_acc, gyro_z):
        self.x[5:7] = (1.0 - STATIONARY_BIAS_GAIN) * self.x[5:7] + STATIONARY_BIAS_GAIN * planar_acc
        self.x[7] = (1.0 - STATIONARY_BIAS_GAIN) * self.x[7] + STATIONARY_BIAS_GAIN * gyro_z


class OccupancyGrid:
    def __init__(self):
        self.grid = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.float32)
        self.center = MAP_CELLS // 2

    def world_to_cell(self, point_xy):
        cx = int(round(point_xy[0] / MAP_CELL_M)) + self.center
        cy = int(round(point_xy[1] / MAP_CELL_M)) + self.center
        return cx, cy

    def update(self, robot_pose, scan_points_body):
        if len(scan_points_body) == 0:
            return
        robot_cell = self.world_to_cell(robot_pose[:2])
        scan_world = transform_points(scan_points_body, robot_pose)
        for endpoint in scan_world:
            end_cell = self.world_to_cell(endpoint)
            ray = bresenham_line(robot_cell[0], robot_cell[1], end_cell[0], end_cell[1])
            for free_cell in ray[:-1]:
                x, y = free_cell
                if 0 <= x < MAP_CELLS and 0 <= y < MAP_CELLS:
                    self.grid[y, x] += MAP_FREE_LOGODDS
            x, y = end_cell
            if 0 <= x < MAP_CELLS and 0 <= y < MAP_CELLS:
                self.grid[y, x] += MAP_OCC_LOGODDS
        np.clip(self.grid, MAP_MIN_LOGODDS, MAP_MAX_LOGODDS, out=self.grid)

    def probability(self):
        return 1.0 / (1.0 + np.exp(-self.grid))


class LocalSubmap:
    def __init__(self, max_scans=SUBMAP_MAX_SCANS):
        self.scans = deque(maxlen=max_scans)

    def add_scan(self, scan_world):
        if len(scan_world) == 0:
            return
        self.scans.append(scan_world.copy())

    def points(self):
        if not self.scans:
            return np.empty((0, 2), dtype=float)
        merged = np.vstack(self.scans)
        return voxel_downsample(merged, SUBMAP_VOXEL_M, max_points=1200)


def icp_2d(source_points, reference_points, max_iter=ICP_MAX_ITER, tol=ICP_TOL_M):
    if len(source_points) < ICP_MIN_PAIRS or len(reference_points) < ICP_MIN_PAIRS:
        return np.eye(3), {"fitness": float("inf"), "num_pairs": 0}

    transform = np.eye(3, dtype=float)
    aligned = source_points.copy()
    fitness = float("inf")

    for _ in range(max_iter):
        deltas = reference_points[:, None, :] - aligned[None, :, :]
        dist2 = np.sum(deltas * deltas, axis=2)
        nearest_idx = np.argmin(dist2, axis=0)
        nearest_dist = np.sqrt(dist2[nearest_idx, np.arange(dist2.shape[1])])
        valid = nearest_dist < ICP_MAX_PAIR_DIST
        if np.count_nonzero(valid) < ICP_MIN_PAIRS:
            return np.eye(3), {"fitness": float("inf"), "num_pairs": int(np.count_nonzero(valid))}

        src = aligned[valid]
        ref = reference_points[nearest_idx[valid]]
        fitness = float(np.mean(nearest_dist[valid]))

        src_mean = src.mean(axis=0)
        ref_mean = ref.mean(axis=0)
        cross = (src - src_mean).T @ (ref - ref_mean)
        u, _, vt = np.linalg.svd(cross)
        r_step = vt.T @ u.T
        if np.linalg.det(r_step) < 0:
            vt[-1, :] *= -1
            r_step = vt.T @ u.T
        t_step = ref_mean - r_step @ src_mean

        step = np.eye(3, dtype=float)
        step[:2, :2] = r_step
        step[:2, 2] = t_step
        transform = step @ transform
        aligned = (r_step @ aligned.T).T + t_step

        if np.linalg.norm(t_step) < tol:
            break

    return transform, {"fitness": fitness, "num_pairs": int(np.count_nonzero(valid))}


class IMUReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.acc = np.zeros(3, dtype=float)
        self.gyro = np.zeros(3, dtype=float)
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        if not HAS_IMU:
            return
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            imu = ICM20948(i2c, address=IMU_ADDR)
            while not self._stop.is_set():
                ax, ay, az = imu.acceleration
                gx, gy, gz = imu.gyro
                with self._lock:
                    self.acc = np.array([ax, ay, az], dtype=float)
                    self.gyro = np.array([gx, gy, gz], dtype=float)
                time.sleep(DT_IMU)
        except Exception as exc:
            print(f"[IMU] Error: {exc}")

    def read(self):
        with self._lock:
            return self.acc.copy(), self.gyro.copy()


class LidarReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.angles = np.array([], dtype=float)
        self.ranges = np.array([], dtype=float)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._new_scan = threading.Event()

    def run(self):
        if not HAS_RPLIDAR:
            return
        while not self._stop.is_set():
            lidar = None
            try:
                lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=3)
                lidar.get_health = lambda: ("Good", 0)
                lidar.connect()
                lidar.start_motor()
                time.sleep(0.5)
                lidar.clean_input()
                for scan in lidar.iter_scans(max_buf_meas=3000):
                    if self._stop.is_set():
                        break
                    angles = np.array([m[1] for m in scan if m[0] > 0], dtype=float)
                    ranges = np.array([m[2] / 1000.0 for m in scan if m[0] > 0], dtype=float)
                    angles, ranges = filter_scan(angles, ranges)
                    if len(angles) == 0:
                        continue
                    with self._lock:
                        self.angles = angles.copy()
                        self.ranges = ranges.copy()
                    self._new_scan.set()
            except Exception as exc:
                print(f"[LiDAR] Error: {exc}. Restarting device...")
                if lidar is not None:
                    try:
                        lidar.stop()
                        lidar.stop_motor()
                        lidar.disconnect()
                    except Exception:
                        pass
                time.sleep(1.0)

    def get_scan(self):
        self._new_scan.clear()
        with self._lock:
            return self.angles.copy(), self.ranges.copy()

    def shutdown(self):
        self._stop.set()


def calibrate_imu(imu_reader, seconds=CALIBRATION_SECONDS):
    if not HAS_IMU:
        return CalibrationResult(
            accel_bias=np.zeros(3, dtype=float),
            gyro_bias=np.zeros(3, dtype=float),
            initial_roll=0.0,
            initial_pitch=0.0,
        )

    print(f"[IMU] Calibrating for {seconds:.1f}s. Keep the setup static.")
    acc_samples = []
    gyro_samples = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        acc, gyro = imu_reader.read()
        if np.linalg.norm(acc) > 1e-6:
            acc_samples.append(acc)
            gyro_samples.append(gyro)
        time.sleep(DT_IMU)

    if not acc_samples:
        print("[IMU] Calibration failed. Falling back to zero biases.")
        return CalibrationResult(
            accel_bias=np.zeros(3, dtype=float),
            gyro_bias=np.zeros(3, dtype=float),
            initial_roll=0.0,
            initial_pitch=0.0,
        )

    acc_mean = np.mean(np.vstack(acc_samples), axis=0)
    gyro_mean = np.mean(np.vstack(gyro_samples), axis=0)
    roll0, pitch0 = accel_to_roll_pitch(acc_mean)
    gravity_body = compute_expected_gravity_body(roll0, pitch0)
    accel_bias = acc_mean - gravity_body
    print(f"[IMU] Gyro bias: {gyro_mean.round(4)} | Accel bias: {accel_bias.round(4)}")
    return CalibrationResult(
        accel_bias=accel_bias,
        gyro_bias=gyro_mean,
        initial_roll=roll0,
        initial_pitch=pitch0,
    )


class FusionDiagnostics:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.speed = []
        self.icp_fitness = []
        self.stationary = []

    def append(self, timestamp, pose, speed, stationary, fitness):
        self.t.append(timestamp)
        self.x.append(float(pose[0]))
        self.y.append(float(pose[1]))
        self.yaw.append(float(pose[2]))
        self.speed.append(float(speed))
        self.stationary.append(1.0 if stationary else 0.0)
        self.icp_fitness.append(np.nan if fitness is None else float(fitness))


def save_outputs(grid, diagnostics, output_prefix):
    map_path = f"{output_prefix}_map.png"
    diag_path = f"{output_prefix}_diagnostics.png"

    prob = grid.probability()
    map_extent = np.array([-MAP_CELLS / 2, MAP_CELLS / 2, -MAP_CELLS / 2, MAP_CELLS / 2]) * MAP_CELL_M

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_map = axes[0, 0]
    ax_traj = axes[0, 1]
    ax_speed = axes[1, 0]
    ax_icp = axes[1, 1]

    ax_map.imshow(prob, cmap="bone_r", extent=map_extent, origin="lower")
    ax_map.set_title("Occupancy Grid")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    if diagnostics.x:
        ax_traj.plot(diagnostics.x, diagnostics.y, color="red", linewidth=2.0, label="EKF fused path")
        ax_traj.scatter(diagnostics.x[0], diagnostics.y[0], color="blue", s=45, label="start")
        ax_traj.scatter(diagnostics.x[-1], diagnostics.y[-1], color="green", s=55, label="current")
    ax_traj.set_title("Fused Trajectory")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.axis("equal")
    ax_traj.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if diagnostics.x:
        ax_traj.legend(loc="best")

    if diagnostics.t:
        t0 = diagnostics.t[0]
        rel_t = np.array(diagnostics.t) - t0
        ax_speed.plot(rel_t, diagnostics.speed, color="tab:orange", linewidth=1.5, label="speed")
        ax_speed.plot(rel_t, np.array(diagnostics.stationary) * max(0.05, np.max(diagnostics.speed) + 0.02),
                      color="tab:green", linewidth=1.0, alpha=0.7, label="stationary flag")
        ax_speed.set_title("Motion Diagnostics")
        ax_speed.set_xlabel("time (s)")
        ax_speed.set_ylabel("speed (m/s)")
        ax_speed.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_speed.legend(loc="best")

        ax_icp.plot(rel_t, diagnostics.icp_fitness, color="tab:purple", linewidth=1.5)
        ax_icp.axhline(ICP_MAX_FITNESS, color="tab:red", linestyle="--", linewidth=1.0, label="ICP gate")
        ax_icp.set_title("ICP Fitness")
        ax_icp.set_xlabel("time (s)")
        ax_icp.set_ylabel("mean pair distance (m)")
        ax_icp.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_icp.legend(loc="best")

    fig.tight_layout()
    fig.savefig(map_path, dpi=140)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.imshow(prob, cmap="bone_r", extent=map_extent, origin="lower")
    if diagnostics.x:
        ax2.plot(diagnostics.x, diagnostics.y, color="red", linewidth=2.2, label="EKF fused path")
        ax2.scatter(diagnostics.x[0], diagnostics.y[0], color="blue", s=45, label="start")
        ax2.scatter(diagnostics.x[-1], diagnostics.y[-1], color="green", s=55, label="current")
        ax2.legend(loc="best")
    ax2.set_title("LiDAR + IMU Fusion Result")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.axis("equal")
    fig2.tight_layout()
    fig2.savefig(diag_path, dpi=150)
    plt.close(fig2)
    return map_path, diag_path


def make_measurement_covariance(fitness, stationary):
    pos_sigma = 0.025 + 0.45 * max(0.0, fitness)
    yaw_sigma = math.radians(1.2 if stationary else 2.2) + 0.35 * max(0.0, fitness)
    return np.diag([pos_sigma ** 2, pos_sigma ** 2, yaw_sigma ** 2])


def preprocess_scan(angles, ranges):
    scan_body = scan_to_xy(angles, ranges)
    scan_body = voxel_downsample(scan_body, SCAN_VOXEL_M, max_points=MAX_SCAN_POINTS)
    return scan_body


def run_fusion(max_seconds=0.0, output_prefix="fusion_output", save_interval=5.0):
    imu = IMUReader()
    lidar = LidarReader()
    imu.start()
    lidar.start()

    ekf = PlanarEKF()
    grid = OccupancyGrid()
    submap = LocalSubmap()
    diagnostics = FusionDiagnostics()

    calibration = calibrate_imu(imu)
    tilt = TiltEstimator(calibration.initial_roll, calibration.initial_pitch)

    start_time = time.monotonic()
    last_save = start_time
    previous_time = start_time

    accepted_scans = 0
    latest_fitness = None

    print("[SYSTEM] Fusion loop started.")

    try:
        while True:
            now = time.monotonic()
            dt = min(0.05, max(1e-3, now - previous_time))
            previous_time = now

            if max_seconds > 0.0 and now - start_time >= max_seconds:
                print("[SYSTEM] Timed run complete.")
                break

            acc_raw, gyro_raw = imu.read()
            corrected_acc = acc_raw - calibration.accel_bias
            corrected_gyro = gyro_raw - calibration.gyro_bias

            acc_norm = float(np.linalg.norm(corrected_acc))
            gyro_norm = float(np.linalg.norm(corrected_gyro))
            stationary = abs(acc_norm - GRAVITY) < STATIONARY_ACC_TOL and gyro_norm < STATIONARY_GYRO_TOL

            roll, pitch = tilt.update(corrected_acc, corrected_gyro, dt, stationary=stationary)
            planar_acc = level_acceleration(corrected_acc, roll, pitch)

            if stationary:
                ekf.adapt_biases_when_stationary(planar_acc, corrected_gyro[2])
                planar_acc = np.zeros(2, dtype=float)

            ekf.predict(planar_acc, corrected_gyro[2], dt)

            if stationary:
                ekf.zero_velocity_update()

            if lidar._new_scan.is_set():
                angles, ranges = lidar.get_scan()
                if len(angles) > 0:
                    scan_body = preprocess_scan(angles, ranges)
                    reference = submap.points()
                    latest_fitness = None

                    if len(reference) >= ICP_MIN_PAIRS and len(scan_body) >= ICP_MIN_PAIRS:
                        predicted_pose = ekf.pose
                        scan_world_pred = transform_points(scan_body, predicted_pose)
                        transform, stats = icp_2d(scan_world_pred, reference)
                        candidate_pose = apply_transform_to_pose(predicted_pose, transform)

                        delta_xy = np.linalg.norm(candidate_pose[:2] - predicted_pose[:2])
                        delta_yaw = abs(wrap(candidate_pose[2] - predicted_pose[2]))
                        latest_fitness = stats["fitness"]

                        if (
                            stats["num_pairs"] >= ICP_MIN_PAIRS
                            and stats["fitness"] < ICP_MAX_FITNESS
                            and delta_xy < ICP_MAX_TRANSLATION_STEP
                            and delta_yaw < ICP_MAX_YAW_STEP
                        ):
                            measurement_cov = make_measurement_covariance(stats["fitness"], stationary)
                            ekf.correct_pose(candidate_pose, measurement_cov)
                            accepted_scans += 1

                    current_pose = ekf.pose
                    scan_world = transform_points(scan_body, current_pose)
                    if len(scan_world) > 0:
                        if accepted_scans == 0 or latest_fitness is None or latest_fitness < ICP_MAX_FITNESS:
                            submap.add_scan(scan_world)
                        grid.update(current_pose, scan_body)

                    diagnostics.append(now - start_time, current_pose, ekf.speed, stationary, latest_fitness)

            if now - last_save >= save_interval:
                map_path, diag_path = save_outputs(grid, diagnostics, output_prefix)
                print(f"[SAVE] {map_path} | {diag_path}")
                last_save = now

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted. Saving final outputs...")
    finally:
        lidar.shutdown()
        imu._stop.set()
        map_path, diag_path = save_outputs(grid, diagnostics, output_prefix)
        print(f"[DONE] Saved {map_path} and {diag_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Improved LiDAR + IMU EKF fusion with drift suppression.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Stop automatically after this many seconds. 0 runs indefinitely.")
    parser.add_argument("--output-prefix", type=str, default="fusion_output", help="Prefix for saved plot filenames.")
    parser.add_argument("--save-interval", type=float, default=5.0, help="Periodic save interval in seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_fusion(max_seconds=args.max_seconds, output_prefix=args.output_prefix, save_interval=args.save_interval)


if __name__ == "__main__":
    main()
