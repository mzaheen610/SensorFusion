"""
Improved real-time 2D LiDAR-IMU EKF fusion for Raspberry Pi.

Key improvements for static drift reduction:
1. Startup IMU bias calibration (ax, ay, gz).
2. EKF state includes IMU bias terms.
3. Zero-velocity update (ZUPT) when stationary.
4. Slow online bias adaptation while stationary.
5. ICP quality gating and keyframe reset recovery.
6. Trajectory and diagnostics plots saved to disk.
"""

import math
import os
import threading
import time
from collections import deque

import numpy as np

try:
    from rplidar import RPLidar

    HAS_RPLIDAR = True
except ImportError:
    HAS_RPLIDAR = False
    print("[WARN] rplidar not installed. Lidar disabled.")

try:
    import board
    import busio
    from adafruit_icm20x import ICM20948

    HAS_IMU = True
except ImportError:
    HAS_IMU = False
    print("[WARN] adafruit-circuitpython-icm20x not installed. IMU disabled.")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# Configuration
# ==========================================================================

LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 256000
IMU_ADDR = 0x69

IMU_HZ = 100
DT_IMU = 1.0 / IMU_HZ

MAX_RANGE_M = 8.0
MIN_RANGE_M = 0.20
MAX_SCAN_POINTS = 280

# ICP settings
ICP_ITER = 20
ICP_TOL = 1e-3
ICP_MAX_CORR_M = 0.80
ICP_MAX_FITNESS = 0.12
ICP_MIN_POINTS = 70
ICP_MIN_PAIRS = 40
ICP_REJECT_RESET_COUNT = 12
ICP_MAX_TRANSLATION_CORR_M = 0.35
ICP_MAX_YAW_CORR_RAD = math.radians(12.0)

# Keyframe refresh criteria
KEYFRAME_DISTANCE_M = 0.05
KEYFRAME_YAW_RAD = math.radians(1.0)

# Stationary detection / ZUPT
ACC_NORM_TOL = 0.40
GYRO_NORM_TOL = 0.08
STATIONARY_CONFIRM_SAMPLES = 12
ZUPT_STD = 0.03

# Bias calibration
CALIBRATION_SAMPLES = 260
BIAS_ADAPT_ALPHA = 0.01

# EKF noise
ACC_NOISE_STD = 0.45
GYRO_NOISE_STD = 0.05
ACC_BIAS_RW_STD = 0.01
GYRO_BIAS_RW_STD = 0.005

# Motion stability guards
ACC_DEADBAND_MPS2 = 0.10
MAX_SPEED_MPS = 1.20
VEL_DAMPING_PER_S = 1.8
NO_ICP_DRAG_PER_S = 3.5
NO_ICP_TIMEOUT_S = 0.7
HARD_STOP_TIMEOUT_S = 2.0
MAX_ICP_ABS_JUMP_M = 0.45
MAX_ICP_ABS_JUMP_YAW_RAD = math.radians(10.0)

# Lightweight plotting settings
POINT_DECIMATION_STEP = 3
MAX_PLOT_POINTS = 50000
MAX_TRAJECTORY_POINTS = 25000
PLOT_POINT_SIZE = 1.0
PLOT_VOXEL_M = 0.06
ENABLE_DIAGNOSTICS_PLOT = False

SAVE_INTERVAL_S = 2.0
OUTPUT_DIR = "outputs"

# LiDAR stream recovery controls
LIDAR_MAX_BUF_MEAS = 12000
LIDAR_SOFT_RECOVERY_MAX = 4
LIDAR_SOFT_RECOVERY_SLEEP_S = 0.03
LIDAR_RECONNECT_DELAY_S = 0.20
LIDAR_RECOVERY_WINDOW_S = 12.0
LIDAR_MAX_RECOVERIES_PER_WINDOW = 6


# ==========================================================================
# Math helpers
# ==========================================================================


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angles(arr):
    return (arr + math.pi) % (2.0 * math.pi) - math.pi


def rot2(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points(points_xy, t_xy, yaw):
    r = rot2(yaw)
    return (r @ points_xy.T).T + t_xy


def scan_to_xy(angles_deg, ranges_m):
    rad = np.deg2rad(angles_deg)
    return np.column_stack([ranges_m * np.cos(rad), ranges_m * np.sin(rad)])


def filter_scan(angles, ranges):
    mask = (ranges > MIN_RANGE_M) & (ranges < MAX_RANGE_M)
    return angles[mask], ranges[mask]


def downsample_points(points_xy, max_points):
    n = len(points_xy)
    if n <= max_points:
        return points_xy
    idx = np.linspace(0, n - 1, max_points, dtype=np.int32)
    return points_xy[idx]


def voxel_filter_points(points_xy, voxel_size):
    if len(points_xy) == 0 or voxel_size <= 0:
        return points_xy
    q = np.floor(points_xy / voxel_size).astype(np.int32)
    _, keep = np.unique(q, axis=0, return_index=True)
    return points_xy[np.sort(keep)]


def icp_2d(src, ref, max_iter=ICP_ITER, tol=ICP_TOL, max_corr=ICP_MAX_CORR_M):
    if len(src) < ICP_MIN_POINTS or len(ref) < ICP_MIN_POINTS:
        return np.eye(3), float("inf"), 0

    t = np.eye(3)
    pts = src.copy()
    rmse = float("inf")
    pairs = 0

    for _ in range(max_iter):
        diffs = ref[:, None, :] - pts[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        idx = np.argmin(d2, axis=0)
        nn_d = np.sqrt(d2[idx, np.arange(d2.shape[1])])

        mask = nn_d < max_corr
        pairs = int(np.sum(mask))
        if pairs < ICP_MIN_PAIRS:
            return np.eye(3), float("inf"), pairs

        src_m = pts[mask]
        ref_m = ref[idx[mask]]

        mu_s = src_m.mean(axis=0)
        mu_r = ref_m.mean(axis=0)

        s_cov = (src_m - mu_s).T @ (ref_m - mu_r)
        u, _, vt = np.linalg.svd(s_cov)
        r_step = vt.T @ u.T

        if np.linalg.det(r_step) < 0:
            vt[-1, :] *= -1
            r_step = vt.T @ u.T

        t_step = mu_r - r_step @ mu_s

        d_t = np.eye(3)
        d_t[:2, :2] = r_step
        d_t[:2, 2] = t_step

        t = d_t @ t
        pts = (r_step @ pts.T).T + t_step

        rmse = float(np.sqrt(np.mean(np.sum((src_m @ r_step.T + t_step - ref_m) ** 2, axis=1))))
        d_yaw = math.atan2(r_step[1, 0], r_step[0, 0])

        if np.linalg.norm(t_step) < tol and abs(d_yaw) < 1e-4:
            break

    return t, rmse, pairs


# ==========================================================================
# EKF
# state = [px, py, vx, vy, yaw, bax, bay, bgz]
# ==========================================================================


class EKF2DBias:
    def __init__(self):
        self.x = np.zeros(8, dtype=np.float64)

        self.P = np.diag(
            [
                0.25,
                0.25,
                0.40,
                0.40,
                0.12,
                0.08,
                0.08,
                0.02,
            ]
        )

    @property
    def pose(self):
        return np.array([self.x[0], self.x[1], self.x[4]], dtype=np.float64)

    @property
    def speed(self):
        return float(math.hypot(self.x[2], self.x[3]))

    def predict(self, ax, ay, gz, dt):
        if dt <= 0.0:
            return

        px, py, vx, vy, yaw, bax, bay, bgz = self.x

        ax_b = ax - bax
        ay_b = ay - bay
        wz = gz - bgz

        # Ignore tiny linear acceleration fluctuations that otherwise integrate into drift.
        if abs(ax_b) < ACC_DEADBAND_MPS2:
            ax_b = 0.0
        if abs(ay_b) < ACC_DEADBAND_MPS2:
            ay_b = 0.0

        c = math.cos(yaw)
        s = math.sin(yaw)

        ax_w = c * ax_b - s * ay_b
        ay_w = s * ax_b + c * ay_b

        px_n = px + vx * dt + 0.5 * ax_w * dt * dt
        py_n = py + vy * dt + 0.5 * ay_w * dt * dt
        vx_n = vx + ax_w * dt
        vy_n = vy + ay_w * dt
        yaw_n = wrap_angle(yaw + wz * dt)

        # Apply baseline velocity damping and speed clamp to prevent IMU-only blow-up.
        base_damp = math.exp(-VEL_DAMPING_PER_S * dt)
        vx_n *= base_damp
        vy_n *= base_damp
        speed_n = math.hypot(vx_n, vy_n)
        if speed_n > MAX_SPEED_MPS:
            scale = MAX_SPEED_MPS / speed_n
            vx_n *= scale
            vy_n *= scale

        self.x[0] = px_n
        self.x[1] = py_n
        self.x[2] = vx_n
        self.x[3] = vy_n
        self.x[4] = yaw_n

        d_axw_dyaw = -s * ax_b - c * ay_b
        d_ayw_dyaw = c * ax_b - s * ay_b

        f = np.eye(8, dtype=np.float64)
        f[0, 2] = dt
        f[1, 3] = dt

        f[0, 4] = 0.5 * dt * dt * d_axw_dyaw
        f[1, 4] = 0.5 * dt * dt * d_ayw_dyaw
        f[2, 4] = dt * d_axw_dyaw
        f[3, 4] = dt * d_ayw_dyaw

        f[0, 5] = -0.5 * dt * dt * c
        f[0, 6] = 0.5 * dt * dt * s
        f[1, 5] = -0.5 * dt * dt * s
        f[1, 6] = -0.5 * dt * dt * c

        f[2, 5] = -dt * c
        f[2, 6] = dt * s
        f[3, 5] = -dt * s
        f[3, 6] = -dt * c

        f[4, 7] = -dt

        q = np.diag(
            [
                0.25 * (ACC_NOISE_STD**2) * dt**4,
                0.25 * (ACC_NOISE_STD**2) * dt**4,
                (ACC_NOISE_STD**2) * dt**2,
                (ACC_NOISE_STD**2) * dt**2,
                (GYRO_NOISE_STD**2) * dt**2,
                (ACC_BIAS_RW_STD**2) * dt,
                (ACC_BIAS_RW_STD**2) * dt,
                (GYRO_BIAS_RW_STD**2) * dt,
            ]
        )

        self.P = f @ self.P @ f.T + q

    def correct_pose(self, z, r):
        h = np.zeros((3, 8), dtype=np.float64)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        h[2, 4] = 1.0

        y = z - h @ self.x
        y[2] = wrap_angle(y[2])

        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ y
        self.x[4] = wrap_angle(self.x[4])
        self.P = (np.eye(8) - k @ h) @ self.P

    def correct_zero_velocity(self):
        h = np.zeros((2, 8), dtype=np.float64)
        h[0, 2] = 1.0
        h[1, 3] = 1.0

        z = np.zeros(2, dtype=np.float64)
        y = z - h @ self.x

        r = np.diag([ZUPT_STD**2, ZUPT_STD**2])
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ y
        self.P = (np.eye(8) - k @ h) @ self.P


# ==========================================================================
# Sensor readers
# ==========================================================================


class IMUReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.acc = np.zeros(3, dtype=np.float64)
        self.gyro = np.zeros(3, dtype=np.float64)
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
                    self.acc[:] = (ax, ay, az)
                    self.gyro[:] = (gx, gy, gz)
                time.sleep(DT_IMU)
        except Exception as exc:
            print(f"[IMU] Error: {exc}")

    def read(self):
        with self._lock:
            return self.acc.copy(), self.gyro.copy()

    def stop(self):
        self._stop.set()


class LidarReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._angles = np.array([], dtype=np.float64)
        self._ranges = np.array([], dtype=np.float64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._new = threading.Event()
        self._scan_count = 0
        self._lidar = None

    def run(self):
        if not HAS_RPLIDAR:
            return

        while not self._stop.is_set():
            lidar = None
            try:
                lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=3)
                lidar.get_health = lambda: ("Good", 0)
                lidar.connect()
                _ = lidar.get_info()
                lidar.start_motor()
                time.sleep(0.4)
                lidar.clean_input()
                self._lidar = lidar

                scan_iter = lidar.iter_scans(max_buf_meas=LIDAR_MAX_BUF_MEAS)
                soft_recoveries = 0
                recovery_events = deque()

                while not self._stop.is_set():
                    try:
                        scan = next(scan_iter)
                    except StopIteration:
                        break
                    except Exception as exc:
                        msg = str(exc)
                        recoverable = (
                            "New scan flags mismatch" in msg
                            or "Check bit not equal to 1" in msg
                            or "Incorrect descriptor starting bytes" in msg
                            or "Too many bytes in the input buffer" in msg
                        )

                        if recoverable:
                            now_err = time.monotonic()
                            recovery_events.append(now_err)
                            while recovery_events and (
                                now_err - recovery_events[0] > LIDAR_RECOVERY_WINDOW_S
                            ):
                                recovery_events.popleft()

                            if len(recovery_events) >= LIDAR_MAX_RECOVERIES_PER_WINDOW:
                                raise RuntimeError(
                                    "Excessive recoverable LiDAR stream errors "
                                    f"({len(recovery_events)} in {LIDAR_RECOVERY_WINDOW_S:.1f}s): {msg}"
                                )

                        if recoverable and soft_recoveries < LIDAR_SOFT_RECOVERY_MAX:
                            soft_recoveries += 1
                            print(
                                f"[Lidar] Soft recovery {soft_recoveries}/{LIDAR_SOFT_RECOVERY_MAX}: {msg}"
                            )
                            try:
                                # Stop current scan command before reissuing iter_scans.
                                lidar.stop()
                            except Exception:
                                pass
                            try:
                                lidar.clean_input()
                            except Exception:
                                pass
                            scan_iter = lidar.iter_scans(max_buf_meas=LIDAR_MAX_BUF_MEAS)
                            time.sleep(LIDAR_SOFT_RECOVERY_SLEEP_S)
                            continue

                        raise

                    soft_recoveries = 0

                    a = np.array([m[1] for m in scan if m[0] > 0], dtype=np.float64)
                    r = np.array([m[2] / 1000.0 for m in scan if m[0] > 0], dtype=np.float64)
                    a, r = filter_scan(a, r)
                    if len(a) == 0:
                        continue

                    with self._lock:
                        self._angles = a
                        self._ranges = r
                        self._scan_count += 1
                    self._new.set()

            except Exception as exc:
                if self._stop.is_set() and "Bad file descriptor" in str(exc):
                    pass
                else:
                    print(f"[Lidar] Hard reconnect: {exc}, scans so far: {self._scan_count}")
            finally:
                self._safe_shutdown_lidar(lidar)
                self._lidar = None

            time.sleep(LIDAR_RECONNECT_DELAY_S)

    @staticmethod
    def _safe_shutdown_lidar(lidar):
        if lidar is None:
            return
        try:
            lidar.stop()
        except Exception:
            pass
        try:
            lidar.stop_motor()
        except Exception:
            pass
        try:
            lidar.disconnect()
        except Exception:
            pass

    def get_scan(self):
        self._new.clear()
        with self._lock:
            return self._angles.copy(), self._ranges.copy()

    def stop(self):
        self._stop.set()
        self._safe_shutdown_lidar(self._lidar)


# ==========================================================================
# Motion classification / calibration / plotting
# ==========================================================================


class MotionClassifier:
    def __init__(self):
        self._counter = 0

    def update(self, acc_xyz, gyro_xyz):
        acc_norm = float(np.linalg.norm(acc_xyz))
        gyro_norm = float(np.linalg.norm(gyro_xyz))

        is_candidate = (
            abs(acc_norm - 9.81) < ACC_NORM_TOL and gyro_norm < GYRO_NORM_TOL
        )

        if is_candidate:
            self._counter = min(self._counter + 1, STATIONARY_CONFIRM_SAMPLES * 2)
        else:
            self._counter = max(self._counter - 1, 0)

        return self._counter >= STATIONARY_CONFIRM_SAMPLES


def calibrate_imu_bias(imu_reader):
    if not HAS_IMU:
        return 0.0, 0.0, 0.0

    print(f"[CAL] Collecting {CALIBRATION_SAMPLES} IMU samples. Keep setup still.")
    acc_samples = []
    gyro_samples = []

    for _ in range(CALIBRATION_SAMPLES):
        acc, gyro = imu_reader.read()
        acc_samples.append(acc)
        gyro_samples.append(gyro)
        time.sleep(DT_IMU)

    acc_m = np.mean(np.array(acc_samples), axis=0)
    gyro_m = np.mean(np.array(gyro_samples), axis=0)

    bax0 = float(acc_m[0])
    bay0 = float(acc_m[1])
    bgz0 = float(gyro_m[2])

    print(
        "[CAL] Bias init"
        f" bax={bax0:+.4f} m/s^2"
        f" bay={bay0:+.4f} m/s^2"
        f" bgz={bgz0:+.4f} rad/s"
    )
    return bax0, bay0, bgz0


def save_map_plot(point_cloud_xy, trajectory_xy, file_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#efefef")
    ax.set_facecolor("#efefef")

    if len(point_cloud_xy) > 0:
        pts = np.array(point_cloud_xy, dtype=np.float64)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=PLOT_POINT_SIZE,
            c="#ff00ff",
            alpha=0.85,
            linewidths=0,
            label="Scan Points",
        )

    if len(trajectory_xy) > 0:
        path = np.array(trajectory_xy)
        ax.plot(path[:, 0], path[:, 1], color="#1f55ff", linewidth=1.8, label="EKF Trajectory")
        ax.scatter(path[0, 0], path[0, 1], s=40, c="#1f55ff", label="Start", zorder=3)
        ax.scatter(path[-1, 0], path[-1, 1], s=45, c="#0033aa", label="Current", zorder=3)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Map of the Environment and Robot Trajectory")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.45)
    ax.axis("equal")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(file_path, dpi=130)
    plt.close(fig)


def save_diagnostics_plot(hist, file_path):
    if len(hist["time_s"]) < 2:
        return

    t = np.array(hist["time_s"])
    speed = np.array(hist["speed_mps"])
    yaw_deg = np.rad2deg(np.array(hist["yaw_rad"]))
    fit = np.array(hist["icp_fitness"])
    stat = np.array(hist["stationary"], dtype=np.float64)
    bax = np.array(hist["bax"])
    bay = np.array(hist["bay"])
    bgz = np.array(hist["bgz"])

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("#fbfaf8")

    axs[0].plot(t, speed, color="#d62728", linewidth=1.5)
    axs[0].plot(t, stat * max(0.15, np.nanmax(speed) + 0.02), color="#2ca02c", alpha=0.35)
    axs[0].set_ylabel("Speed m/s")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, yaw_deg, color="#1f77b4", linewidth=1.4)
    axs[1].set_ylabel("Yaw deg")
    axs[1].grid(True, alpha=0.3)

    valid = np.isfinite(fit)
    axs[2].plot(t[valid], fit[valid], color="#9467bd", linewidth=1.3)
    axs[2].axhline(ICP_MAX_FITNESS, color="#8c564b", linestyle="--", linewidth=1.0)
    axs[2].set_ylabel("ICP fitness")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t, bax, color="#ff7f0e", linewidth=1.2, label="bax")
    axs[3].plot(t, bay, color="#bcbd22", linewidth=1.2, label="bay")
    axs[3].plot(t, bgz, color="#17becf", linewidth=1.2, label="bgz")
    axs[3].set_ylabel("Bias")
    axs[3].set_xlabel("Time s")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    fig.suptitle("EKF Diagnostics")
    fig.tight_layout()
    fig.savefig(file_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    imu = IMUReader()
    lidar = LidarReader()

    imu.start()
    lidar.start()

    ekf = EKF2DBias()
    motion_cls = MotionClassifier()

    # Keep a bounded scatter cloud and trajectory for fast plotting.
    point_cloud = deque(maxlen=MAX_PLOT_POINTS)
    trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)

    time.sleep(0.8)
    bax0, bay0, bgz0 = calibrate_imu_bias(imu)
    ekf.x[5] = bax0
    ekf.x[6] = bay0
    ekf.x[7] = bgz0

    keyframe_pts = None
    keyframe_pose = None
    rejected_icp = 0

    hist = {
        "time_s": [],
        "speed_mps": [],
        "yaw_rad": [],
        "icp_fitness": [],
        "stationary": [],
        "bax": [],
        "bay": [],
        "bgz": [],
    }

    t0 = time.monotonic()
    t_prev = t0
    t_last_save = t0
    t_last_hist = t0
    t_last_icp_accept = t0

    map_file = os.path.join(OUTPUT_DIR, "ekf_map.png")
    diag_file = os.path.join(OUTPUT_DIR, "ekf_diagnostics.png")

    print("[SYSTEM] Improved EKF fusion started.")
    print(f"[SYSTEM] Saving outputs in: {OUTPUT_DIR}")

    try:
        while True:
            now = time.monotonic()
            dt = max(1e-3, min(0.05, now - t_prev))
            t_prev = now
            icp_accepted_this_cycle = False

            acc, gyro = imu.read()
            is_stationary = motion_cls.update(acc, gyro)

            if is_stationary:
                ekf.x[5] = (1.0 - BIAS_ADAPT_ALPHA) * ekf.x[5] + BIAS_ADAPT_ALPHA * acc[0]
                ekf.x[6] = (1.0 - BIAS_ADAPT_ALPHA) * ekf.x[6] + BIAS_ADAPT_ALPHA * acc[1]
                ekf.x[7] = (1.0 - BIAS_ADAPT_ALPHA) * ekf.x[7] + BIAS_ADAPT_ALPHA * gyro[2]

            ekf.predict(float(acc[0]), float(acc[1]), float(gyro[2]), dt)

            if is_stationary:
                ekf.correct_zero_velocity()

            icp_fitness = float("nan")

            if lidar._new.is_set():
                angles, ranges = lidar.get_scan()
                if len(angles) > 0:
                    angles, ranges = filter_scan(angles, ranges)
                    if len(angles) >= ICP_MIN_POINTS:
                        icp_accepted = False
                        scan_local = scan_to_xy(angles, ranges)
                        scan_local = downsample_points(scan_local, MAX_SCAN_POINTS)
                        scan_local = voxel_filter_points(scan_local, PLOT_VOXEL_M)

                        pose_pred = ekf.pose
                        scan_world = transform_points(scan_local, pose_pred[:2], pose_pred[2])

                        if keyframe_pts is None:
                            keyframe_pts = scan_world.copy()
                            keyframe_pose = pose_pred.copy()
                            icp_accepted = True
                            print(f"[SLAM] Initial keyframe with {len(scan_world)} points")
                        else:
                            t_icp, icp_fitness, pairs = icp_2d(scan_world, keyframe_pts)
                            delta_yaw = abs(math.atan2(t_icp[1, 0], t_icp[0, 0]))
                            delta_trans = float(np.linalg.norm(t_icp[:2, 2]))

                            if (
                                pairs >= ICP_MIN_PAIRS
                                and icp_fitness < ICP_MAX_FITNESS
                                and delta_trans < ICP_MAX_TRANSLATION_CORR_M
                                and delta_yaw < ICP_MAX_YAW_CORR_RAD
                            ):
                                delta_yaw_signed = math.atan2(t_icp[1, 0], t_icp[0, 0])

                                corrected_xy = t_icp[:2, :2] @ pose_pred[:2] + t_icp[:2, 2]
                                z_pose = np.array(
                                    [
                                        corrected_xy[0],
                                        corrected_xy[1],
                                        wrap_angle(pose_pred[2] + delta_yaw_signed),
                                    ],
                                    dtype=np.float64,
                                )

                                jump_m = float(np.linalg.norm(z_pose[:2] - pose_pred[:2]))
                                jump_yaw = abs(wrap_angle(z_pose[2] - pose_pred[2]))

                                if (
                                    jump_m > MAX_ICP_ABS_JUMP_M
                                    or jump_yaw > MAX_ICP_ABS_JUMP_YAW_RAD
                                ):
                                    rejected_icp += 1
                                else:
                                    rejected_icp = 0
                                    icp_accepted = True
                                    icp_accepted_this_cycle = True
                                    t_last_icp_accept = now

                                    sigma_xy = float(
                                        np.clip(0.015 + 0.55 * icp_fitness, 0.015, 0.20)
                                    )
                                    sigma_yaw = float(
                                        np.clip(
                                            math.radians(0.8) + 0.9 * icp_fitness,
                                            math.radians(0.8),
                                            math.radians(8.0),
                                        )
                                    )
                                    r_icp = np.diag(
                                        [
                                            sigma_xy * sigma_xy,
                                            sigma_xy * sigma_xy,
                                            sigma_yaw * sigma_yaw,
                                        ]
                                    )

                                    ekf.correct_pose(z_pose, r_icp)
                                    pose_upd = ekf.pose
                                    scan_world = transform_points(
                                        scan_local, pose_upd[:2], pose_upd[2]
                                    )

                                    dist_from_keyframe = float(
                                        np.linalg.norm(pose_upd[:2] - keyframe_pose[:2])
                                    )
                                    yaw_from_keyframe = abs(
                                        wrap_angle(pose_upd[2] - keyframe_pose[2])
                                    )

                                    if (
                                        dist_from_keyframe > KEYFRAME_DISTANCE_M
                                        or yaw_from_keyframe > KEYFRAME_YAW_RAD
                                    ):
                                        keyframe_pts = scan_world.copy()
                                        keyframe_pose = pose_upd.copy()
                                if rejected_icp >= ICP_REJECT_RESET_COUNT:
                                    keyframe_pts = scan_world.copy()
                                    keyframe_pose = pose_pred.copy()
                                    rejected_icp = 0
                                    point_cloud.clear()
                            else:
                                rejected_icp += 1
                                if rejected_icp >= ICP_REJECT_RESET_COUNT:
                                    keyframe_pts = scan_world.copy()
                                    keyframe_pose = pose_pred.copy()
                                    rejected_icp = 0
                                    point_cloud.clear()

                        if icp_accepted:
                            scan_for_plot = scan_world[::POINT_DECIMATION_STEP]
                            scan_for_plot = voxel_filter_points(scan_for_plot, PLOT_VOXEL_M)
                            for pt in scan_for_plot:
                                point_cloud.append((float(pt[0]), float(pt[1])))

                        trajectory.append((float(ekf.x[0]), float(ekf.x[1])))

            # If LiDAR corrections are missing, aggressively damp velocity to prevent runaway path growth.
            t_since_icp = now - t_last_icp_accept
            if t_since_icp > NO_ICP_TIMEOUT_S and not icp_accepted_this_cycle:
                drag = math.exp(-NO_ICP_DRAG_PER_S * dt)
                ekf.x[2] *= drag
                ekf.x[3] *= drag
                if t_since_icp > HARD_STOP_TIMEOUT_S:
                    ekf.x[2] = 0.0
                    ekf.x[3] = 0.0

            if ENABLE_DIAGNOSTICS_PLOT and now - t_last_hist > 0.05:
                hist["time_s"].append(now - t0)
                hist["speed_mps"].append(ekf.speed)
                hist["yaw_rad"].append(float(ekf.x[4]))
                hist["icp_fitness"].append(icp_fitness)
                hist["stationary"].append(1.0 if is_stationary else 0.0)
                hist["bax"].append(float(ekf.x[5]))
                hist["bay"].append(float(ekf.x[6]))
                hist["bgz"].append(float(ekf.x[7]))
                t_last_hist = now

            if now - t_last_save > SAVE_INTERVAL_S:
                save_map_plot(point_cloud, trajectory, map_file)
                if ENABLE_DIAGNOSTICS_PLOT:
                    save_diagnostics_plot(hist, diag_file)
                pose = ekf.pose
                fit_txt = f"{icp_fitness:.3f}" if np.isfinite(icp_fitness) else "n/a"
                print(
                    f"[{time.strftime('%H:%M:%S')}] pose=({pose[0]:+.3f},{pose[1]:+.3f},"
                    f"{math.degrees(pose[2]):+.1f}deg) speed={ekf.speed:.3f} m/s icp={fit_txt}"
                )
                t_last_save = now

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping. Saving final plots...")
        save_map_plot(point_cloud, trajectory, os.path.join(OUTPUT_DIR, "ekf_map_final.png"))
        if ENABLE_DIAGNOSTICS_PLOT:
            save_diagnostics_plot(hist, os.path.join(OUTPUT_DIR, "ekf_diagnostics_final.png"))

    finally:
        lidar.stop()
        imu.stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
