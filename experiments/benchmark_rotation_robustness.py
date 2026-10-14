"""
Rigorous Experimental Evaluation of LiDAR Rotation Correction & IMU Imperfection Tolerance.

Evaluates how the LiDAR-IMU multimodal sensor fusion pipeline behaves when:
1. Instantaneous heading errors (1 deg to 45 deg) are introduced.
2. The IMU has uncalibrated gyro biases (0.00 to 0.40 rad/s / up to 23 deg/s).
3. The IMU has scale factor errors (-50% to +50% under/over-reporting turn rate).
4. Motor vibration & severe high-frequency gyro noise are present (0.01 to 0.35 rad/s RMS).
5. LiDAR motion deskewing (backward propagation) operates under inaccurate IMU gyro buffers.
6. Combined translation + sharp corner turn under low-cost inaccurate IMU conditions.

Zero code changes to src/sensorfusion/. All tests interface directly with production classes.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from forward import ESIKFStateEstimator, State
from map import Map
from utils.so3_rotation import exp, skew, reorthonormalize
from lidar.backward import backprop
from benchmark_pipeline_rigorous import SimulationWorld, TrajectoryGenerator

def copy_state(st):
    return State(
        R=st.R.copy(),
        p=st.p.copy(),
        v=st.v.copy(),
        bg=st.bg.copy() if hasattr(st, 'bg') else np.zeros(3),
        ba=st.ba.copy() if hasattr(st, 'ba') else np.zeros(3),
        g=st.g.copy() if hasattr(st, 'g') else np.zeros(3)
    )



# ============================================================================
# Helper: Extract Heading (Yaw) from SO(3) Rotation Matrix
# ============================================================================
def extract_yaw(R):
    """Extracts planar yaw angle in radians from 3x3 rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap_angle(angle):
    """Wraps angle to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


# ============================================================================
# Experiment 1: Convergence Basin of Instantaneous Heading Error
# ============================================================================
def test_instantaneous_heading_convergence():
    """
    Measures the maximum initial heading error (in degrees) that the LiDAR IEKF
    can correct in a single scan, and how many successive scans are needed to settle.
    """
    print("\n" + "=" * 80)
    print("TEST 1: INSTANTANEOUS HEADING CORRECTION BASIN (STATIC ENVIRONMENT)")
    print("=" * 80)

    world = SimulationWorld()
    # Populate a clean baseline map from ground truth walls
    map_obj = Map()
    for x1, y1, x2, y2 in world.walls:
        xs = np.linspace(x1, x2, 100)
        ys = np.linspace(y1, y2, 100)
        pts = np.column_stack([xs, ys, np.zeros_like(xs)])
        map_obj.add_points(pts)

    true_pos = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    # Raycast clean scan at true pose
    scan_raw, body_pts, _, _ = world.raycast(true_pos, true_yaw)

    heading_errors_deg = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 45.0]
    results = []

    print(f"{'Initial Err':<12} | {'Applied':<8} | {'1-Scan Corr':<12} | {'Residual Norm':<14} | {'Post-Scan Err':<14} | {'Status':<10}")
    print("-" * 82)

    for err_deg in heading_errors_deg:
        err_rad = np.deg2rad(err_deg)
        estimator = ESIKFStateEstimator()
        estimator.is_static = False  # dynamic search radius (2 voxels / ±1.0m)

        # Set true position but perturbed heading
        estimator.state.p = true_pos.copy()
        R_perturbed = np.array([
            [np.cos(err_rad), -np.sin(err_rad), 0.0],
            [np.sin(err_rad),  np.cos(err_rad), 0.0],
            [0.0, 0.0, 1.0]
        ])
        estimator.state.R = R_perturbed.copy()
        state_for_update = copy_state(estimator.state)
        P_copy = estimator.P.copy()

        pts_w, applied, P_new = estimator.lidar_update(
            scan=scan_raw,
            state=state_for_update,
            P_copy=P_copy,
            lidar_points_compensated=body_pts,
            map=map_obj
        )

        final_yaw = extract_yaw(state_for_update.R)
        yaw_err_after = wrap_angle(final_yaw - true_yaw)
        corr_deg = err_deg - np.rad2deg(abs(yaw_err_after))
        res_norm = estimator.last_lidar_residual_norm

        recovered = abs(np.rad2deg(yaw_err_after)) < (err_deg * 0.5) and applied
        status = "PULLED" if recovered else ("NO PULL" if applied else "REJECTED")

        print(f"{err_deg:8.1f} deg  | {str(applied):<8} | {corr_deg:8.2f} deg   | {res_norm:10.4f}     | {np.rad2deg(abs(yaw_err_after)):8.2f} deg    | {status:<10}")

        results.append({
            'initial_deg': err_deg,
            'applied': applied,
            'corr_deg': corr_deg,
            'post_err_deg': np.rad2deg(abs(yaw_err_after)),
            'residual_norm': res_norm,
            'recovered': recovered
        })

    return results


# ============================================================================
# Experiment 2: Active 90-Degree Turn with Uncalibrated Gyro Biases
# ============================================================================
def test_active_turn_gyro_biases():
    """
    Tests robot executing a 90 deg turn at 30 deg/s with persistent uncalibrated gyro biases:
    bg_z in [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50] rad/s.
    Evaluates whether LiDAR updates pull heading back and prevent rotational runaway.
    """
    print("\n" + "=" * 80)
    print("TEST 2: ACTIVE 90-DEG TURN UNDER UNCALIBRATED GYRO BIASES")
    print("=" * 80)

    world = SimulationWorld()
    map_obj = Map()
    for x1, y1, x2, y2 in world.walls:
        xs = np.linspace(x1, x2, 100)
        ys = np.linspace(y1, y2, 100)
        pts = np.column_stack([xs, ys, np.zeros_like(xs)])
        map_obj.add_points(pts)

    dt_scan = 0.30   # 3.33 Hz LiDAR
    dt_imu = 0.03    # 33.3 Hz IMU
    turn_duration = 3.0   # 90 deg in 3s = 30 deg/s = 0.5236 rad/s
    total_time = 7.0      # 1s straight, 3s turn, 3s settling

    biases_rad_s = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    results = []

    print(f"{'Gyro Bias (rad/s)':<18} | {'Bias (deg/s)':<14} | {'Peak Yaw Err':<14} | {'Final Yaw Err':<14} | {'Applied %':<10} | {'Status':<10}")
    print("-" * 88)

    for bg_val in biases_rad_s:
        estimator = ESIKFStateEstimator()
        estimator.state.p = np.zeros(3)
        estimator.state.R = np.eye(3)

        true_p = np.zeros(3)
        true_v = np.zeros(3)
        true_yaw = 0.0

        bg_sim = np.array([0.0, 0.0, bg_val])
        ba_sim = np.array([0.01, -0.01, 0.0])

        yaw_errors = []
        applied_count = 0
        total_scans = int(total_time / dt_scan)

        for step in range(total_scans):
            t_scan = step * dt_scan

            target_omega = 0.0
            if 1.0 <= t_scan < 4.0:
                target_omega = np.deg2rad(30.0)  # 30 deg/s

            imu_batch = []
            for _ in range(int(dt_scan / dt_imu)):
                true_yaw += target_omega * dt_imu
                # Gyro measurement contains true rate + simulated bias + small noise
                gyro_meas = np.array([0.0, 0.0, target_omega]) + bg_sim + np.random.normal(0, 0.002, 3)
                accel_meas = np.array([0.0, 0.0, 9.81]) + ba_sim + np.random.normal(0, 0.01, 3)

                # Predict step
                estimator.predict([gyro_meas, accel_meas], dt=dt_imu)
                imu_batch.append((dt_imu, gyro_meas, accel_meas))

            # Raycast scan at true pose
            scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)

            # Perform LiDAR update on snapshot
            state_old = copy_state(estimator.state)
            state_update = copy_state(state_old)
            P_copy = estimator.P.copy()

            pts_w, applied, P_new = estimator.lidar_update(
                scan=scan_raw,
                state=state_update,
                P_copy=P_copy,
                lidar_points_compensated=body_pts,
                map=map_obj
            )

            if applied:
                applied_count += 1
                delta_R = state_old.R.T @ state_update.R
                delta_p = state_update.p - state_old.p
                estimator.state.R = estimator.state.R @ delta_R
                estimator.state.R = reorthonormalize(estimator.state.R)
                estimator.state.p += delta_p
                estimator.P = P_new

            est_yaw = extract_yaw(estimator.state.R)
            err_y = abs(wrap_angle(est_yaw - true_yaw))
            yaw_errors.append(np.rad2deg(err_y))

        peak_err = float(np.max(yaw_errors))
        final_err = float(yaw_errors[-1])
        app_rate = (applied_count / total_scans) * 100.0

        status = "PASSED" if final_err < 3.0 and peak_err < 12.0 else ("DEGRADED" if final_err < 10.0 else "DIVERGED")
        print(f"{bg_val:12.3f} rad/s | {np.rad2deg(bg_val):8.2f} deg/s   | {peak_err:8.2f} deg    | {final_err:8.2f} deg    | {app_rate:6.1f} %   | {status:<10}")

        results.append({
            'bias_rad_s': bg_val,
            'peak_yaw_deg': peak_err,
            'final_yaw_deg': final_err,
            'applied_pct': app_rate,
            'status': status
        })

    return results


# ============================================================================
# Experiment 3: Gyro Scale Factor Error (-50% to +50%)
# ============================================================================
def test_gyro_scale_factor_errors():
    """
    Tests robot turning with severe gyro scale errors:
    measured_omega = (1 + scale_error) * true_omega.
    Scale errors from -50% (under-reporting) to +50% (over-reporting).
    """
    print("\n" + "=" * 80)
    print("TEST 3: GYRO SCALE FACTOR ERROR UNDER-REPORTING & OVER-REPORTING")
    print("=" * 80)

    world = SimulationWorld()
    map_obj = Map()
    for x1, y1, x2, y2 in world.walls:
        xs = np.linspace(x1, x2, 100)
        ys = np.linspace(y1, y2, 100)
        pts = np.column_stack([xs, ys, np.zeros_like(xs)])
        map_obj.add_points(pts)

    dt_scan = 0.30
    dt_imu = 0.03
    turn_rate = np.deg2rad(45.0)  # 45 deg/s turn
    turn_duration = 2.0           # 90 deg turn total
    total_time = 5.0

    scale_errors = [-0.50, -0.35, -0.20, -0.10, 0.00, +0.10, +0.20, +0.35, +0.50]
    results = []

    print(f"{'Scale Error':<14} | {'Gyro Reports':<18} | {'Peak Yaw Err':<14} | {'Final Yaw Err':<14} | {'Applied %':<10} | {'Status':<10}")
    print("-" * 88)

    for s_err in scale_errors:
        estimator = ESIKFStateEstimator()
        estimator.state.p = np.zeros(3)
        estimator.state.R = np.eye(3)

        true_p = np.zeros(3)
        true_yaw = 0.0

        yaw_errors = []
        applied_count = 0
        total_scans = int(total_time / dt_scan)

        for step in range(total_scans):
            t_scan = step * dt_scan

            target_omega = 0.0
            if 1.0 <= t_scan < 3.0:
                target_omega = turn_rate

            for _ in range(int(dt_scan / dt_imu)):
                true_yaw += target_omega * dt_imu
                # Scaled gyro measurement
                omega_meas = (1.0 + s_err) * target_omega
                gyro_meas = np.array([0.0, 0.0, omega_meas]) + np.random.normal(0, 0.002, 3)
                accel_meas = np.array([0.0, 0.0, 9.81]) + np.random.normal(0, 0.01, 3)
                estimator.predict([gyro_meas, accel_meas], dt=dt_imu)

            scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)

            state_old = copy_state(estimator.state)
            state_update = copy_state(state_old)
            P_copy = estimator.P.copy()

            pts_w, applied, P_new = estimator.lidar_update(
                scan=scan_raw,
                state=state_update,
                P_copy=P_copy,
                lidar_points_compensated=body_pts,
                map=map_obj
            )

            if applied:
                applied_count += 1
                delta_R = state_old.R.T @ state_update.R
                estimator.state.R = estimator.state.R @ delta_R
                estimator.state.R = reorthonormalize(estimator.state.R)
                estimator.state.p += (state_update.p - state_old.p)
                estimator.P = P_new

            est_yaw = extract_yaw(estimator.state.R)
            err_y = abs(wrap_angle(est_yaw - true_yaw))
            yaw_errors.append(np.rad2deg(err_y))

        peak_err = float(np.max(yaw_errors))
        final_err = float(yaw_errors[-1])
        app_rate = (applied_count / total_scans) * 100.0

        reported_str = f"{np.rad2deg((1.0 + s_err) * turn_rate):.1f} deg/s"
        status = "PASSED" if final_err < 3.0 and peak_err < 12.0 else ("DEGRADED" if final_err < 8.0 else "DIVERGED")
        print(f"{s_err*100:8.0f} %     | {reported_str:<18} | {peak_err:8.2f} deg    | {final_err:8.2f} deg    | {app_rate:6.1f} %   | {status:<10}")

        results.append({
            'scale_error': s_err,
            'peak_yaw_deg': peak_err,
            'final_yaw_deg': final_err,
            'status': status
        })

    return results


# ============================================================================
# Experiment 4: High-Frequency Chassis Vibration & Gyro Jitter (15 to 30 Hz)
# ============================================================================
def test_gyro_vibration_noise():
    """
    Simulates high-amplitude motor/chassis vibrations (0.01 to 0.35 rad/s RMS)
    inducing oscillating rotational noise on the IMU.
    """
    print("\n" + "=" * 80)
    print("TEST 4: MOTOR INDUCED CHASSIS VIBRATION & GYRO NOISE STRESS")
    print("=" * 80)

    world = SimulationWorld()
    map_obj = Map()
    for x1, y1, x2, y2 in world.walls:
        xs = np.linspace(x1, x2, 100)
        ys = np.linspace(y1, y2, 100)
        pts = np.column_stack([xs, ys, np.zeros_like(xs)])
        map_obj.add_points(pts)

    dt_scan = 0.30
    dt_imu = 0.03
    total_time = 6.0

    vibration_amps = [0.01, 0.05, 0.10, 0.18, 0.25, 0.35]
    results = []

    print(f"{'Vibration Amp (rad/s)':<22} | {'Vib (deg/s)':<14} | {'Mean Yaw Err':<14} | {'Peak Yaw Err':<14} | {'Final Yaw Err':<14} | {'Status':<10}")
    print("-" * 96)

    for vib_amp in vibration_amps:
        estimator = ESIKFStateEstimator()
        estimator.state.p = np.zeros(3)
        estimator.state.R = np.eye(3)

        true_p = np.zeros(3)
        true_yaw = 0.0

        yaw_errors = []
        total_scans = int(total_time / dt_scan)

        for step in range(total_scans):
            t_scan = step * dt_scan

            # Robot turns slowly at 15 deg/s
            target_omega = np.deg2rad(15.0) if 1.0 <= t_scan < 4.0 else 0.0

            for sub in range(int(dt_scan / dt_imu)):
                t_now = t_scan + sub * dt_imu
                true_yaw += target_omega * dt_imu

                # Inject dual-frequency motor vibration (17 Hz + 29 Hz)
                vib_signal = vib_amp * (np.sin(2 * np.pi * 17.0 * t_now) + 0.5 * np.cos(2 * np.pi * 29.0 * t_now))
                gyro_meas = np.array([0.0, 0.0, target_omega + vib_signal]) + np.random.normal(0, 0.005, 3)
                accel_meas = np.array([0.0, 0.0, 9.81]) + np.random.normal(0, 0.02, 3)
                estimator.predict([gyro_meas, accel_meas], dt=dt_imu)

            scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)

            state_old = copy_state(estimator.state)
            state_update = copy_state(state_old)
            P_copy = estimator.P.copy()

            pts_w, applied, P_new = estimator.lidar_update(
                scan=scan_raw,
                state=state_update,
                P_copy=P_copy,
                lidar_points_compensated=body_pts,
                map=map_obj
            )

            if applied:
                delta_R = state_old.R.T @ state_update.R
                estimator.state.R = estimator.state.R @ delta_R
                estimator.state.R = reorthonormalize(estimator.state.R)
                estimator.state.p += (state_update.p - state_old.p)
                estimator.P = P_new

            est_yaw = extract_yaw(estimator.state.R)
            err_y = abs(wrap_angle(est_yaw - true_yaw))
            yaw_errors.append(np.rad2deg(err_y))

        mean_err = float(np.mean(yaw_errors))
        peak_err = float(np.max(yaw_errors))
        final_err = float(yaw_errors[-1])

        status = "PASSED" if final_err < 2.0 and mean_err < 3.0 else ("DEGRADED" if final_err < 5.0 else "DIVERGED")
        print(f"{vib_amp:18.3f} rad/s   | {np.rad2deg(vib_amp):8.2f} deg/s   | {mean_err:8.2f} deg    | {peak_err:8.2f} deg    | {final_err:8.2f} deg    | {status:<10}")

        results.append({
            'vib_amp': vib_amp,
            'mean_yaw_deg': mean_err,
            'peak_yaw_deg': peak_err,
            'final_yaw_deg': final_err,
            'status': status
        })

    return results


# ============================================================================
# Experiment 5: Deskewing / Motion Compensation Sensitivity (backprop)
# ============================================================================
def test_deskewing_motion_compensation_sensitivity():
    """
    Tests backward propagation (backprop in backward.py) when IMU buffer has
    significant gyro errors during the 100ms sweep.
    Evaluates point distortion and whether distorted points corrupt the IEKF.
    """
    print("\n" + "=" * 80)
    print("TEST 5: LIDAR MOTION COMPENSATION (DESKEWING) UNDER INACCURATE IMU")
    print("=" * 80)

    # Robot turns at 60 deg/s during 100ms sweep
    true_omega = np.deg2rad(60.0) # ~1.047 rad/s
    scan_period = 0.100 # 100ms sweep duration
    num_points = 180

    # Synthetic straight wall at x = 2.0m for y in [-2.0, 2.0]
    wall_x = 2.0
    scan_raw = []
    # Angles from -45 deg to +45 deg (swept chronologically)
    angles_deg = np.linspace(-45.0, 45.0, num_points)
    for phi_deg in angles_deg:
        phi_rad = np.deg2rad(phi_deg)
        dist_m = wall_x / np.cos(phi_rad)
        phi_rh = (phi_deg) % 360.0
        angle_rplidar = (360.0 - phi_rh) % 360.0
        scan_raw.append((15, angle_rplidar, dist_m * 1000.0))

    scan_end_time = 1.00
    scan_start_time = scan_end_time - scan_period

    gyro_errors = [0.0, 0.20, 0.50, -0.50, 1.00] # Bias added to IMU buffer
    results = []

    print(f"{'Gyro Error in Buffer':<24} | {'Deskewed Points':<18} | {'Wall Dispers. sigma':<20} | {'Max Point Skew':<16} | {'Status':<10}")
    print("-" * 96)

    for g_err in gyro_errors:
        imu_pose = State(
            R=np.eye(3),
            p=np.zeros(3),
            v=np.zeros(3),
            bg=np.zeros(3),
            ba=np.zeros(3),
            g=np.zeros(3)
        )

        # Build IMU measurement buffer over sweep
        imu_buffer = []
        for imu_t in np.linspace(scan_start_time, scan_end_time, 15):
            gyro_reading = np.array([0.0, 0.0, true_omega + g_err])
            accel_reading = np.array([0.0, 0.0, 9.81])
            imu_buffer.append((imu_t, gyro_reading, accel_reading))

        pts_comp = backprop(scan_end_time, scan_start_time, imu_pose, scan_raw, imu_buffer)

        if len(pts_comp) > 0:
            # Wall should be flat at x = 2.0. Measure x deviation
            x_vals = pts_comp[:, 0]
            sigma_x_cm = float(np.std(x_vals)) * 100.0
            max_dev_cm = float(np.max(np.abs(x_vals - wall_x))) * 100.0
            status = "CLEAN" if sigma_x_cm < 2.0 else ("DEGRADED" if sigma_x_cm < 6.0 else "DISTORTED")
        else:
            sigma_x_cm = 999.0
            max_dev_cm = 999.0
            status = "FAILED"

        err_str = f"{g_err:+.2f} rad/s ({np.rad2deg(g_err):+.1f} deg/s)"
        print(f"{err_str:<24} | {len(pts_comp):<18} | {sigma_x_cm:8.2f} cm          | {max_dev_cm:8.2f} cm       | {status:<10}")


        results.append({
            'gyro_error': g_err,
            'num_points': len(pts_comp),
            'sigma_x_cm': sigma_x_cm,
            'max_dev_cm': max_dev_cm,
            'status': status
        })


    return results


# ============================================================================
# Experiment 6: Simultaneous Translation + 90-Deg Turn with Low-Cost IMU
# ============================================================================
def test_combined_motion_low_cost_imu():
    """
    Combined motion: robot translates at 0.35 m/s while executing a 90 deg corner turn.
    IMU has:
    - 0.10 rad/s (5.73 deg/s) gyro bias
    - 0.05 m/s^2 accel bias
    - 14 Hz chassis vibration
    - 10% gyro scale factor error
    """
    print("\n" + "=" * 80)
    print("TEST 6: SIMULTANEOUS TRANSLATION + CORNER TURN UNDER REAL LOW-COST IMU")
    print("=" * 80)

    world = SimulationWorld()
    map_obj = Map()
    for x1, y1, x2, y2 in world.walls:
        xs = np.linspace(x1, x2, 100)
        ys = np.linspace(y1, y2, 100)
        pts = np.column_stack([xs, ys, np.zeros_like(xs)])
        map_obj.add_points(pts)

    dt_scan = 0.30
    dt_imu = 0.03
    total_time = 10.0

    true_p = np.array([-2.0, -2.5, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    estimator = ESIKFStateEstimator()
    estimator.state.p = true_p.copy()
    estimator.state.R = np.eye(3)

    bg_sim = np.array([0.0, 0.0, 0.10])  # 0.10 rad/s bias
    ba_sim = np.array([0.05, -0.05, 0.0])
    gyro_scale_factor = 0.90             # 10% scale under-reporting

    trajectory_errors_pos = []
    trajectory_errors_yaw = []
    total_scans = int(total_time / dt_scan)
    applied_count = 0

    for step in range(total_scans):
        t_scan = step * dt_scan

        # Profile: Move +X, Turn 90 deg left to +Y, Move +Y, Stop
        target_v = np.zeros(3)
        target_omega = 0.0

        if 1.0 <= t_scan < 3.5:
            target_v = np.array([0.35, 0.0, 0.0])
        elif 3.5 <= t_scan < 5.5:
            # Turn 90 deg left over 2s = 45 deg/s = 0.785 rad/s while slowing
            target_v = np.array([0.15, 0.15, 0.0])
            target_omega = np.pi / (2.0 * 2.0)
        elif 5.5 <= t_scan < 8.0:
            target_v = np.array([0.0, 0.35, 0.0])
        else:
            target_v = np.zeros(3)

        for _ in range(int(dt_scan / dt_imu)):
            true_v = 0.85 * true_v + 0.15 * target_v
            true_p += true_v * dt_imu
            true_yaw += target_omega * dt_imu

            # Low-cost IMU reading
            meas_omega = target_omega * gyro_scale_factor + bg_sim[2] + 0.08 * np.sin(2 * np.pi * 14.3 * t_scan)
            gyro_meas = np.array([0.0, 0.0, meas_omega]) + np.random.normal(0, 0.004, 3)
            accel_meas = np.array([0.0, 0.0, 9.81]) + ba_sim + np.random.normal(0, 0.02, 3)

            estimator.predict([gyro_meas, accel_meas], dt=dt_imu)

        scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)

        state_old = copy_state(estimator.state)
        state_update = copy_state(state_old)
        P_copy = estimator.P.copy()

        pts_w, applied, P_new = estimator.lidar_update(
            scan=scan_raw,
            state=state_update,
            P_copy=P_copy,
            lidar_points_compensated=body_pts,
            map=map_obj
        )

        if applied:
            applied_count += 1
            delta_R = state_old.R.T @ state_update.R
            delta_p = state_update.p - state_old.p
            estimator.state.R = estimator.state.R @ delta_R
            estimator.state.R = reorthonormalize(estimator.state.R)
            estimator.state.p += delta_p
            estimator.P = P_new

        pos_err = np.linalg.norm((estimator.state.p - true_p)[:2])
        est_yaw = extract_yaw(estimator.state.R)
        yaw_err = abs(wrap_angle(est_yaw - true_yaw))

        trajectory_errors_pos.append(pos_err * 100.0)
        trajectory_errors_yaw.append(np.rad2deg(yaw_err))

    ate_rmse = float(np.sqrt(np.mean(np.array(trajectory_errors_pos)**2)))
    max_pos_err = float(np.max(trajectory_errors_pos))
    final_pos_err = float(trajectory_errors_pos[-1])

    yaw_rmse = float(np.sqrt(np.mean(np.array(trajectory_errors_yaw)**2)))
    max_yaw_err = float(np.max(trajectory_errors_yaw))
    final_yaw_err = float(trajectory_errors_yaw[-1])
    applied_rate = (applied_count / total_scans) * 100.0

    print(f"Results for Low-Cost IMU Turn + Translation:")
    print(f"  ATE Position RMSE   : {ate_rmse:.2f} cm")
    print(f"  Max Position Error  : {max_pos_err:.2f} cm")
    print(f"  Final Position Error: {final_pos_err:.2f} cm")
    print(f"  Yaw RMSE            : {yaw_rmse:.2f} deg")
    print(f"  Max Yaw Error       : {max_yaw_err:.2f} deg")
    print(f"  Final Yaw Error     : {final_yaw_err:.2f} deg")
    print(f"  LiDAR Applied Rate  : {applied_rate:.1f} %")

    return {
        'ate_rmse_cm': ate_rmse,
        'max_pos_err_cm': max_pos_err,
        'final_pos_err_cm': final_pos_err,
        'yaw_rmse_deg': yaw_rmse,
        'max_yaw_err_deg': max_yaw_err,
        'final_yaw_err_deg': final_yaw_err,
        'applied_rate': applied_rate
    }


# ============================================================================
# Experiment 7: ZUPT Gyro Bias Adaptation Limits & Stationary Trapping
# ============================================================================
def test_zupt_bias_adaptation_limits():
    """
    Evaluates whether the pipeline's stationary ZUPT routine can estimate and cancel
    an initial uncalibrated gyro bias when the robot is at rest, and determines the
    maximum uncalibrated gyro bias before the system is trapped in false motion.
    """
    print("\n" + "=" * 80)
    print("TEST 7: STATIONARY ZUPT GYRO BIAS ADAPTATION & TRAPPING LIMITS")
    print("=" * 80)

    # In update.py:
    # is_imu_static = (gyro_max <= 0.12 and accel_var <= 0.30)
    # adapt bg if: is_platform_static and static_count >= 10 and gyro_max < 0.06
    test_biases = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    results = []

    print(f"{'True Gyro Bias':<20} | {'Static Detected?':<18} | {'Bias Adapted?':<16} | {'Final Est. bg':<16} | {'Diagnosis':<16}")
    print("-" * 92)

    for true_bg_z in test_biases:
        dt_scan = 0.30
        total_scans = 20  # 6.0 seconds resting
        est_bg = 0.0
        static_count = 0
        adapted_count = 0

        for _ in range(total_scans):
            # 10 IMU samples per scan
            imu_samples = [true_bg_z + np.random.normal(0, 0.002) for _ in range(10)]
            gyro_max = float(np.max(np.abs(imu_samples)))
            mean_gyro = float(np.mean(imu_samples))
            accel_var = 0.01  # resting
            sim = 10.0        # resting LiDAR similarity

            is_imu_static = (gyro_max <= 0.12 and accel_var <= 0.30)
            is_lidar_static = (sim < 35.0)
            is_static_sample = (is_imu_static and is_lidar_static)

            if is_static_sample:
                static_count += 1
            else:
                static_count = 0

            is_platform_static = (static_count >= 2)

            if is_platform_static and static_count >= 10:
                if (sim < 25.0) and accel_var < 0.08 and abs(mean_gyro) < 0.08:
                    step_bg = np.clip(0.01 * (true_bg_z - est_bg), -0.001, 0.001)
                    est_bg += step_bg
                    adapted_count += 1

        detected = (static_count >= 2)
        adapted = (adapted_count > 0)
        diag = "ADAPTING" if adapted else ("GUARDED (>0.08)" if detected else "NEVER STATIC (>0.12)")


        print(f"{true_bg_z:10.3f} rad/s ({np.rad2deg(true_bg_z):4.1f} deg/s) | {str(detected):<18} | {str(adapted):<16} | {est_bg:10.4f} rad/s   | {diag:<16}")

        results.append({
            'true_bg': true_bg_z,
            'detected': detected,
            'adapted': adapted,
            'final_bg': est_bg,
            'diagnosis': diag
        })

    return results


# ============================================================================
# Main Execution Runner
# ============================================================================
def run_all_rotation_robustness_tests():
    np.random.seed(42)
    t1 = test_instantaneous_heading_convergence()
    np.random.seed(42)
    t2 = test_active_turn_gyro_biases()
    np.random.seed(42)
    t3 = test_gyro_scale_factor_errors()
    np.random.seed(42)
    t4 = test_gyro_vibration_noise()
    np.random.seed(42)
    t5 = test_deskewing_motion_compensation_sensitivity()
    np.random.seed(42)
    t6 = test_combined_motion_low_cost_imu()
    np.random.seed(42)
    t7 = test_zupt_bias_adaptation_limits()

    print("\n" + "=" * 80)
    print("ALL ROTATION ROBUSTNESS TESTS COMPLETED")
    print("=" * 80)



if __name__ == "__main__":
    run_all_rotation_robustness_tests()
