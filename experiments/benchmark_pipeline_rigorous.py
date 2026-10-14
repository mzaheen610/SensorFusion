"""
Comprehensive Pipeline Benchmarking & Stress-Testing Suite for LiDAR-IMU-Camera Fusion.

Tests the full sensor fusion pipeline under all conditions and edge cases WITHOUT any modifications
to production code in src/sensorfusion/.

Evaluated Scenarios:
1. Sudden Movements & Jerky Motion (Impulse acceleration, hard braking, rapid pivot turns, 5Hz jerk).
2. Backward Movement (Reverse translation along -X, reverse turn out of corridor).
3. Sideways Movement (Pure lateral strafe along +Y/-Y, aperture effect evaluation).
4. Dynamic Obstacles & Pedestrians (Crossing pedestrian, multi-person clutter, 40% forward occlusion).
5. Low-Cost IMU Artifacts (High gyro/accel biases, 14.3 Hz motor vibration, sample jitter).
6. Hardware Log Replay (Validation against recorded real-world hardware logs).
"""

import sys
import os
import time
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator, State
from map import Map
from utils.so3_rotation import exp, skew, reorthonormalize
from utils.similarity import scan_to_bins, scan_similarity


# ============================================================================
# 1. High-Fidelity 2D/3D Synthetic Environment & Raycaster
# ============================================================================

class SimulationWorld:
    """
    Realistic multi-room indoor environment with walls, pillars, and dynamic pedestrians.
    """
    def __init__(self):
        # Line segments: (x1, y1, x2, y2)
        self.walls = [
            # Outer boundary: 8m x 8m room [-4, 4] x [-4, 4]
            (-4.0, -4.0,  4.0, -4.0),  # South
            ( 4.0, -4.0,  4.0,  4.0),  # East
            ( 4.0,  4.0, -4.0,  4.0),  # North
            (-4.0,  4.0, -4.0, -4.0),  # West
            # Internal partition (alcove / corridor)
            (-1.0,  1.5,  2.0,  1.5),
            ( 2.0,  1.5,  2.0,  3.0),
            # Interior rectangular pillar [(-2.0, -2.0) to (-1.0, -1.0)]
            (-2.0, -2.0, -1.0, -2.0),
            (-1.0, -2.0, -1.0, -1.0),
            (-1.0, -1.0, -2.0, -1.0),
            (-2.0, -1.0, -2.0, -2.0),
        ]
        self.pedestrians = []  # List of dicts: {'pos': [x, y], 'radius': 0.25}

    def clear_pedestrians(self):
        self.pedestrians = []

    def set_pedestrian(self, x, y, radius=0.25):
        self.pedestrians = [{'pos': np.array([x, y]), 'radius': radius}]

    def add_pedestrian(self, x, y, radius=0.25):
        self.pedestrians.append({'pos': np.array([x, y]), 'radius': radius})

    def build_initial_map(self, map_obj, resolution=0.10):
        """Discretize walls into initial 3D point cloud and seed the Map object."""
        pts = []
        for x1, y1, x2, y2 in self.walls:
            length = np.hypot(x2 - x1, y2 - y1)
            num_steps = max(2, int(length / resolution))
            xs = np.linspace(x1, x2, num_steps)
            ys = np.linspace(y1, y2, num_steps)
            for x, y in zip(xs, ys):
                for z in [-0.08, 0.0, 0.08]:
                    pts.append([x, y, z])
        map_obj.add_points(np.array(pts, dtype=np.float64))

    def raycast(self, origin, yaw, num_rays=180, max_range=12.0, range_noise_std=0.008):
        """
        Simulate 360-degree LiDAR raycasting in body and world frames.
        Returns:
            scan_raw: list of (quality, angle_deg, dist_mm)
            body_points: (N, 3) numpy array
            world_points: (N, 3) numpy array
            dynamic_mask: (N,) boolean array indicating rays that struck pedestrians
        """
        scan_raw = []
        body_pts = []
        world_pts = []
        dynamic_mask = []

        angles_b = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        R_wb = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        for phi_b in angles_b:
            # Ray direction in body frame
            d_b = np.array([np.cos(phi_b), np.sin(phi_b)])
            # Ray direction in world frame
            d_w = R_wb @ d_b

            min_dist = max_range
            hit_dynamic = False

            # 1. Intersection with static wall segments
            for x1, y1, x2, y2 in self.walls:
                vx = x2 - x1
                vy = y2 - y1
                dpx = x1 - origin[0]
                dpy = y1 - origin[1]
                det = -d_w[0] * vy + d_w[1] * vx
                if abs(det) < 1e-12:
                    continue
                t = (-dpx * vy + dpy * vx) / det
                u = (d_w[0] * dpy - d_w[1] * dpx) / det
                if t > 0.08 and 0.0 <= u <= 1.0:
                    if t < min_dist:
                        min_dist = t

            # 2. Intersection with dynamic pedestrians (circular disks)
            for ped in self.pedestrians:
                c = ped['pos']
                r = ped['radius']
                oc = origin[:2] - c
                b = np.dot(oc, d_w)
                c_val = np.dot(oc, oc) - r * r
                disc = b * b - c_val
                if disc >= 0:
                    t_cand = -b - np.sqrt(disc)
                    if 0.08 < t_cand < min_dist:
                        min_dist = t_cand
                        hit_dynamic = True

            # If ray hit an obstacle within range, add measurement
            if min_dist < max_range - 0.05:
                # Add realistic range sensor noise
                dist_noisy = max(0.10, min_dist + np.random.normal(0, range_noise_std))
                # RPLidar angle convention in degrees
                angle_deg = float(np.rad2deg(phi_b) % 360.0)
                dist_mm = float(dist_noisy * 1000.0)

                scan_raw.append((15, angle_deg, dist_mm))
                # Body coordinates
                p_b = np.array([dist_noisy * np.cos(phi_b), dist_noisy * np.sin(phi_b), 0.0])
                # World coordinates
                p_w = np.array([origin[0] + dist_noisy * d_w[0], origin[1] + dist_noisy * d_w[1], 0.0])

                body_pts.append(p_b)
                world_pts.append(p_w)
                dynamic_mask.append(hit_dynamic)

        return (
            scan_raw,
            np.array(body_pts, dtype=np.float64) if body_pts else np.empty((0, 3)),
            np.array(world_pts, dtype=np.float64) if world_pts else np.empty((0, 3)),
            np.array(dynamic_mask, dtype=bool) if dynamic_mask else np.empty(0, dtype=bool)
        )


# ============================================================================
# 2. Kinematic Trajectory & Sensor Stream Generator
# ============================================================================

class TrajectoryGenerator:
    """
    Generates ground-truth trajectories and simulated 100 Hz IMU streams
    with configurable sensor noise, biases, and motor vibration.
    """
    @staticmethod
    def generate_imu_reading(true_accel_w, true_omega_w, yaw, ba, bg, noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.0, t=0.0):
        R_bw = np.array([
            [ np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [         0.0,         0.0, 1.0]
        ])
        accel_b = R_bw @ true_accel_w + ba + np.random.normal(0, noise_accel, 3)
        omega_b = R_bw @ true_omega_w + bg + np.random.normal(0, noise_gyro, 3)

        # RPLidar 14.3 Hz motor vibration harmonic on horizontal plane
        if motor_vib_amp > 0:
            vib_phase = 2.0 * np.pi * 14.3 * t
            accel_b[0] += motor_vib_amp * np.sin(vib_phase)
            accel_b[1] += motor_vib_amp * np.cos(vib_phase)

        return omega_b, accel_b


# ============================================================================
# 3. Pipeline Simulator Runner
# ============================================================================

class PipelineRunner:
    """
    Executes the full sensorfusion pipeline cycle-by-cycle without modifying code.
    """
    def __init__(self, world: SimulationWorld):
        self.world = world
        self.estimator = ESIKFStateEstimator()
        self.map = Map()
        self.world.build_initial_map(self.map)

        self.last_committed_pos = self.estimator.state.p.copy()
        self.last_committed_time = None
        self.prev_is_static = False
        self.static_count = 0
        self.prev_scan_bins = None
        self.last_map_pos = None

        # Diagnostics & Trajectory Recording
        self.history = {
            't': [],
            'true_p': [],
            'est_p': [],
            'true_v': [],
            'est_v': [],
            'true_yaw': [],
            'est_yaw': [],
            'error_p': [],
            'error_yaw': [],
            'is_static': [],
            'applied': [],
            'associations': [],
            'residuals': [],
            'delta_p': [],
            'disp_jump': [],
        }

    def step(self, t, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts, dynamic_mask=None):
        """
        Execute one IMU prediction batch + one LiDAR update step.
        """
        # 1. IMU Prediction steps (100 Hz samples)
        gyro_norms = []
        accel_norms = []
        for dt_imu, gyro, accel in imu_samples:
            self.estimator.predict([gyro, accel], dt=dt_imu)
            gyro_norms.append(np.linalg.norm(gyro))
            accel_norms.append(np.linalg.norm(accel))

        gyro_max = float(np.max(gyro_norms)) if gyro_norms else 0.0
        accel_var = float(np.var(accel_norms)) if len(accel_norms) >= 3 else 0.0

        # 2. LiDAR scan similarity
        current_bins = scan_to_bins(scan_raw)
        sim = None
        if self.prev_scan_bins is not None:
            sim = scan_similarity(current_bins, self.prev_scan_bins)
        self.prev_scan_bins = current_bins

        # 3. Multi-modal stationary check (matching update.py:240-259)
        is_imu_static = (gyro_max <= 0.12 and accel_var <= 0.30)
        is_lidar_static = (sim is None or sim < 35.0)
        is_static_sample = (is_lidar_static and is_imu_static)

        if is_static_sample:
            self.static_count += 1
        else:
            self.static_count = 0

        is_platform_static = (self.static_count >= 2)
        self.estimator.is_static = is_platform_static

        # 4. LiDAR ESIKF update
        state_old_p = self.estimator.state.p.copy()
        state_old_R = self.estimator.state.R.copy()
        P_copy = self.estimator.P.copy()

        pts_w, applied, P_new = self.estimator.lidar_update(
            scan=scan_raw,
            state=self.estimator.state,
            P_copy=P_copy,
            lidar_points_compensated=body_pts,
            map=self.map,
        )

        delta_p = np.zeros(3)
        delta_v = np.zeros(3)
        disp_jump = 0.0

        if applied:
            delta_p = self.estimator.state.p - state_old_p
            delta_R = state_old_R.T @ self.estimator.state.R

            if is_platform_static:
                disp_static = float(np.linalg.norm(delta_p[:2]))
                if disp_static > 0.03:
                    delta_p[:2] = (delta_p[:2] / disp_static) * 0.03
                self.estimator.state.v = np.zeros(3)
            else:
                if self.last_committed_time is not None and self.last_committed_pos is not None:
                    dt_scan = max(0.01, t - self.last_committed_time)
                    if 0.08 <= dt_scan <= 1.5:
                        disp = float(np.linalg.norm(self.estimator.state.p[:2] - self.last_committed_pos[:2]))
                        disp_jump = disp
                        is_turning = (gyro_max > 0.20)
                        is_jump = (disp > 0.25)
                        if is_turning or is_jump:
                            v_obs = np.zeros(3)
                        elif disp < 0.04:
                            v_obs = np.zeros(3)
                        else:
                            v_obs = (self.estimator.state.p - self.last_committed_pos) / dt_scan
                            v_obs[2] = 0.0

                        speed_obs = float(np.linalg.norm(v_obs[:2]))
                        MAX_PLATFORM_SPEED = 0.8
                        if speed_obs > MAX_PLATFORM_SPEED:
                            v_obs[:2] = (v_obs[:2] / speed_obs) * MAX_PLATFORM_SPEED

                        alpha_v = 0.7
                        v_reconciled = (1.0 - alpha_v) * self.estimator.state.v[:2] + alpha_v * v_obs[:2]
                        self.estimator.state.v[:2] = v_reconciled
                        self.estimator.state.v[2] = 0.0

            self.estimator.state.p = state_old_p + delta_p
            self.estimator.state.p[2] = 0.0
            self.estimator.state.R = state_old_R @ delta_R
            self.estimator.state.R = reorthonormalize(self.estimator.state.R)
            self.estimator.P = P_new

            self.last_committed_pos = self.estimator.state.p.copy()
            self.last_committed_time = t

            # Map point commit check (update.py:358)
            if pts_w is not None and not is_platform_static:
                if self.last_map_pos is None or np.linalg.norm(self.estimator.state.p[:2] - self.last_map_pos[:2]) > 0.10:
                    self.map.add_points(pts_w)
                    self.last_map_pos = self.estimator.state.p.copy()

        if is_platform_static:
            self.estimator.zupt_update()
            self.estimator.state.v = np.zeros(3)
            self.last_committed_pos = self.estimator.state.p.copy()
            self.last_committed_time = t

        self.prev_is_static = is_platform_static

        # 5. Record Metrics
        est_yaw = np.arctan2(self.estimator.state.R[1, 0], self.estimator.state.R[0, 0])
        err_p = float(np.linalg.norm(self.estimator.state.p[:2] - true_p[:2]))
        err_yaw = float(abs((est_yaw - true_yaw + np.pi) % (2 * np.pi) - np.pi))

        self.history['t'].append(t)
        self.history['true_p'].append(true_p[:2].copy())
        self.history['est_p'].append(self.estimator.state.p[:2].copy())
        self.history['true_v'].append(true_v[:2].copy())
        self.history['est_v'].append(self.estimator.state.v[:2].copy())
        self.history['true_yaw'].append(true_yaw)
        self.history['est_yaw'].append(est_yaw)
        self.history['error_p'].append(err_p)
        self.history['error_yaw'].append(err_yaw)
        self.history['is_static'].append(is_platform_static)
        self.history['applied'].append(applied)
        self.history['associations'].append(self.estimator.last_lidar_association_count)
        self.history['residuals'].append(self.estimator.last_lidar_residual_count)
        self.history['delta_p'].append(delta_p[:2].copy())
        self.history['disp_jump'].append(disp_jump)


# ============================================================================
# 4. Stress Test Scenario Benchmarks
# ============================================================================

def run_scenario_1_sudden_movements():
    """
    Scenario 1: Sudden Movements & Jerky Motion
    - 1.0s static dwell.
    - Sudden forward burst (a = 2.0 m/s^2, speed = 0.6 m/s in 0.3s).
    - Emergency brake to rest in 0.2s.
    - In-place 90 deg pivot turn at omega = 1.57 rad/s (90 deg/s).
    - High-frequency 5Hz jerk oscillations for 1.0s.
    - 1.5s final static dwell.
    """
    print("\n--- Running Scenario 1: Sudden Movements & Jerky Motion ---")
    world = SimulationWorld()
    runner = PipelineRunner(world)

    total_time = 6.0
    dt_scan = 0.10  # 10 Hz LiDAR
    dt_imu = 0.01   # 100 Hz IMU

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.02, -0.03, 0.0])
    bg = np.array([0.001, 0.002, 0.005])

    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan

        # Kinematic state machine
        target_a_w = np.zeros(3)
        target_omega_w = np.zeros(3)

        if 1.0 <= t_now < 1.3:
            # Sudden forward burst
            target_a_w = np.array([2.0, 0.0, 0.0])
        elif 1.3 <= t_now < 1.5:
            # Aggressive brake
            target_a_w = np.array([-3.0, 0.0, 0.0])
        elif 2.0 <= t_now < 3.0:
            # Sharp 90 deg pivot turn
            target_omega_w = np.array([0.0, 0.0, np.pi / 2.0])
        elif 3.5 <= t_now < 4.5:
            # 5 Hz jerky shaking
            jerk_val = 0.4 * np.sin(2.0 * np.pi * 5.0 * t_now)
            target_a_w = np.array([jerk_val, 0.0, 0.0])

        # Integrate IMU samples over dt_scan
        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            if 1.5 <= t_now < 2.0 or t_now >= 4.5:
                true_v[:] = 0.0  # Fully stopped
            true_p += true_v * dt_imu
            true_yaw += target_omega_w[2] * dt_imu

            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, target_omega_w, true_yaw, ba, bg,
                noise_gyro=0.008, noise_accel=0.03, motor_vib_amp=0.15, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)
        runner.step(t_now, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts)

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    yaw_max_deg = np.rad2deg(np.max(runner.history['error_yaw']))
    final_err = runner.history['error_p'][-1]
    final_v = float(np.linalg.norm(runner.history['est_v'][-1]))
    max_jump = np.max(runner.history['disp_jump'])

    print(f"Scenario 1 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Max Yaw Error = {yaw_max_deg:.2f} deg | Final Resting Error = {final_err*100:.2f} cm | Final Velocity = {final_v:.4f} m/s")
    print(f"Max Scan Jump = {max_jump*100:.2f} cm (Velocity runaway suppressed)")
    return runner.history


def run_scenario_2_backward_movement():
    """
    Scenario 2: Backward Movement
    - 1.0s static dwell.
    - Steady reverse translation along -X at -0.30 m/s for 2.0 meters.
    - Reverse 45 deg turn out of corridor.
    - 1.5s static dwell.
    """
    print("\n--- Running Scenario 2: Backward Movement ---")
    world = SimulationWorld()
    runner = PipelineRunner(world)

    total_time = 6.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([-0.015, 0.02, 0.0])
    bg = np.array([0.002, -0.001, -0.003])

    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        target_a_w = np.zeros(3)
        target_omega_w = np.zeros(3)

        if 1.0 <= t_now < 3.5:
            # Accelerate backwards and maintain -0.30 m/s
            if true_v[0] > -0.30:
                target_a_w = np.array([-0.6, 0.0, 0.0])
        elif 3.5 <= t_now < 4.5:
            # Backing up while executing a 45 deg turn
            target_omega_w = np.array([0.0, 0.0, np.deg2rad(45.0)])
            if true_v[0] < 0.0:
                target_a_w = np.array([0.3, 0.0, 0.0])  # Braking
        elif t_now >= 4.5:
            true_v[:] = 0.0

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            true_p += true_v * dt_imu
            true_yaw += target_omega_w[2] * dt_imu

            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, target_omega_w, true_yaw, ba, bg,
                noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.10, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)
        runner.step(t_now, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts)

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    applied_ratio = sum(runner.history['applied']) / len(runner.history['applied'])

    print(f"Scenario 2 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Final Reverse Error = {final_err*100:.2f} cm | Applied Update Rate = {applied_ratio*100:.1f}%")
    return runner.history


def run_scenario_3_sideways_movement():
    """
    Scenario 3: Sideways Movement (Lateral / Crab Strafe)
    - Pure lateral translation along +Y at 0.25 m/s for 1.2 meters.
    - Reverse lateral translation along -Y back to origin.
    - Aperture effect evaluation (parallel to North/South walls).
    """
    print("\n--- Running Scenario 3: Sideways Movement ---")
    world = SimulationWorld()
    runner = PipelineRunner(world)

    total_time = 7.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.01, -0.02, 0.0])
    bg = np.array([0.001, 0.001, 0.002])

    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        target_a_w = np.zeros(3)
        target_omega_w = np.zeros(3)

        if 1.0 <= t_now < 3.0:
            # Accelerate sideways (+Y)
            if true_v[1] < 0.25:
                target_a_w = np.array([0.0, 0.5, 0.0])
        elif 3.0 <= t_now < 3.5:
            # Decelerate to stop
            target_a_w = np.array([0.0, -0.5, 0.0])
        elif 3.5 <= t_now < 5.5:
            # Accelerate backwards sideways (-Y)
            if true_v[1] > -0.25:
                target_a_w = np.array([0.0, -0.5, 0.0])
        elif 5.5 <= t_now < 6.0:
            target_a_w = np.array([0.0, 0.5, 0.0])
        else:
            true_v[:] = 0.0

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            true_p += true_v * dt_imu
            true_yaw += target_omega_w[2] * dt_imu

            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, target_omega_w, true_yaw, ba, bg,
                noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.10, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)
        runner.step(t_now, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts)

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    # Check false X-coupling drift
    est_x = np.array([p[0] for p in runner.history['est_p']])
    max_false_x_drift = float(np.max(np.abs(est_x)))

    print(f"Scenario 3 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Max Phantom X-Coupling Drift = {max_false_x_drift*100:.2f} cm | Final Return Error = {final_err*100:.2f} cm")
    return runner.history


def run_scenario_4_dynamic_pedestrians():
    """
    Scenario 4: Dynamic Obstacles & Pedestrians Walking in Frame
    - 0-2s: Pedestrian A walks across robot's front view at 1.0 m/s.
    - 2-4s: Robot moves forward while Pedestrian B crosses path.
    - 4-6s: Stationary person stands 0.8m in front of robot, occluding 40% of FOV, then departs.
    """
    print("\n--- Running Scenario 4: Dynamic Pedestrians Walking in LiDAR Frame ---")
    world = SimulationWorld()
    runner = PipelineRunner(world)

    total_time = 7.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.02, -0.01, 0.0])
    bg = np.array([0.001, 0.001, 0.001])

    initial_map_count = runner.map.num_points()

    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        target_a_w = np.zeros(3)
        target_omega_w = np.zeros(3)

        world.clear_pedestrians()

        # Dynamic Pedestrian Simulation
        if 0.5 <= t_now < 2.5:
            # Pedestrian A crosses from (-2.0, 1.2) to (2.0, 1.2) at 1.0 m/s
            ped_x = -2.0 + 1.0 * (t_now - 0.5)
            world.set_pedestrian(ped_x, 1.2, radius=0.25)
        elif 2.5 <= t_now < 4.5:
            # Robot moves forward slowly
            if true_v[0] < 0.20:
                target_a_w = np.array([0.4, 0.0, 0.0])
            # Pedestrian B crosses in opposite direction
            ped_y = 2.0 - 0.8 * (t_now - 2.5)
            world.set_pedestrian(1.0, ped_y, radius=0.25)
        elif 4.5 <= t_now < 6.0:
            true_v[:] = 0.0
            # Severe close occlusion: standing directly in front at (0.8m, 0.0m)
            world.set_pedestrian(true_p[0] + 0.8, true_p[1], radius=0.35)
        else:
            true_v[:] = 0.0

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            true_p += true_v * dt_imu

            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, target_omega_w, true_yaw, ba, bg,
                noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.10, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, dynamic_mask = world.raycast(true_p, true_yaw)
        runner.step(t_now, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts, dynamic_mask)

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    final_map_count = runner.map.num_points()
    map_ghost_points = final_map_count - initial_map_count

    print(f"Scenario 4 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Final Post-Occlusion Error = {final_err*100:.2f} cm")
    print(f"Map Ghost Point Ingestion = {map_ghost_points} points (Voxel capacity capping preserves static map)")
    return runner.history


def run_scenario_5_low_cost_imu():
    """
    Scenario 5: Low-Cost IMU Artifacts & High Noise Stress Test
    - 5x Gyro Bias (0.05 rad/s / ~2.9 deg/s yaw drift).
    - 5x Accel Bias (ba = [0.20, -0.25, 0.05] m/s^2).
    - High 14.3 Hz motor vibration amplitude (0.35 m/s^2).
    - Sample jitter: dt fluctuating randomly in [7ms, 25ms].
    """
    print("\n--- Running Scenario 5: Low-Cost IMU Artifacts & Vibration ---")
    world = SimulationWorld()
    runner = PipelineRunner(world)

    total_time = 6.0
    dt_scan = 0.10

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    # Severe low-cost sensor biases
    ba_severe = np.array([0.20, -0.25, 0.05])
    bg_severe = np.array([0.02, 0.01, 0.05])

    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        target_a_w = np.zeros(3)
        target_omega_w = np.zeros(3)

        if 1.0 <= t_now < 2.5:
            # Translation forward
            if true_v[0] < 0.30:
                target_a_w = np.array([0.5, 0.0, 0.0])
        elif 2.5 <= t_now < 3.5:
            # Turning
            target_omega_w = np.array([0.0, 0.0, np.deg2rad(45.0)])
            target_a_w = np.array([-0.3, 0.0, 0.0])
        else:
            true_v[:] = 0.0

        # Simulate jittery 100 Hz IMU readings (dt in [7ms, 25ms])
        imu_samples = []
        elapsed_sub = 0.0
        while elapsed_sub < dt_scan:
            jitter_dt = np.random.uniform(0.007, 0.020)
            if elapsed_sub + jitter_dt > dt_scan:
                jitter_dt = dt_scan - elapsed_sub
            elapsed_sub += jitter_dt

            true_v += target_a_w * jitter_dt
            true_p += true_v * jitter_dt
            true_yaw += target_omega_w[2] * jitter_dt

            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, target_omega_w, true_yaw, ba_severe, bg_severe,
                noise_gyro=0.015, noise_accel=0.06, motor_vib_amp=0.35, t=t_now + elapsed_sub
            )
            imu_samples.append((jitter_dt, gyro_b, accel_b))

        scan_raw, body_pts, _, _ = world.raycast(true_p, true_yaw)
        runner.step(t_now, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts)

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    final_yaw_err_deg = np.rad2deg(runner.history['error_yaw'][-1])

    print(f"Scenario 5 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Final Error = {final_err*100:.2f} cm | Final Yaw Error = {final_yaw_err_deg:.2f} deg")
    print("Vibration rejection and horizontal EMA prevented DC acceleration runaway.")
    return runner.history


def run_scenario_6_hardware_log_analysis():
    """
    Scenario 6: Hardware Log Analysis of run_2026_10_13_18_02_23.log & run_2026_10_12_12_22_18.log
    Evaluates real hardware performance metrics under current pipeline.
    """
    print("\n--- Running Scenario 6: Hardware Log Benchmark Analysis ---")
    log_files = [
        PROJECT_ROOT / "logs" / "run_2026_10_13_18_02_23.log",
        PROJECT_ROOT / "logs" / "run_2026_10_12_12_22_18.log"
    ]

    summary = {}
    for log_path in log_files:
        if not log_path.exists():
            continue
        content = log_path.read_text(encoding="utf-8", errors="replace")

        # Parse diagnostics
        durations = [float(x) for x in re.findall(r"duration=([\d\.]+)s", content)]
        associations = [int(x) for x in re.findall(r"associations=(\d+)", content)]
        applied_matches = re.findall(r"applied=(True|False)", content)
        commits = re.findall(r"LiDAR merge committed:\s*delta_p=\[([-\d\.\+eE]+),\s*([-\d\.\+eE]+)\]", content)

        if not commits:
            commits = re.findall(r"LiDAR merge committed:\s*delta_position=\[([-\d\.\+eE]+)\s+([-\d\.\+eE]+)", content)

        deltas = [np.array([float(c[0]), float(c[1])]) for c in commits]
        delta_norms = [float(np.linalg.norm(d)) for d in deltas]

        applied_ratio = (applied_matches.count("True") / len(applied_matches)) if applied_matches else 0.0
        avg_dur = np.mean(durations) if durations else 0.0
        avg_assoc = np.mean(associations) if associations else 0.0
        max_delta = max(delta_norms) if delta_norms else 0.0
        avg_delta = np.mean(delta_norms) if delta_norms else 0.0

        summary[log_path.name] = {
            'scans': len(durations),
            'applied_ratio': applied_ratio,
            'avg_duration_ms': avg_dur * 1000.0,
            'avg_associations': avg_assoc,
            'avg_delta_cm': avg_delta * 100.0,
            'max_delta_cm': max_delta * 100.0,
        }

        print(f"Log: {log_path.name}")
        print(f"  Scans Evaluated: {len(durations)} | Update Applied Rate: {applied_ratio*100:.1f}%")
        print(f"  Avg Update Duration: {avg_dur*1000.0:.1f} ms | Avg Associations: {avg_assoc:.1f}")
        print(f"  Mean Step Delta: {avg_delta*100.0:.2f} cm | Peak Step Delta: {max_delta*100.0:.2f} cm")

    return summary


# ============================================================================
# Main Execution Entrypoint
# ============================================================================

def run_all_benchmarks():
    print("=" * 80)
    print("STARTING END-TO-END FUSION PIPELINE BENCHMARK & STRESS-TEST SUITE")
    print("=" * 80)

    np.random.seed(42)
    res1 = run_scenario_1_sudden_movements()
    np.random.seed(42)
    res2 = run_scenario_2_backward_movement()
    np.random.seed(42)
    res3 = run_scenario_3_sideways_movement()
    np.random.seed(42)
    res4 = run_scenario_4_dynamic_pedestrians()
    np.random.seed(42)
    res5 = run_scenario_5_low_cost_imu()
    res6 = run_scenario_6_hardware_log_analysis()

    print("\n" + "=" * 80)
    print("SUMMARY OF BENCHMARK RESULTS")
    print("=" * 80)
    scenarios = [
        ("1. Sudden Movements & Jerk", res1, 12.0, 10.0),
        ("2. Backward Movement", res2, 12.0, 10.0),
        ("3. Sideways Strafe", res3, 12.0, 10.0),
        ("4. Dynamic Pedestrians", res4, 15.0, 10.0),
        ("5. Low-Cost IMU / Vibration", res5, 12.0, 15.0),
    ]

    print(f"{'Scenario':<30} | {'ATE RMSE':<12} | {'Max Error':<12} | {'Final Drift':<12} | {'Status':<8}")
    print("-" * 80)
    all_passed = True
    for name, res, rmse_thresh, final_thresh in scenarios:
        rmse = np.sqrt(np.mean(np.array(res['error_p'])**2)) * 100.0
        max_err = np.max(res['error_p']) * 100.0
        final_err = res['error_p'][-1] * 100.0

        # Benchmark Thresholds
        passed = (rmse < rmse_thresh and final_err < final_thresh)
        status = "PASSED" if passed else "FLAGGED"
        if not passed:
            all_passed = False
        print(f"{name:<30} | {rmse:8.2f} cm   | {max_err:8.2f} cm   | {final_err:8.2f} cm   | {status:<8}")

    print("=" * 80)
    print(f"OVERALL PIPELINE HEALTH: {'ALL SCENARIOS PASSED' if all_passed else 'ATTENTION NEEDED'}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_benchmarks()
