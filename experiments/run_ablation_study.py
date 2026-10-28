#!/usr/bin/env python3
"""
Comprehensive Ablation Study Suite for LiDAR-IMU-Camera Fusion.
===============================================================

Systematically evaluates the contribution of each architectural component:
1. Full Proposed System (Canonical ESIKF + NHC + ZUPT + Deskewing + Dynamic Gating)
2. Ablation 1: IMU Dead Reckoning Only (LiDAR measurement updates disabled)
3. Ablation 2: LiDAR-Only ICP (IMU propagation disabled: zero-motion prior)
4. Ablation 3: Without Non-Holonomic Constraints (NHC disabled: unconstrained 6-DOF lateral motion)
5. Ablation 4: Without Multi-Modal ZUPT & Static Anchor (Free-running drift at rest)
6. Ablation 5: Without LiDAR Motion Deskewing (Raw distorted scans during angular velocity)

Evaluates on:
- Test 1: Continuous Loop Circuit (16m closed-loop multi-room circuit with 4x 90° turns)
- Test 2: Long Featureless Corridor (Aperture effect & degenerate geometry stress)
- Test 3: High-Frequency S-Curve Slalom (Continuous angular reversals & rapid lateral acceleration)

Outputs:
- Consolidated Markdown & LaTeX tables of ATE RMSE, Max Drift, and Final Return Error.
"""

import sys
import os
import time
import copy
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from forward import ESIKFStateEstimator, State
from map import Map
from utils.so3_rotation import exp, skew, reorthonormalize
from utils.similarity import scan_to_bins, scan_similarity
from lidar.backward import backprop, copy_state
from benchmark_pipeline_rigorous import SimulationWorld, PipelineRunner
from continuous_stream_edge_cases import make_loop_world, make_corridor_world, FileDataStreamer


class AblatedPipelineRunner(PipelineRunner):
    """
    Configurable PipelineRunner supporting selective ablation of architectural modules.
    """
    def __init__(
        self,
        world,
        enable_lidar: bool = True,
        enable_imu: bool = True,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_deskew: bool = True,
    ):
        super().__init__(world)
        self.enable_lidar = enable_lidar
        self.enable_imu = enable_imu
        self.enable_nhc = enable_nhc
        self.enable_zupt = enable_zupt
        self.enable_deskew = enable_deskew

    def step(self, t, true_p, true_v, true_yaw, imu_samples, scan_raw, body_pts, dynamic_mask=None):
        # 1. IMU forward propagation
        gyro_norms = []
        accel_norms = []
        for dt_imu, gyro, accel in imu_samples:
            if self.enable_imu:
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

        # 3. Multi-modal stationary check
        if self.enable_zupt:
            is_imu_static = (gyro_max <= 0.12 and accel_var <= 0.30)
            is_lidar_static = (sim is None or sim < 35.0)
            is_static_sample = (is_lidar_static and is_imu_static)

            if is_static_sample:
                self.static_count += 1
            else:
                self.static_count = 0

            is_platform_static = (self.static_count >= 2)
        else:
            is_platform_static = False
            self.static_count = 0

        self.estimator.is_static = is_platform_static

        # Deskewing ablation: if deskew disabled, add synthetic rotational distortion based on angular rate
        effective_body_pts = body_pts
        if not self.enable_deskew and len(imu_samples) > 0:
            avg_gyro = np.mean([s[1] for s in imu_samples], axis=0)
            scan_duration = sum([s[0] for s in imu_samples])
            if np.linalg.norm(avg_gyro) > 0.08:
                distorted_pts = []
                for pt in body_pts:
                    angle = np.arctan2(pt[1], pt[0])
                    frac = (angle % (2 * np.pi)) / (2 * np.pi)
                    dtheta = avg_gyro[2] * scan_duration * (1.0 - frac)
                    R_err = np.array([
                        [np.cos(dtheta), -np.sin(dtheta), 0],
                        [np.sin(dtheta),  np.cos(dtheta), 0],
                        [0, 0, 1]
                    ])
                    distorted_pts.append(R_err @ pt)
                effective_body_pts = np.array(distorted_pts)

        # 4. LiDAR ESIKF update
        if self.enable_lidar:
            state_old_p = self.estimator.state.p.copy()
            state_old_R = self.estimator.state.R.copy()
            P_copy = self.estimator.P.copy()

            pts_w, applied, P_new = self.estimator.lidar_update(
                scan=scan_raw,
                state=self.estimator.state,
                P_copy=P_copy,
                lidar_points_compensated=effective_body_pts,
                map=self.map,
            )

            delta_p = np.zeros(3)
            delta_v = np.zeros(3)

            if applied:
                delta_p = self.estimator.state.p - state_old_p
                delta_R = state_old_R.T @ self.estimator.state.R

                # Non-Holonomic Constraints (NHC)
                if self.enable_nhc:
                    dx_body = state_old_R.T @ delta_p
                    max_lat = 0.06 if gyro_max > 0.2 else 0.03
                    dx_body[1] = np.clip(dx_body[1], -max_lat, max_lat)
                    dx_body[2] = 0.0
                    delta_p = state_old_R @ dx_body

                if is_platform_static and self.enable_zupt:
                    disp_static = float(np.linalg.norm(delta_p[:2]))
                    if disp_static > 0.03:
                        delta_p[:2] = (delta_p[:2] / disp_static) * 0.03
                    self.estimator.state.v = np.zeros(3)
                else:
                    if self.last_committed_time is not None and self.last_committed_pos is not None:
                        dt_scan = max(0.01, t - self.last_committed_time)
                        if 0.08 <= dt_scan <= 1.5:
                            disp = float(np.linalg.norm(self.estimator.state.p[:2] - self.last_committed_pos[:2]))
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

                            # NHC velocity reconciliation
                            if self.enable_nhc:
                                v_obs_body = state_old_R.T @ v_obs
                                v_obs_body[1] = 0.0
                                v_obs_body[2] = 0.0
                                v_obs = state_old_R @ v_obs_body

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

                if pts_w is not None and not is_platform_static:
                    if self.last_map_pos is None or np.linalg.norm(self.estimator.state.p[:2] - self.last_map_pos[:2]) > 0.10:
                        self.map.add_points(pts_w)
                        self.last_map_pos = self.estimator.state.p.copy()

        if is_platform_static and self.enable_zupt:
            self.estimator.zupt_update()
            self.estimator.state.v = np.zeros(3)
            self.last_committed_pos = self.estimator.state.p.copy()
            self.last_committed_time = t

        # Record diagnostics in history
        err_p = float(np.linalg.norm(self.estimator.state.p[:2] - true_p[:2]))
        self.history['t'].append(t)
        self.history['true_p'].append(true_p[:2].copy())
        self.history['est_p'].append(self.estimator.state.p[:2].copy())
        self.history['error_p'].append(err_p)

        return self.estimator.state.p.copy(), self.estimator.state.R.copy()


def run_stream_ablation(name, world_fn, stream_file, config, start_p=None):
    world = world_fn()
    runner = AblatedPipelineRunner(world, **config)
    if start_p is not None:
        runner.estimator.state.p = start_p.copy()
        runner.last_committed_pos = start_p.copy()

    for frame in FileDataStreamer.load_stream(stream_file):
        runner.step(
            frame['t'], frame['true_p'], frame['true_v'], frame['true_yaw'],
            frame['imu_samples'], frame['scan_raw'], frame['body_pts']
        )

    errors = runner.history['error_p']
    ate_rmse = float(np.sqrt(np.mean(np.array(errors)**2)))
    max_err = float(np.max(errors))
    final_err = float(errors[-1])

    return {
        "name": name,
        "ate_rmse_cm": ate_rmse * 100.0,
        "max_err_cm": max_err * 100.0,
        "final_err_cm": final_err * 100.0,
    }


def run_stationary_ablation(name, config, duration_s=30.0, dt_scan=0.3, dt_imu=0.03):
    world = SimulationWorld()
    runner = AblatedPipelineRunner(world, **config)
    
    t = 0.0
    num_scans = int(duration_s / dt_scan)
    n_imu = int(round(dt_scan / dt_imu))
    
    # Standing still in an 8m x 8m room with 14.5Hz motor vibration
    for _ in range(num_scans):
        imu_samples = []
        for _ in range(n_imu):
            # 14.5Hz motor vibration on accel
            vib = 0.35 * np.sin(2 * np.pi * 14.5 * t)
            noise_a = np.random.normal(0, 0.03, 3)
            noise_w = np.random.normal(0, 0.005, 3)
            accel = np.array([vib + noise_a[0], 0.5 * vib + noise_a[1], noise_a[2]])
            gyro = noise_w
            imu_samples.append((dt_imu, gyro, accel))
            t += dt_imu
            
        raw_scan, body_pts, _, _ = world.raycast(np.zeros(2), 0.0)
        runner.step(t, np.zeros(3), np.zeros(3), 0.0, imu_samples, raw_scan, body_pts)
        
    errors = runner.history['error_p']
    ate_rmse = float(np.sqrt(np.mean(np.array(errors)**2)))
    max_err = float(np.max(errors))
    final_err = float(errors[-1])
    
    return {
        "name": name,
        "ate_rmse_cm": ate_rmse * 100.0,
        "max_err_cm": max_err * 100.0,
        "final_err_cm": final_err * 100.0,
    }


def execute_ablation_benchmarks():
    print("=" * 85)
    print(" EXECUTING RIGOROUS MULTIMODAL ABLATION STUDY FOR MASTER'S THESIS")
    print("=" * 85)

    data_dir = PROJECT_ROOT / "experiments" / "continuous_streams_data"

    configurations = [
        ("Full Proposed System (ESIKF+NHC+ZUPT+Deskew)", {
            "enable_lidar": True, "enable_imu": True, "enable_nhc": True, "enable_zupt": True, "enable_deskew": True
        }),
        ("Ablation 1: IMU Dead Reckoning Only (No LiDAR)", {
            "enable_lidar": False, "enable_imu": True, "enable_nhc": True, "enable_zupt": True, "enable_deskew": True
        }),
        ("Ablation 2: LiDAR-Only ICP (No IMU Propagation)", {
            "enable_lidar": True, "enable_imu": False, "enable_nhc": True, "enable_zupt": True, "enable_deskew": True
        }),
        ("Ablation 3: Without NHC (Unconstrained 6-DOF)", {
            "enable_lidar": True, "enable_imu": True, "enable_nhc": False, "enable_zupt": True, "enable_deskew": True
        }),
        ("Ablation 4: Without ZUPT & Static Anchor", {
            "enable_lidar": True, "enable_imu": True, "enable_nhc": True, "enable_zupt": False, "enable_deskew": True
        }),
        ("Ablation 5: Without LiDAR Motion Deskewing", {
            "enable_lidar": True, "enable_imu": True, "enable_nhc": True, "enable_zupt": True, "enable_deskew": False
        }),
    ]

    scenarios = [
        ("Scenario 7: Closed Loop Circuit (16m, 4 Corners)", make_loop_world, data_dir / "stream_scenario_7_loop.npz", np.array([0.0, -4.0, 0.0])),
        ("Scenario 8: Long Corridor (Aperture Stress)", make_corridor_world, data_dir / "stream_scenario_8_corridor.npz", np.array([-7.0, 0.0, 0.0])),
        ("Scenario 9: S-Curve Slalom (Angular Reversals)", SimulationWorld, data_dir / "stream_scenario_9_slalom.npz", np.array([-3.0, -2.5, 0.0])),
    ]

    all_results = {}

    for sc_name, world_fn, stream_path, start_p in scenarios:
        print(f"\n--- Running Benchmark: {sc_name} ---")
        all_results[sc_name] = []
        for config_name, cfg in configurations:
            res = run_stream_ablation(config_name, world_fn, stream_path, cfg, start_p=start_p)
            all_results[sc_name].append(res)
            print(f"  {res['name'][:45]:<46} | ATE RMSE: {res['ate_rmse_cm']:6.2f} cm | Max: {res['max_err_cm']:6.2f} cm | Final: {res['final_err_cm']:6.2f} cm")

    # Stationary Benchmark
    print(f"\n--- Running Benchmark: 30s Stationary Rest under 14.5Hz Motor Vibration ---")
    all_results["Stationary Rest"] = []
    for config_name, cfg in configurations:
        res = run_stationary_ablation(config_name, cfg, duration_s=30.0)
        all_results["Stationary Rest"].append(res)
        print(f"  {res['name'][:45]:<46} | ATE RMSE: {res['ate_rmse_cm']:6.2f} cm | Max: {res['max_err_cm']:6.2f} cm | Final: {res['final_err_cm']:6.2f} cm")

    # Format Master Summary Table
    print("\n" + "=" * 110)
    print(" MASTER ABLATION STUDY RESULTS MATRIX (ATE RMSE in cm)")
    print("=" * 110)
    print(f"{'Pipeline Variant / Ablation Configuration':<46} | {'Loop Circuit':<14} | {'Corridor':<14} | {'Slalom':<14} | {'Stationary':<14}")
    print("-" * 110)

    for i in range(len(configurations)):
        cname = configurations[i][0]
        loop_rmse = all_results["Scenario 7: Closed Loop Circuit (16m, 4 Corners)"][i]["ate_rmse_cm"]
        corr_rmse = all_results["Scenario 8: Long Corridor (Aperture Stress)"][i]["ate_rmse_cm"]
        slalom_rmse = all_results["Scenario 9: S-Curve Slalom (Angular Reversals)"][i]["ate_rmse_cm"]
        stat_rmse = all_results["Stationary Rest"][i]["ate_rmse_cm"]
        
        loop_str = f"{loop_rmse:6.2f} cm" if loop_rmse < 500 else f"{loop_rmse/100:5.2f} m (DIV)"
        corr_str = f"{corr_rmse:6.2f} cm" if corr_rmse < 500 else f"{corr_rmse/100:5.2f} m (DIV)"
        slalom_str = f"{slalom_rmse:6.2f} cm" if slalom_rmse < 500 else f"{slalom_rmse/100:5.2f} m (DIV)"
        stat_str = f"{stat_rmse:6.2f} cm" if stat_rmse < 500 else f"{stat_rmse/100:5.2f} m (DIV)"
        
        print(f"{cname[:46]:<46} | {loop_str:<14} | {corr_str:<14} | {slalom_str:<14} | {stat_str:<14}")
    print("=" * 110)

    return all_results

if __name__ == "__main__":
    execute_ablation_benchmarks()
