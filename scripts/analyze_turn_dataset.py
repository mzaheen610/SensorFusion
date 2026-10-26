#!/usr/bin/env python3
"""
Turn Dataset Analyzer and Kinematic Diagnostic Tool
===================================================

Analyzes recorded turn datasets (.npz) to inspect turning dynamics, verify 
gyro/accel physical consistency, compute lever-arm offsets, detect bias shifts, 
and optionally simulate ESIKF state estimation offline.

Usage:
  # Analyze a recorded dataset and display diagnostic report:
  python3 scripts/analyze_turn_dataset.py data/turns/turn_pivot_left_90_*.npz

  # Replay dataset through ESIKFStateEstimator to evaluate fusion performance:
  python3 scripts/analyze_turn_dataset.py data/turns/turn_pivot_left_90_*.npz --simulate-fusion

  # Export high-resolution analysis plots:
  python3 scripts/analyze_turn_dataset.py data/turns/turn_pivot_left_90_*.npz --save-plot
"""

import argparse
import copy
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

try:
    from forward import ESIKFStateEstimator, State, reorthonormalize
    from map import Map
    from lidar.backward import backprop
    from utils.similarity import scan_to_bins, scan_similarity
    from utils.so3_rotation import log
    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False


def analyze_dataset(
    npz_path: str,
    simulate_fusion: bool = False,
    save_plot: bool = False,
    invert_gyro_z: bool = False,
    clean_spikes: bool = False,
):
    path = Path(npz_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    print("=" * 70)
    print(f" TURN DATASET ANALYSIS: {path.name}")
    print("=" * 70)

    data = np.load(path, allow_pickle=True)

    # Extract arrays
    t_imu = data["imu_t_mono"]
    t_wall = data["imu_t_wall"] if "imu_t_wall" in data else t_imu
    gyro = data["imu_gyro"].copy()  # (M, 3) in rad/s
    lin_accel = data["imu_lin_accel"].copy()  # (M, 3) in m/s^2
    raw_accel = data["imu_raw_accel"].copy()  # (M, 3) in m/s^2
    t_start_turn = float(data["turn_start_time"])
    t_end_turn = float(data["turn_end_time"])

    t_lidar_start = data["lidar_t_start"] if "lidar_t_start" in data else None
    t_lidar_end = data["lidar_t_end"]
    lidar_points = data["lidar_points"]
    lidar_scan_offsets = data["lidar_scan_offsets"]
    num_scans = len(lidar_scan_offsets) - 1

    t0 = t_imu[0]
    t_rel = t_imu - t0
    t_start_rel = t_start_turn - t0
    t_end_rel = t_end_turn - t0
    total_dur = t_imu[-1] - t0
    turn_dur = t_end_turn - t_start_turn

    # Segment phases
    mask_pre = t_imu < t_start_turn
    mask_turn = (t_imu >= t_start_turn) & (t_imu <= t_end_turn)
    mask_post = t_imu > t_end_turn

    # 1. Basic Stats
    print(f"\n[1] ACQUISITION TELEMETRY:")
    print(f"  Total Duration:     {total_dur:.2f} s")
    print(f"  Pre-Turn Dwell:     {np.sum(mask_pre) * (total_dur / len(t_imu)):.2f} s ({np.sum(mask_pre)} samples)")
    print(f"  Active Turn Window: {turn_dur:.2f} s ({np.sum(mask_turn)} samples)")
    print(f"  Post-Turn Dwell:    {np.sum(mask_post) * (total_dur / len(t_imu)):.2f} s ({np.sum(mask_post)} samples)")
    print(f"  IMU Effective Rate: {len(t_imu) / total_dur:.1f} Hz")
    print(f"  LiDAR Scans:        {num_scans} scans ({num_scans / total_dur:.1f} Hz, {len(lidar_points)} total points)")

    # 2. Kinematic Metrics
    dt_arr = np.gradient(t_imu)
    yaw_rate_deg = np.degrees(gyro[:, 2])
    yaw_integrated_deg = np.cumsum(gyro[:, 2] * dt_arr) * (180.0 / np.pi)

    peak_wz_deg = float(np.max(np.abs(yaw_rate_deg)))
    peak_wz_dir = "CCW (Left)" if np.max(yaw_rate_deg) > abs(np.min(yaw_rate_deg)) else "CW (Right)"
    net_turn_deg = float(yaw_integrated_deg[mask_turn][-1] - yaw_integrated_deg[mask_turn][0]) if np.any(mask_turn) else 0.0

    print(f"\n[2] TURNING KINEMATICS:")
    print(f"  Peak Yaw Rate:       {peak_wz_deg:.1f} deg/s ({peak_wz_deg * np.pi / 180.0:.2f} rad/s) [{peak_wz_dir}]")
    print(f"  Net Integrated Yaw:  {net_turn_deg:+.1f} deg ({net_turn_deg * np.pi / 180.0:+.2f} rad)")

    # 3. Bias & Stability Analysis (Pre vs Post Dwell)
    if np.sum(mask_pre) > 5 and np.sum(mask_post) > 5:
        pre_gyro_bias = np.mean(gyro[mask_pre], axis=0)
        post_gyro_bias = np.mean(gyro[mask_post], axis=0)
        delta_gyro_bias = post_gyro_bias - pre_gyro_bias

        pre_accel_bias = np.mean(lin_accel[mask_pre], axis=0)
        post_accel_bias = np.mean(lin_accel[mask_post], axis=0)
        delta_accel_bias = post_accel_bias - pre_accel_bias

        print(f"\n[3] SENSOR BIAS STABILITY (Pre-Turn vs Post-Turn):")
        print(f"  Pre-Turn Gyro Bias:   [{pre_gyro_bias[0]:+.4f}, {pre_gyro_bias[1]:+.4f}, {pre_gyro_bias[2]:+.4f}] rad/s")
        print(f"  Post-Turn Gyro Bias:  [{post_gyro_bias[0]:+.4f}, {post_gyro_bias[1]:+.4f}, {post_gyro_bias[2]:+.4f}] rad/s")
        print(f"  Gyro Bias Shift:      [{delta_gyro_bias[0]:+.4f}, {delta_gyro_bias[1]:+.4f}, {delta_gyro_bias[2]:+.4f}] rad/s")
        print(f"  Pre-Turn Accel Mean:  [{pre_accel_bias[0]:+.4f}, {pre_accel_bias[1]:+.4f}, {pre_accel_bias[2]:+.4f}] m/s²")
        print(f"  Post-Turn Accel Mean: [{post_accel_bias[0]:+.4f}, {post_accel_bias[1]:+.4f}, {post_accel_bias[2]:+.4f}] m/s²")
        print(f"  Accel Bias Shift:     [{delta_accel_bias[0]:+.4f}, {delta_accel_bias[1]:+.4f}, {delta_accel_bias[2]:+.4f}] m/s²")

    # 4. Centripetal / Lever-Arm Coupling Analysis
    if np.any(mask_turn):
        w_turn = gyro[mask_turn, 2]
        ax_turn = lin_accel[mask_turn, 0]
        ay_turn = lin_accel[mask_turn, 1]

        # In-place turn lever arm: a_x = -w^2 * r_x
        w_sq = w_turn**2
        high_rot_mask = np.abs(w_turn) > 0.3  # > ~17 deg/s
        if np.sum(high_rot_mask) > 10:
            # Fit lever arm r_x: a_x ≈ -r_x * w_z^2
            rx_est = -np.mean(ax_turn[high_rot_mask] / (w_sq[high_rot_mask] + 1e-4))
            print(f"\n[4] DYNAMIC COUPLING & LEVER ARM:")
            print(f"  Peak Lateral Accel (ay):    {np.max(np.abs(ay_turn)):.3f} m/s²")
            print(f"  Peak Longitudinal Accel (ax): {np.max(np.abs(ax_turn)):.3f} m/s²")
            print(f"  Estimated IMU X Lever Arm:  {rx_est * 100.0:+.1f} cm (distance in front of rotation center)")

    # 5. LiDAR Sweep Motion Skew Analysis
    if num_scans > 0:
        scan_dt = total_dur / num_scans
        max_scan_skew_deg = peak_wz_deg * scan_dt
        print(f"\n[5] LIDAR MOTION DISTORTION:")
        print(f"  Sweep Duration per Scan:  ~{scan_dt * 1000:.1f} ms")
        print(f"  Worst-case Angular Skew:  ~{max_scan_skew_deg:.2f}° per single 360° sweep")
        if max_scan_skew_deg > 3.0:
            print(f"  [CRITICAL] High scan skew (> 3°) detected! Backpropagation/deskewing is essential.")
        else:
            print(f"  [OK] Mild scan skew (<= 3°). Standard rigid matching is viable.")

    # 6. Optional ESIKF Simulation
    if simulate_fusion:
        print(f"\n[6] REPLAYING THROUGH ESIKF STATE ESTIMATOR...")
        if not HAS_FUSION:
            print("  [ERROR] Sensor fusion module could not be imported; skipping replay.")
        else:
            run_fusion_simulation(
                t_imu,
                gyro,
                lin_accel,
                t_lidar_start,
                t_lidar_end,
                lidar_points,
                lidar_scan_offsets,
                invert_gyro_z=invert_gyro_z,
                clean_spikes=clean_spikes,
            )

    # 7. Generate Plot if requested
    if save_plot:
        plot_path = path.with_suffix(".png")
        generate_analysis_plot(
            plot_path=plot_path,
            t_rel=t_rel,
            t_start_rel=t_start_rel,
            t_end_rel=t_end_rel,
            gyro=gyro,
            lin_accel=lin_accel,
            yaw_integrated_deg=yaw_integrated_deg,
            path_name=path.name,
        )


def run_fusion_simulation(
    t_imu,
    gyro,
    lin_accel,
    t_lidar_start,
    t_lidar_end,
    lidar_points,
    lidar_scan_offsets,
    invert_gyro_z: bool = False,
    clean_spikes: bool = False,
):
    """Simulate ESIKF state estimator offline over the recorded data with full fidelity."""
    gyro = gyro.copy()
    lin_accel = lin_accel.copy()

    if invert_gyro_z:
        gyro[:, 2] = -gyro[:, 2]
        print("  [PREPROCESS] Inverted Gyro Z sign to match REP-103 (+Z Up, CCW Left).")

    if clean_spikes:
        spike_count = 0
        for idx in range(len(gyro)):
            if np.linalg.norm(gyro[idx]) > 15.0:
                gyro[idx] = gyro[max(0, idx - 1)]
                spike_count += 1
            if np.linalg.norm(lin_accel[idx]) > 25.0:
                lin_accel[idx] = lin_accel[max(0, idx - 1)]
                spike_count += 1
        if spike_count > 0:
            print(f"  [PREPROCESS] Filtered {spike_count} corrupted 0x7FFF/saturation spikes.")

    estimator = ESIKFStateEstimator()
    estimator.is_static = True
    m = Map()

    imu_meas_buf = deque()
    imu_state_buf = deque()

    num_scans = len(lidar_scan_offsets) - 1
    scan_cursor = 0
    static_count = 2
    static_anchor_pos = None
    static_anchor_R = None
    prev_scan_bins = None
    last_committed_time = None
    last_committed_pos = None
    last_map_pos = None
    last_map_heading = None
    last_turn_time = 0.0

    poses = []
    times = []
    residuals_hist = []
    applied_count = 0
    skipped_count = 0

    # Build initial map from first 2 scans if available
    for s in range(min(2, num_scans)):
        s_idx = lidar_scan_offsets[s]
        e_idx = lidar_scan_offsets[s + 1]
        raw_pts = lidar_points[s_idx:e_idx]
        ranges = raw_pts[:, 2] / 1000.0
        angles = np.radians((360.0 - raw_pts[:, 1]) % 360.0)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        m.add_points(np.column_stack([xs, ys, np.zeros_like(xs)]))

    for i in range(1, len(t_imu)):
        now = t_imu[i]
        dt = float(now - t_imu[i - 1])
        if dt <= 0.0 or dt > 0.1:
            dt = 0.0167

        g_sample = gyro[i].copy()
        a_sample = lin_accel[i].copy()

        imu_meas_buf.append((now, g_sample.copy(), a_sample.copy()))
        while imu_meas_buf and now - imu_meas_buf[0][0] > 3.0:
            imu_meas_buf.popleft()

        # IMU forward prediction
        state, cov = estimator.predict((g_sample, a_sample), dt)
        imu_state_buf.append((now, copy.deepcopy(state), copy.deepcopy(cov)))
        while imu_state_buf and now - imu_state_buf[0][0] > 3.0:
            imu_state_buf.popleft()

        # Check if a LiDAR scan finishes in this interval
        while scan_cursor < num_scans and t_lidar_end[scan_cursor] <= now:
            scan_end = t_lidar_end[scan_cursor]
            scan_start = t_lidar_start[scan_cursor] if t_lidar_start is not None else scan_end - 0.072

            s_idx = lidar_scan_offsets[scan_cursor]
            e_idx = lidar_scan_offsets[scan_cursor + 1]
            raw_s = lidar_points[s_idx:e_idx]
            scan = [(int(p[0]), float(p[1]), float(p[2])) for p in raw_s]
            scan_cursor += 1

            if len(scan) < 10:
                continue

            # Motion deskewing via IMU backpropagation
            compensated_points = backprop(scan_end, scan_start, estimator.state, scan, imu_meas_buf)
            if len(compensated_points) == 0:
                ranges = raw_s[:, 2] / 1000.0
                angles = np.radians((360.0 - raw_s[:, 1]) % 360.0)
                compensated_points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges)])

            # ZUPT and multi-modal motion classification
            current_bins = scan_to_bins(scan)
            recent_imu = [m_rec for m_rec in imu_meas_buf if scan_end - 0.30 <= m_rec[0] <= scan_end + 0.05]
            if len(recent_imu) < 3:
                recent_imu = list(imu_meas_buf)[-10:]
            gyro_max = 0.0
            accel_var = 0.0
            if len(recent_imu) >= 3:
                gyros = np.array([m_rec[1] for m_rec in recent_imu])
                accels = np.array([m_rec[2] for m_rec in recent_imu])
                gyro_max = float(np.max(np.linalg.norm(gyros, axis=1)))
                accel_var = float(np.sum(np.var(accels, axis=0)))

            is_imu_static = (gyro_max < 0.12 and accel_var < 0.40)
            sim = None
            if prev_scan_bins is not None:
                sim = scan_similarity(prev_scan_bins, current_bins)
            prev_scan_bins = current_bins

            is_lidar_static = (sim is None or sim < 20.0)
            is_sample_static = is_imu_static and is_lidar_static

            if is_sample_static:
                static_count += 1
            else:
                static_count = 0
            is_platform_static = (static_count >= 2)
            estimator.is_static = is_platform_static

            # ESIKF LiDAR update
            state_snapshot = copy.deepcopy(estimator.state)
            state_old = copy.deepcopy(state_snapshot)
            P_copy = estimator.P.copy()

            pts_world, update_applied, P_new = estimator.lidar_update(
                scan, state_snapshot, P_copy, compensated_points, m
            )

            if update_applied:
                applied_count += 1
                residuals_hist.append(estimator.last_lidar_residual_norm)
                delta_p = state_snapshot.p - state_old.p
                delta_R = state_old.R.T @ state_snapshot.R

                if is_platform_static:
                    if static_anchor_pos is None:
                        static_anchor_pos = estimator.state.p[:2].copy()
                        static_anchor_R = estimator.state.R.copy()
                    p_cand = estimator.state.p[:2] + delta_p[:2]
                    dist_from_anchor = float(np.linalg.norm(p_cand - static_anchor_pos))
                    MAX_STATIONARY_DRIFT = 0.03
                    if dist_from_anchor > MAX_STATIONARY_DRIFT:
                        p_cand = static_anchor_pos + (p_cand - static_anchor_pos) / dist_from_anchor * MAX_STATIONARY_DRIFT
                    estimator.state.p[:2] = p_cand
                    estimator.state.p[2] = 0.0
                    estimator.state.v = np.zeros(3)
                    if static_anchor_R is not None:
                        estimator.state.R = static_anchor_R.copy()
                else:
                    if static_anchor_pos is not None:
                        estimator.state.p[:2] = static_anchor_pos.copy()
                        static_anchor_pos = None
                        static_anchor_R = None
                        delta_p_body = estimator.state.R.T @ delta_p
                        if delta_p_body[0] < 0.0:
                            delta_p_body[0] = 0.0
                            delta_p = estimator.state.R @ delta_p_body
                    estimator.state.p += delta_p
                    estimator.state.p[2] = 0.0

                if is_platform_static:
                    estimator.state.v = np.zeros(3)
                else:
                    # Velocity reconciliation with Non-Holonomic Constraints (NHC)
                    if last_committed_time is not None and last_committed_pos is not None:
                        dt_scan = scan_end - last_committed_time
                        if 0.08 <= dt_scan <= 1.5:
                            disp = float(np.linalg.norm(estimator.state.p[:2] - last_committed_pos[:2]))
                            is_jump = (disp > 0.12 or (disp / max(dt_scan, 0.01)) > 1.2)
                            if is_jump or disp < 0.03:
                                v_obs = np.zeros(3)
                            else:
                                v_obs = (estimator.state.p - last_committed_pos) / dt_scan
                                v_obs[2] = 0.0
                            v_obs_body = estimator.state.R.T @ v_obs
                            v_obs_body[1] = 0.0
                            v_obs_body[2] = 0.0
                            is_active_turn = (gyro_max > 0.15)
                            if is_active_turn and v_obs_body[0] < 0.0:
                                v_obs_body[0] = 0.0
                            v_obs = estimator.state.R @ v_obs_body
                            speed_obs = float(np.linalg.norm(v_obs[:2]))
                            if speed_obs > 0.8:
                                v_obs[:2] = (v_obs[:2] / speed_obs) * 0.8
                            alpha_v = 0.7
                            estimator.state.v[:2] = (1.0 - alpha_v) * estimator.state.v[:2] + alpha_v * v_obs[:2]
                            estimator.state.v[2] = 0.0

                last_committed_pos = estimator.state.p.copy()
                last_committed_time = scan_end
                if not is_platform_static:
                    estimator.state.R = estimator.state.R @ delta_R
                    estimator.state.R = reorthonormalize(estimator.state.R)
                estimator.P = P_new
            else:
                skipped_count += 1

            # Map frontier ingestion and post-turn settling logic
            current_heading = float(np.arctan2(estimator.state.R[1, 0], estimator.state.R[0, 0]))
            if last_map_heading is None:
                heading_diff = 1.0
            else:
                heading_diff = abs(float(np.arctan2(np.sin(current_heading - last_map_heading), np.cos(current_heading - last_map_heading))))
            pos_diff = float(np.linalg.norm(estimator.state.p[:2] - last_map_pos[:2])) if last_map_pos is not None else 1.0

            is_high_speed_turn = (gyro_max > 0.80)
            is_moderate_turn = (gyro_max > 0.20)
            if is_moderate_turn:
                last_turn_time = now
            turn_settled = (now - last_turn_time >= 0.15)

            can_add_points = False
            if pts_world is not None and update_applied:
                if not is_platform_static and not is_moderate_turn and turn_settled and pos_diff > 0.10:
                    can_add_points = True
                elif not is_high_speed_turn and heading_diff >= 0.26:
                    can_add_points = True
                elif is_platform_static and (heading_diff >= 0.26 or (last_map_pos is None or pos_diff > 0.10)):
                    can_add_points = True

            if can_add_points:
                m.add_points(pts_world)
                last_map_pos = estimator.state.p.copy()
                last_map_heading = current_heading

            if is_platform_static:
                estimator.zupt_update()
                estimator.state.v = np.zeros(3)
                last_committed_pos = estimator.state.p.copy()
                last_committed_time = scan_end

        poses.append(estimator.state.p.copy())
        times.append(now - t_imu[0])

    poses = np.array(poses)
    final_p = poses[-1]
    final_yaw_deg = float(np.degrees(np.arctan2(estimator.state.R[1, 0], estimator.state.R[0, 0])))
    max_disp = float(np.max(np.linalg.norm(poses[:, :2], axis=1)))
    net_disp = float(np.linalg.norm(final_p[:2]))

    print(f"  Simulation completed successfully ({len(poses)} IMU steps, {num_scans} LiDAR scans).")
    print(f"  Scans Applied:            {applied_count}/{num_scans} ({applied_count / max(num_scans, 1) * 100:.1f}%) | Skipped: {skipped_count}")
    print(f"  Map Points Built:         {m.num_points()} points")
    print(f"  Final Estimated Position: [x={final_p[0]:+.3f}m, y={final_p[1]:+.3f}m]")
    print(f"  Net Trajectory Length:    {net_disp:.3f}m (Max Excursion: {max_disp:.3f}m)")
    print(f"  Final Estimated Heading:  {final_yaw_deg:+.1f}°")
    if residuals_hist:
        print(f"  Residual Norm:            mean={np.mean(residuals_hist):.4f}, median={np.median(residuals_hist):.4f}, max={np.max(residuals_hist):.4f}")


def generate_analysis_plot(plot_path, t_rel, t_start_rel, t_end_rel, gyro, lin_accel, yaw_integrated_deg, path_name):
    """Save publication-quality 3-panel figure of turning kinematics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; plot skipped.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Kinematic Turn Profile: {path_name}", fontsize=13, fontweight="bold")

    # Panel 1: Yaw Rate & Integrated Heading
    ax1 = axes[0]
    ax1.plot(t_rel, np.degrees(gyro[:, 2]), color="crimson", lw=1.5, label=r"Yaw Rate $\omega_z$ (deg/s)")
    ax1.axvspan(t_start_rel, t_end_rel, color="gold", alpha=0.15, label="Active Turn Window")
    ax1.set_ylabel("Yaw Rate (deg/s)", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_rel, yaw_integrated_deg, color="navy", lw=2, linestyle="--", label="Integrated Heading (deg)")
    ax1_twin.set_ylabel("Heading (deg)", color="navy")
    ax1_twin.tick_params(axis="y", labelcolor="navy")
    ax1_twin.legend(loc="upper right")

    # Panel 2: Accelerations
    ax2 = axes[1]
    ax2.plot(t_rel, lin_accel[:, 0], color="forestgreen", lw=1.2, label=r"Forward Accel $a_x$ (m/s²)")
    ax2.plot(t_rel, lin_accel[:, 1], color="darkorange", lw=1.5, label=r"Lateral Accel $a_y$ (m/s²)")
    ax2.axvspan(t_start_rel, t_end_rel, color="gold", alpha=0.15)
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Panel 3: High-frequency Angular Wobble (Roll & Pitch)
    ax3 = axes[2]
    ax3.plot(t_rel, np.degrees(gyro[:, 0]), color="purple", lw=1.0, alpha=0.8, label=r"Roll Rate $\omega_x$ (deg/s)")
    ax3.plot(t_rel, np.degrees(gyro[:, 1]), color="teal", lw=1.0, alpha=0.8, label=r"Pitch Rate $\omega_y$ (deg/s)")
    ax3.axvspan(t_start_rel, t_end_rel, color="gold", alpha=0.15)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Off-Axis Rate (deg/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  [SAVED] Analysis plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded turn dataset for sensor fusion debugging.")
    parser.add_argument("npz_file", type=str, help="Path to .npz turn dataset file")
    parser.add_argument("--simulate-fusion", action="store_true", help="Replay data through ESIKFStateEstimator")
    parser.add_argument("--save-plot", action="store_true", help="Generate and save diagnostic kinematic plots")
    parser.add_argument("--invert-gyro-z", action="store_true", help="Invert Gyro Z sign to align with ROS REP-103 (+Z Up, CCW Left)")
    parser.add_argument("--clean-spikes", action="store_true", help="Filter BNO055 I2C 0x7FFF / saturation spikes (>25 m/s² or >15 rad/s)")

    args = parser.parse_args()
    analyze_dataset(
        args.npz_file,
        simulate_fusion=args.simulate_fusion,
        save_plot=args.save_plot,
        invert_gyro_z=args.invert_gyro_z,
        clean_spikes=args.clean_spikes,
    )


if __name__ == "__main__":
    main()
