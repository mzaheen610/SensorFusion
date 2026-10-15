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
import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

try:
    from forward import ESIKFStateEstimator, State
    from map import Map
    from utils.so3_rotation import log
    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False


def analyze_dataset(npz_path: str, simulate_fusion: bool = False, save_plot: bool = False):
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
    gyro = data["imu_gyro"]  # (M, 3) in rad/s
    lin_accel = data["imu_lin_accel"]  # (M, 3) in m/s^2
    raw_accel = data["imu_raw_accel"]  # (M, 3) in m/s^2
    t_start_turn = float(data["turn_start_time"])
    t_end_turn = float(data["turn_end_time"])

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
            run_fusion_simulation(t_imu, gyro, lin_accel, t_lidar_end, lidar_points, lidar_scan_offsets)

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


def run_fusion_simulation(t_imu, gyro, lin_accel, t_lidar_end, lidar_points, lidar_scan_offsets):
    """Simulate ESIKF state estimator offline over the recorded data."""
    estimator = ESIKFStateEstimator()
    m = Map()

    # Build initial map from first 2 scans
    if len(lidar_scan_offsets) > 2:
        for s in range(min(2, len(lidar_scan_offsets) - 1)):
            s_idx = lidar_scan_offsets[s]
            e_idx = lidar_scan_offsets[s + 1]
            pts = lidar_points[s_idx:e_idx]
            ranges = pts[:, 2] / 1000.0
            angles = np.radians((360.0 - pts[:, 1]) % 360.0)
            xs = ranges * np.cos(angles)
            ys = ranges * np.sin(angles)
            pts_w = np.column_stack([xs, ys, np.zeros_like(xs)])
            m.add_points(pts_w)

    dt_nominal = 0.033
    poses = []
    times = []

    scan_cursor = 0
    num_scans = len(lidar_scan_offsets) - 1

    for i in range(1, len(t_imu)):
        dt = float(t_imu[i] - t_imu[i - 1])
        if dt <= 0 or dt > 0.1:
            dt = dt_nominal

        u = (gyro[i], lin_accel[i])
        estimator.predict(u, dt)

        # Check if a LiDAR scan falls in this interval
        if scan_cursor < num_scans and t_lidar_end[scan_cursor] <= t_imu[i]:
            s_idx = lidar_scan_offsets[scan_cursor]
            e_idx = lidar_scan_offsets[scan_cursor + 1]
            pts = lidar_points[s_idx:e_idx]
            if len(pts) > 10:
                ranges = pts[:, 2] / 1000.0
                angles = np.radians((360.0 - pts[:, 1]) % 360.0)
                body_pts = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges)])

                scan_tuples = [(int(p[0]), float(p[1]), float(p[2])) for p in pts]
                state_copy = State(
                    R=estimator.state.R.copy(),
                    p=estimator.state.p.copy(),
                    v=estimator.state.v.copy(),
                    bg=estimator.state.bg.copy(),
                    ba=estimator.state.ba.copy(),
                    g=estimator.state.g.copy(),
                )
                estimator.lidar_update(
                    scan=scan_tuples,
                    state=state_copy,
                    P_copy=estimator.P.copy(),
                    lidar_points_compensated=body_pts,
                    map=m,
                )
            scan_cursor += 1

        poses.append(estimator.state.p.copy())
        times.append(t_imu[i] - t_imu[0])

    poses = np.array(poses)
    final_p = poses[-1]
    max_disp = np.max(np.linalg.norm(poses[:, :2], axis=1))

    print(f"  Simulation completed successfully ({len(poses)} steps).")
    print(f"  Final Estimated Position: [x={final_p[0]:+.3f}m, y={final_p[1]:+.3f}m]")
    print(f"  Max Horizontal Excursion: {max_disp:.3f}m")


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

    args = parser.parse_args()
    analyze_dataset(args.npz_file, simulate_fusion=args.simulate_fusion, save_plot=args.save_plot)


if __name__ == "__main__":
    main()
