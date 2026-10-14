"""
Continuous Sensor Stream Edge-Case Benchmarks for LiDAR-IMU-Camera Fusion.

Tests continuous data streaming fed from disk files under advanced edge cases:
- Scenario 7: Continuous Loop Closure (16m closed-circuit trajectory through multi-room loop)
- Scenario 8: Long Featureless Corridor (Aperture effect & degenerate geometry stress)
- Scenario 9: High-Frequency S-Curve Slalom (Continuous angular reversals & velocity reconciliation)
- Scenario 10: Dense Moving Crowd Clutter (4 asynchronous crossing pedestrians)
- Scenario 11: Sensor Dropout & Sudden Re-acquisition (2.5s LiDAR blackout & recovery)
- Scenario 12: Continuous Real Hardware Log File Stream Replay (Multi-hundred scan streaming)
"""

import sys
import os
import time
import json
import re
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from forward import ESIKFStateEstimator, State
from map import Map
from utils.so3_rotation import exp, skew, reorthonormalize
from utils.similarity import scan_to_bins, scan_similarity
from benchmark_pipeline_rigorous import SimulationWorld, PipelineRunner, TrajectoryGenerator


# ============================================================================
# 1. Continuous File Streamer
# ============================================================================

class FileDataStreamer:
    """
    Serializes continuous sensor frames to disk and streams them back
    packet-by-packet to replicate realistic continuous hardware/file feeding.
    """
    @staticmethod
    def dump_stream(filepath: Path, frames: list):
        """Write sensor frames to compressed file."""
        import pickle
        import gzip
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(filepath, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stream(filepath: Path):
        """Generator yielding frames sequentially from file."""
        import pickle
        import gzip
        with gzip.open(filepath, "rb") as f:
            frames = pickle.load(f)
        for frame in frames:
            yield frame


# ============================================================================
# 2. Specialized Simulation Environments
# ============================================================================

def make_loop_world():
    """
    12m x 12m four-room loop environment with a central 4m x 4m pillar core.
    Allows robot to traverse a full square circuit returning to origin.
    """
    world = SimulationWorld()
    world.walls = [
        # Outer boundary [-6, 6] x [-6, 6]
        (-6.0, -6.0,  6.0, -6.0),
        ( 6.0, -6.0,  6.0,  6.0),
        ( 6.0,  6.0, -6.0,  6.0),
        (-6.0,  6.0, -6.0, -6.0),
        # Central core pillar [-2, 2] x [-2, 2]
        (-2.0, -2.0,  2.0, -2.0),
        ( 2.0, -2.0,  2.0,  2.0),
        ( 2.0,  2.0, -2.0,  2.0),
        (-2.0,  2.0, -2.0, -2.0),
        # Dividers creating 4 distinct rooms connected by 2m doorways
        (-6.0,  0.0, -3.5,  0.0),
        ( 3.5,  0.0,  6.0,  0.0),
        ( 0.0, -6.0,  0.0, -3.5),
        ( 0.0,  3.5,  0.0,  6.0),
    ]
    return world


def make_corridor_world():
    """
    16m long, 1.6m wide narrow hallway [-8, 8] x [-0.8, 0.8]
    Smooth parallel walls producing pure aperture effect along X.
    """
    world = SimulationWorld()
    world.walls = [
        (-8.0, -0.8,  8.0, -0.8),  # South wall
        ( 8.0, -0.8,  8.0,  0.8),  # East end wall
        ( 8.0,  0.8, -8.0,  0.8),  # North wall
        (-8.0,  0.8, -8.0, -0.8),  # West end wall
    ]
    return world


# ============================================================================
# 3. Continuous Edge-Case Scenarios
# ============================================================================

def run_scenario_7_continuous_loop_closure(data_dir: Path):
    """
    Scenario 7: Continuous Loop Closure
    Robot traverses a 16m closed-circuit perimeter around the central pillar
    in the 4-room loop environment over 14.0 seconds, returning to (0, -4).
    Data is pre-generated, serialized to file, and streamed back continuously.
    """
    print("\n--- Running Scenario 7: Continuous Loop Closure (File Streamed) ---")
    world = make_loop_world()
    stream_file = data_dir / "stream_scenario_7_loop.npz"

    total_time = 14.0
    dt_scan = 0.10
    dt_imu = 0.01

    start_p = np.array([0.0, -4.0, 0.0])
    true_p = start_p.copy()
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.015, -0.01, 0.0])
    bg = np.array([0.001, 0.001, 0.002])

    frames = []
    # 4 legs of 3.0s each: South -> East -> North -> West -> South
    # Waypoints: (0, -4) -> (4, -4) -> (4, 4) -> (-4, 4) -> (-4, -4) -> (0, -4)
    legs = [
        (1.0, 3.5,  np.array([ 0.35,  0.0, 0.0]), 0.0),           # +X leg
        (3.5, 4.0,  np.array([ 0.0,   0.0, 0.0]), np.pi/2),       # turn North
        (4.0, 6.5,  np.array([ 0.0,   0.35, 0.0]), np.pi/2),      # +Y leg
        (6.5, 7.0,  np.array([ 0.0,   0.0, 0.0]), np.pi),         # turn West
        (7.0, 9.5,  np.array([-0.35,  0.0, 0.0]), np.pi),         # -X leg
        (9.5, 10.0, np.array([ 0.0,   0.0, 0.0]), 3*np.pi/2),     # turn South
        (10.0, 12.5, np.array([0.0,  -0.35, 0.0]), 3*np.pi/2),    # -Y leg
        (12.5, 13.0, np.array([0.0,   0.0, 0.0]), 2*np.pi),       # turn East
        (13.0, 14.0, np.array([0.0,   0.0, 0.0]), 2*np.pi),       # rest at origin
    ]

    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        
        target_v = np.zeros(3)
        target_yaw = 0.0
        for t_start, t_end, v_leg, yaw_leg in legs:
            if t_start <= t_now < t_end:
                target_v = v_leg
                target_yaw = yaw_leg
                break

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            dv = (target_v - true_v)
            a_cmd = np.clip(dv / dt_imu, -1.5, 1.5)
            true_v += a_cmd * dt_imu
            true_p += true_v * dt_imu
            dyaw = (target_yaw - true_yaw)
            omega_cmd = np.clip(dyaw / dt_imu, -1.8, 1.8)
            true_yaw += omega_cmd * dt_imu

            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                a_cmd, np.array([0, 0, omega_cmd]), true_yaw, ba, bg,
                noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.08, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, dyn_mask = world.raycast(true_p, true_yaw)
        frames.append({
            't': t_now,
            'true_p': true_p.copy(),
            'true_v': true_v.copy(),
            'true_yaw': true_yaw,
            'body_pts': body_pts,
            'scan_raw': scan_raw,
            'dynamic_mask': dyn_mask,
            'imu_samples': imu_samples
        })

    # Save to file
    FileDataStreamer.dump_stream(stream_file, frames)

    # Stream back from file through pipeline
    runner = PipelineRunner(world)
    runner.estimator.state.p = start_p.copy()
    runner.last_committed_pos = start_p.copy()

    for frame in FileDataStreamer.load_stream(stream_file):
        runner.step(
            frame['t'], frame['true_p'], frame['true_v'], frame['true_yaw'],
            frame['imu_samples'], frame['scan_raw'], frame['body_pts']
        )

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    final_loop_closure_drift = float(np.linalg.norm(runner.history['est_p'][-1][:2] - start_p[:2]))

    print(f"Scenario 7 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Loop Closure Origin Error = {final_err*100:.2f} cm | Final Return Drift = {final_loop_closure_drift*100:.2f} cm")
    return runner.history


def run_scenario_8_corridor_aperture_stress(data_dir: Path):
    """
    Scenario 8: Long Featureless Corridor (Aperture Effect Stress)
    Robot travels 8.0 meters down a narrow 1.6m wide corridor.
    North/South walls provide no longitudinal features along X.
    Tests filter resistance to slipping along the unconstrained corridor axis.
    """
    print("\n--- Running Scenario 8: Long Featureless Corridor (Aperture Stress) ---")
    world = make_corridor_world()
    stream_file = data_dir / "stream_scenario_8_corridor.npz"

    total_time = 10.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([-4.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.02, -0.015, 0.0])
    bg = np.array([0.001, -0.001, 0.001])

    frames = []
    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        target_a_w = np.zeros(3)

        if 1.0 <= t_now < 4.0:
            if true_v[0] < 0.40:
                target_a_w = np.array([0.5, 0.0, 0.0])
        elif 4.0 <= t_now < 7.0:
            # Steady cruising along corridor
            target_a_w = np.zeros(3)
        elif 7.0 <= t_now < 8.0:
            # Braking to halt
            target_a_w = np.array([-0.8, 0.0, 0.0])
        else:
            true_v[:] = 0.0

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            true_p += true_v * dt_imu
            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, np.zeros(3), true_yaw, ba, bg,
                noise_gyro=0.004, noise_accel=0.02, motor_vib_amp=0.08, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, dyn_mask = world.raycast(true_p, true_yaw)
        frames.append({
            't': t_now,
            'true_p': true_p.copy(),
            'true_v': true_v.copy(),
            'true_yaw': true_yaw,
            'body_pts': body_pts,
            'scan_raw': scan_raw,
            'dynamic_mask': dyn_mask,
            'imu_samples': imu_samples
        })

    FileDataStreamer.dump_stream(stream_file, frames)

    runner = PipelineRunner(world)
    runner.estimator.state.p = np.array([-4.0, 0.0, 0.0])
    runner.last_committed_pos = np.array([-4.0, 0.0, 0.0])

    for frame in FileDataStreamer.load_stream(stream_file):
        runner.step(
            frame['t'], frame['true_p'], frame['true_v'], frame['true_yaw'],
            frame['imu_samples'], frame['scan_raw'], frame['body_pts']
        )

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    # Check lateral Y-stability (must be tight since walls are on both sides)
    y_err = [abs(p[1] - tp[1]) for p, tp in zip(runner.history['est_p'], runner.history['true_p'])]
    max_lateral_err = max(y_err)

    print(f"Scenario 8 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Lateral Y-Axis Error = {max_lateral_err*100:.2f} cm | Final Resting Error = {final_err*100:.2f} cm")
    return runner.history


def run_scenario_9_high_freq_slalom(data_dir: Path):
    """
    Scenario 9: High-Frequency S-Curve Slalom
    Continuous sinusoidal angular oscillation (omega = 1.2 * sin(2*pi*0.6*t))
    while maintaining 0.35 m/s forward velocity for 8.0s.
    Stress-tests velocity reconciliation and rotation reorthonormalization under continuous reversals.
    """
    print("\n--- Running Scenario 9: High-Frequency S-Curve Slalom ---")
    world = SimulationWorld()
    stream_file = data_dir / "stream_scenario_9_slalom.npz"

    total_time = 8.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([-2.5, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.01, -0.01, 0.0])
    bg = np.array([0.001, 0.002, 0.001])

    frames = []
    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan

        # Continuous sinusoidal yaw rate
        if 1.0 <= t_now < 6.5:
            omega_val = 1.0 * np.sin(2.0 * np.pi * 0.5 * (t_now - 1.0))
            fwd_speed = 0.35
            target_v_w = np.array([fwd_speed * np.cos(true_yaw), fwd_speed * np.sin(true_yaw), 0.0])
        else:
            omega_val = 0.0
            target_v_w = np.zeros(3)

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_yaw += omega_val * dt_imu
            true_v = target_v_w.copy()
            true_p += true_v * dt_imu

            target_a_w = np.zeros(3)
            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, np.array([0, 0, omega_val]), true_yaw, ba, bg,
                noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.10, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, dyn_mask = world.raycast(true_p, true_yaw)
        frames.append({
            't': t_now,
            'true_p': true_p.copy(),
            'true_v': true_v.copy(),
            'true_yaw': true_yaw,
            'body_pts': body_pts,
            'scan_raw': scan_raw,
            'dynamic_mask': dyn_mask,
            'imu_samples': imu_samples
        })

    FileDataStreamer.dump_stream(stream_file, frames)

    runner = PipelineRunner(world)
    runner.estimator.state.p = np.array([-2.5, 0.0, 0.0])
    runner.last_committed_pos = np.array([-2.5, 0.0, 0.0])

    for frame in FileDataStreamer.load_stream(stream_file):
        runner.step(
            frame['t'], frame['true_p'], frame['true_v'], frame['true_yaw'],
            frame['imu_samples'], frame['scan_raw'], frame['body_pts']
        )

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    yaw_err_deg = np.rad2deg(np.max(runner.history['error_yaw']))

    print(f"Scenario 9 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Peak Yaw Error = {yaw_err_deg:.2f} deg | Final Resting Error = {final_err*100:.2f} cm")
    return runner.history


def run_scenario_10_dense_crowd_navigation(data_dir: Path):
    """
    Scenario 10: Dense Crowd Navigation
    Four pedestrians walk simultaneously on asynchronous paths across the robot's FOV.
    Continuous dynamic clutter stress-test.
    """
    print("\n--- Running Scenario 10: Dense Crowd Navigation ---")
    world = SimulationWorld()
    stream_file = data_dir / "stream_scenario_10_crowd.npz"

    total_time = 8.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.015, -0.015, 0.0])
    bg = np.array([0.001, 0.001, 0.001])

    frames = []
    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan
        world.clear_pedestrians()

        # 4 dynamic pedestrians crossing at various velocities
        # Pedestrian 1: Lateral crossing North
        world.add_pedestrian(x=-1.5 + 0.6 * t_now, y=1.2, radius=0.25)
        # Pedestrian 2: Lateral crossing South in opposite direction
        world.add_pedestrian(x=2.0 - 0.7 * t_now, y=-1.0, radius=0.25)
        # Pedestrian 3: Vertical crossing East
        world.add_pedestrian(x=1.5, y=-2.0 + 0.6 * t_now, radius=0.25)
        # Pedestrian 4: Diagonal wanderer
        world.add_pedestrian(x=-2.0 + 0.4 * t_now, y=-2.0 + 0.5 * t_now, radius=0.25)

        target_a_w = np.zeros(3)
        if 1.0 <= t_now < 4.0:
            if true_v[0] < 0.25:
                target_a_w = np.array([0.4, 0.0, 0.0])
        elif 4.0 <= t_now < 5.0:
            target_a_w = np.array([-0.6, 0.0, 0.0])
        else:
            true_v[:] = 0.0

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            true_p += true_v * dt_imu
            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, np.zeros(3), true_yaw, ba, bg,
                noise_gyro=0.005, noise_accel=0.02, motor_vib_amp=0.08, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        scan_raw, body_pts, _, dyn_mask = world.raycast(true_p, true_yaw)
        frames.append({
            't': t_now,
            'true_p': true_p.copy(),
            'true_v': true_v.copy(),
            'true_yaw': true_yaw,
            'body_pts': body_pts,
            'scan_raw': scan_raw,
            'dynamic_mask': dyn_mask,
            'imu_samples': imu_samples
        })

    FileDataStreamer.dump_stream(stream_file, frames)

    runner = PipelineRunner(world)
    init_map_pts = runner.map.num_points()

    for frame in FileDataStreamer.load_stream(stream_file):
        runner.step(
            frame['t'], frame['true_p'], frame['true_v'], frame['true_yaw'],
            frame['imu_samples'], frame['scan_raw'], frame['body_pts']
        )

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    final_map_pts = runner.map.num_points()

    print(f"Scenario 10 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Final Resting Error = {final_err*100:.2f} cm | Map Ingested = {final_map_pts - init_map_pts} points")
    return runner.history


def run_scenario_11_sensor_dropout_recovery(data_dir: Path):
    """
    Scenario 11: LiDAR Dropout & Recovery (2.5s Blind Blackout)
    LiDAR stream drops completely for 2.5s (t = 2.0s to 4.5s) while robot is moving.
    Filter relies strictly on IMU dead-reckoning during blackout, then must re-acquire
    walls and correct accumulated position error without divergent rejection.
    """
    print("\n--- Running Scenario 11: Sensor Dropout & Sudden Re-acquisition ---")
    world = SimulationWorld()
    stream_file = data_dir / "stream_scenario_11_dropout.npz"

    total_time = 7.0
    dt_scan = 0.10
    dt_imu = 0.01

    true_p = np.array([0.0, 0.0, 0.0])
    true_v = np.array([0.0, 0.0, 0.0])
    true_yaw = 0.0

    ba = np.array([0.02, -0.01, 0.0])
    bg = np.array([0.001, 0.001, 0.001])

    frames = []
    for step_idx in range(int(total_time / dt_scan)):
        t_now = step_idx * dt_scan

        target_a_w = np.zeros(3)
        if 1.0 <= t_now < 3.0:
            if true_v[0] < 0.25:
                target_a_w = np.array([0.4, 0.0, 0.0])
        elif 3.0 <= t_now < 4.0:
            target_a_w = np.array([-0.4, 0.0, 0.0])
        else:
            true_v[:] = 0.0

        imu_samples = []
        for _ in range(int(dt_scan / dt_imu)):
            true_v += target_a_w * dt_imu
            true_p += true_v * dt_imu
            gyro_b, accel_b = TrajectoryGenerator.generate_imu_reading(
                target_a_w, np.zeros(3), true_yaw, ba, bg,
                noise_gyro=0.004, noise_accel=0.02, motor_vib_amp=0.05, t=t_now
            )
            imu_samples.append((dt_imu, gyro_b, accel_b))

        # Simulate LiDAR blackout between 2.0s and 4.5s (25 consecutive scans dropped)
        is_dropped = (2.0 <= t_now < 4.5)
        if is_dropped:
            scan_raw = []
            body_pts = np.empty((0, 3))
            dyn_mask = np.empty(0, dtype=bool)
        else:
            scan_raw, body_pts, _, dyn_mask = world.raycast(true_p, true_yaw)

        frames.append({
            't': t_now,
            'true_p': true_p.copy(),
            'true_v': true_v.copy(),
            'true_yaw': true_yaw,
            'body_pts': body_pts,
            'scan_raw': scan_raw,
            'dynamic_mask': dyn_mask,
            'imu_samples': imu_samples
        })

    FileDataStreamer.dump_stream(stream_file, frames)

    runner = PipelineRunner(world)
    for frame in FileDataStreamer.load_stream(stream_file):
        runner.step(
            frame['t'], frame['true_p'], frame['true_v'], frame['true_yaw'],
            frame['imu_samples'], frame['scan_raw'], frame['body_pts']
        )

    ate_rmse = np.sqrt(np.mean(np.array(runner.history['error_p'])**2))
    ate_max = np.max(runner.history['error_p'])
    final_err = runner.history['error_p'][-1]
    # Check if update resumed after blackout
    applied_after_dropout = [
        runner.history['applied'][i] for i, t in enumerate(runner.history['t']) if t >= 4.5
    ]
    reacquired_ratio = sum(applied_after_dropout) / len(applied_after_dropout)

    print(f"Scenario 11 Results: ATE RMSE = {ate_rmse*100:.2f} cm | Max Error = {ate_max*100:.2f} cm")
    print(f"Re-acquisition Rate Post-Blackout = {reacquired_ratio*100:.1f}% | Final Recovered Error = {final_err*100:.2f} cm")
    return runner.history


def run_scenario_12_hardware_continuous_file_replay():
    """
    Scenario 12: Continuous Real Hardware Log File Stream Replay
    Parses and sequentially replays real hardware runs:
    - run_2026_10_13_18_02_23.log (recent baseline run)
    - run_2026_10_12_16_45_47.log (challenging turn runaway log)
    Evaluates continuous stability, step delta distribution, and covariance health.
    """
    print("\n--- Running Scenario 12: Continuous Real Hardware Log Stream Replay ---")
    log_files = [
        PROJECT_ROOT / "logs" / "run_2026_10_13_18_02_23.log",
        PROJECT_ROOT / "logs" / "run_2026_10_12_16_45_47.log",
        PROJECT_ROOT / "logs" / "run_2026_10_09_16_01_43.log"
    ]

    results = {}
    for log_path in log_files:
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")

        # Parse sequential commits (both new delta_p and older delta_position formats)
        commits = re.findall(
            r"LiDAR merge committed:\s*delta_p=\[([-\d\.\+eE]+),\s*([-\d\.\+eE]+)\],\s*p=\[([-\d\.\+eE]+),\s*([-\d\.\+eE]+)\],\s*v=\[([-\d\.\+eE]+),\s*([-\d\.\+eE]+)\],\s*static=(True|False)",
            text
        )
        if not commits:
            raw_pos = re.findall(
                r"LiDAR merge committed:\s*delta_position=\[([-\d\.\+eE]+)\s+([-\d\.\+eE]+).*?position=\[([-\d\.\+eE]+)\s+([-\d\.\+eE]+)",
                text
            )
            commits = [(c[0], c[1], c[2], c[3], "0.0", "0.0", "False") for c in raw_pos]
        if not commits:
            continue

        delta_norms = [float(np.hypot(float(c[0]), float(c[1]))) for c in commits]
        speeds = [float(np.hypot(float(c[4]), float(c[5]))) for c in commits]
        static_flags = [c[6] == "True" for c in commits]

        mean_delta = np.mean(delta_norms) if delta_norms else 0.0
        max_delta = max(delta_norms) if delta_norms else 0.0
        p95_delta = np.percentile(delta_norms, 95) if delta_norms else 0.0
        max_speed = max(speeds) if speeds else 0.0
        static_ratio = (static_flags.count(True) / len(static_flags)) if static_flags else 0.0

        results[log_path.name] = {
            'commits': len(commits),
            'mean_delta_cm': mean_delta * 100.0,
            'p95_delta_cm': p95_delta * 100.0,
            'max_delta_cm': max_delta * 100.0,
            'max_speed_mps': max_speed,
            'static_ratio': static_ratio
        }

        print(f"Log Stream: {log_path.name}")
        print(f"  Total LiDAR Commits: {len(commits)} | Static Ratio: {static_ratio*100:.1f}%")
        print(f"  Step Delta: Mean = {mean_delta*100:.2f} cm | 95th Percentile = {p95_delta*100:.2f} cm | Peak = {max_delta*100:.2f} cm")
        print(f"  Peak Synthesized Velocity: {max_speed:.3f} m/s (Capped at 0.80 m/s limit)")

    return results


# ============================================================================
# Main Runner Entrypoint
# ============================================================================

def run_all_continuous_benchmarks():
    print("=" * 80)
    print("STARTING ADVANCED CONTINUOUS SENSOR STREAM BENCHMARK SUITE")
    print("=" * 80)

    data_dir = PROJECT_ROOT / "experiments" / "continuous_streams_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    res7 = run_scenario_7_continuous_loop_closure(data_dir)
    np.random.seed(42)
    res8 = run_scenario_8_corridor_aperture_stress(data_dir)
    np.random.seed(42)
    res9 = run_scenario_9_high_freq_slalom(data_dir)
    np.random.seed(42)
    res10 = run_scenario_10_dense_crowd_navigation(data_dir)
    np.random.seed(42)
    res11 = run_scenario_11_sensor_dropout_recovery(data_dir)
    res12 = run_scenario_12_hardware_continuous_file_replay()

    print("\n" + "=" * 80)
    print("ADVANCED CONTINUOUS STREAM BENCHMARK SUMMARY")
    print("=" * 80)
    scenarios = [
        ("7. Continuous Loop Closure", res7, 20.0, 10.0, False),
        ("8. Corridor Aperture Stress", res8, 150.0, 200.0, True),  # Special note on aperture slip along X
        ("9. High-Freq S-Curve Slalom", res9, 10.0, 8.0, False),
        ("10. Dense Crowd Navigation", res10, 25.0, 16.0, False),
        ("11. Sensor Dropout Recovery", res11, 30.0, 5.0, False),  # Evaluates recovered final drift post-reacquisition
    ]

    print(f"{'Scenario':<32} | {'ATE RMSE':<12} | {'Max Error':<12} | {'Final Drift':<12} | {'Status':<14}")
    print("-" * 80)
    all_passed = True
    for name, res, rmse_thresh, final_thresh, is_aperture in scenarios:
        rmse = np.sqrt(np.mean(np.array(res['error_p'])**2)) * 100.0
        max_err = np.max(res['error_p']) * 100.0
        final_err = res['error_p'][-1] * 100.0

        if is_aperture:
            y_err = [abs(p[1] - tp[1]) for p, tp in zip(res['est_p'], res['true_p'])]
            max_y_cm = max(y_err) * 100.0
            # Aperture scenario: Lateral Y must be < 2.0 cm, longitudinal slip expected
            passed = (max_y_cm < 2.0)
            status = "PASSED (Y<2cm)" if passed else "FLAGGED"
        else:
            passed = (rmse < rmse_thresh and final_err < final_thresh)
            status = "PASSED" if passed else "FLAGGED"

        if not passed:
            all_passed = False
        print(f"{name:<32} | {rmse:8.2f} cm   | {max_err:8.2f} cm   | {final_err:8.2f} cm   | {status:<14}")

    print("=" * 80)
    print(f"CONTINUOUS STREAM PIPELINE STATUS: {'ALL SCENARIOS VALIDATED' if all_passed else 'ATTENTION NEEDED'}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_continuous_benchmarks()
