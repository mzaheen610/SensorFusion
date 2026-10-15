#!/usr/bin/env python3
"""
High-Precision Turn Dataset Recorder for Multimodal Sensor Fusion
==================================================================

Records synchronized raw IMU readings (gyro, linear accel, total accel, 
gravity, onboard quaternion) and full RPLidar 360-degree point scans during 
various robotic turning maneuvers (pivot turns, arcing turns, cornering, etc.).

Outputs:
  - Compressed NumPy archive (.npz) containing synchronized raw arrays
  - Human-readable metadata file (.json) with trial statistics
  - Diagnostic preview plot (.png) showing turn kinematics and scan progression

Usage:
  # Interactive mode with menu-driven turn selection:
  python3 scripts/record_turn_dataset.py

  # Direct CLI execution for scripted data collection:
  python3 scripts/record_turn_dataset.py --turn-type pivot_left_90 --duration 8.0

  # Offline testing / simulation without physical hardware:
  python3 scripts/record_turn_dataset.py --mock --turn-type arc_right_fast
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

# Try importing hardware drivers; handle absence gracefully for mock/dev mode
HAS_HARDWARE = True
try:
    import board
    import busio
    import adafruit_bno055
    from rplidar import RPLidar
except (ImportError, NotImplementedError, AttributeError):
    HAS_HARDWARE = False


TURN_PRESETS = {
    "1": ("pivot_left_90", "In-place Pivot Turn (Left / CCW ~90 deg, zero forward speed)"),
    "2": ("pivot_right_90", "In-place Pivot Turn (Right / CW ~90 deg, zero forward speed)"),
    "3": ("pivot_180", "In-place Pivot Turn (180 deg U-turn)"),
    "4": ("arc_left_slow", "Forward Arcing Turn - Left (slow speed, R ~0.6m)"),
    "5": ("arc_right_slow", "Forward Arcing Turn - Right (slow speed, R ~0.6m)"),
    "6": ("arc_left_fast", "Forward Arcing Turn - Left (fast speed, R ~0.6m)"),
    "7": ("arc_right_fast", "Forward Arcing Turn - Right (fast speed, R ~0.6m)"),
    "8": ("corner_90_l_junction", "Sharp 90-degree Corner Navigation (forward -> turn -> forward)"),
    "9": ("figure_eight", "Continuous S-curve / Figure-8 maneuver"),
    "0": ("custom_turn", "Custom / Freeform maneuver"),
}


class TurnDatasetRecorder:
    def __init__(
        self,
        turn_type: str = "turn",
        lidar_port: str = "/dev/ttyUSB0",
        lidar_baud: int = 256000,
        output_dir: str = "data/turns",
        pre_dwell_sec: float = 3.0,
        post_dwell_sec: float = 3.0,
        mock_mode: bool = False,
    ):
        self.turn_type = turn_type
        self.lidar_port = lidar_port
        self.lidar_baud = lidar_baud
        self.output_dir = Path(output_dir)
        self.pre_dwell_sec = pre_dwell_sec
        self.post_dwell_sec = post_dwell_sec
        self.mock_mode = mock_mode or not HAS_HARDWARE

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.stop_requested = threading.Event()

        # In-memory record buffers
        self.imu_records = []
        self.lidar_records = []

        # Sensor hardware handles
        self.bno = None
        self.lidar = None

    def initialize_sensors(self):
        """Initialize BNO055 and RPLiDAR hardware or setup mock generators."""
        if self.mock_mode:
            print("[Recorder] Running in MOCK / SIMULATION mode (synthetic hardware).")
            return True

        print(f"[Recorder] Initializing BNO055 IMU on I2C bus...")
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.bno = adafruit_bno055.BNO055_I2C(i2c)
            # Test read
            _ = self.bno.gyro
            _ = self.bno.linear_acceleration
            print("[Recorder] BNO055 IMU connected successfully.")
        except Exception as e:
            print(f"[Recorder] ERROR connecting to BNO055: {type(e).__name__}: {e}")
            return False

        print(f"[Recorder] Initializing RPLiDAR on {self.lidar_port} at {self.lidar_baud} baud...")
        try:
            self.lidar = RPLidar(self.lidar_port, baudrate=self.lidar_baud, timeout=3)
            self.lidar.get_health = lambda: ("Good", 0)
            self.lidar.connect()
            self.lidar.stop_motor()
            self.lidar.start_motor()
            print("[Recorder] Waiting for LiDAR motor to reach steady operating speed (4s)...")
            time.sleep(4.0)
            self.lidar.clean_input()
            print("[Recorder] RPLiDAR ready and streaming.")
        except Exception as e:
            print(f"[Recorder] ERROR connecting to RPLiDAR: {type(e).__name__}: {e}")
            if self.lidar:
                try:
                    self.lidar.disconnect()
                except Exception:
                    pass
            return False

        return True

    def _read_bno_sample(self):
        """Fault-tolerant single read of all BNO055 sensors with clock-stretch recovery."""
        t_mono = time.monotonic()
        t_wall = time.time()
        try:
            gyro = self.bno.gyro
            lin_accel = self.bno.linear_acceleration
            raw_accel = self.bno.acceleration
            gravity = self.bno.gravity
            quat = self.bno.quaternion
            euler = self.bno.euler

            if gyro is None or lin_accel is None:
                return None

            return {
                "t_mono": t_mono,
                "t_wall": t_wall,
                "gyro": [float(x) for x in gyro],
                "lin_accel": [float(x) for x in lin_accel],
                "raw_accel": [float(x) for x in (raw_accel or [0, 0, 0])],
                "gravity": [float(x) for x in (gravity or [0, 0, 0])],
                "quaternion": [float(x) for x in (quat or [1, 0, 0, 0])],
                "euler": [float(x) for x in (euler or [0, 0, 0])],
            }
        except (OSError, RuntimeError, Exception):
            # Transient I2C clock-stretch or bus glitch; return None to retry
            return None

    def _imu_worker(self):
        """High-frequency IMU acquisition thread (~50-80 Hz)."""
        rate_hz = 60.0
        dt = 1.0 / rate_hz

        # Synthetic generator variables for mock mode
        sim_t = 0.0
        while self.is_recording and not self.stop_requested.is_set():
            t_loop_start = time.monotonic()

            if self.mock_mode:
                # Generate realistic synthetic data based on recording phase
                sample = self._generate_mock_imu(sim_t)
                sim_t += dt
                self.imu_records.append(sample)
            else:
                sample = self._read_bno_sample()
                if sample is not None:
                    self.imu_records.append(sample)

            elapsed = time.monotonic() - t_loop_start
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)

    def _lidar_worker(self):
        """LiDAR scan acquisition thread (~14.5 Hz)."""
        if self.mock_mode:
            sim_t = 0.0
            dt = 1.0 / 14.5
            while self.is_recording and not self.stop_requested.is_set():
                t_start = time.monotonic()
                time.sleep(dt)
                t_end = time.monotonic()
                scan_pts = self._generate_mock_lidar_scan(sim_t)
                sim_t += dt
                self.lidar_records.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "points": scan_pts,
                })
            return

        try:
            scans_iter = self.lidar.iter_scans(max_buf_meas=12000)
            prev_t = time.monotonic()
            for scan in scans_iter:
                if not self.is_recording or self.stop_requested.is_set():
                    break
                t_end = time.monotonic()
                t_start = prev_t
                prev_t = t_end

                # scan is list of (quality, angle_deg, distance_mm)
                # Store as compact float32 array
                pts_arr = np.array(
                    [(pt[0], pt[1], pt[2]) for pt in scan if pt[2] > 100],
                    dtype=np.float32,
                )
                self.lidar_records.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "points": pts_arr,
                })
        except Exception as e:
            if not self.stop_requested.is_set():
                print(f"[Recorder] LiDAR thread exception: {type(e).__name__}: {e}")

    def _generate_mock_imu(self, sim_t):
        """Produce synthetic turn kinematics for offline validation."""
        # Pre-dwell: 0 to pre_dwell
        # Turn active: pre_dwell to pre_dwell + 4.0
        # Post-dwell: pre_dwell + 4.0 onwards
        turn_start = self.pre_dwell_sec
        turn_end = self.pre_dwell_sec + 4.0

        w_z = 0.0
        a_x = 0.0
        a_y = 0.0
        if turn_start <= sim_t <= turn_end:
            # Trapeze angular velocity profile (peak ~0.78 rad/s / 45 deg/s)
            tau = (sim_t - turn_start) / 4.0
            w_z = 0.785 * math.sin(math.pi * tau)
            # Centrifugal accel = w_z * v (assume v=0.3 m/s)
            a_y = w_z * 0.30 + np.random.normal(0, 0.05)
            a_x = np.random.normal(0, 0.05)
        else:
            # Stationary motor vibration
            w_z = np.random.normal(0, 0.01)
            a_x = 0.25 * math.sin(2 * math.pi * 14.3 * sim_t) + np.random.normal(0, 0.05)
            a_y = 0.25 * math.cos(2 * math.pi * 14.3 * sim_t) + np.random.normal(0, 0.05)

        t_now = time.monotonic()
        return {
            "t_mono": t_now,
            "t_wall": time.time(),
            "gyro": [np.random.normal(0, 0.005), np.random.normal(0, 0.005), w_z],
            "lin_accel": [a_x, a_y, 0.0],
            "raw_accel": [a_x, a_y, 9.81],
            "gravity": [0.0, 0.0, 9.81],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "euler": [math.degrees(sim_t * 0.2), 0.0, 0.0],
        }

    def _generate_mock_lidar_scan(self, sim_t):
        """Produce synthetic rectangular room scan points."""
        angles = np.linspace(0, 360, 180, endpoint=False)
        pts = []
        for a in angles:
            rad = math.radians(a)
            # Box room: 4m x 4m
            # Distance to nearest wall
            cos_a = abs(math.cos(rad)) + 1e-5
            sin_a = abs(math.sin(rad)) + 1e-5
            d_x = 2000.0 / cos_a
            d_y = 2000.0 / sin_a
            dist = min(d_x, d_y) + np.random.normal(0, 15)
            pts.append((15.0, a, max(150.0, dist)))
        return np.array(pts, dtype=np.float32)

    def record_session(self, duration_sec: float = 10.0):
        """
        Execute the 3-phase guided recording protocol:
          Phase 1: Pre-dwell (stationary noise floor baseline)
          Phase 2: Active turn execution (real-time telemetry)
          Phase 3: Post-dwell (stopping deceleration and bias settling)
        """
        self.imu_records.clear()
        self.lidar_records.clear()
        self.is_recording = True
        self.stop_requested.clear()

        # Start sensor threads
        t_imu = threading.Thread(target=self._imu_worker, daemon=True)
        t_lidar = threading.Thread(target=self._lidar_worker, daemon=True)
        t_imu.start()
        t_lidar.start()

        time.sleep(0.5)  # Ensure threads are running

        print("\n" + "=" * 65)
        print(f" RECORDING PROTOCOL FOR TURN: '{self.turn_type.upper()}'")
        print("=" * 65)

        t_session_start = time.monotonic()

        # -------------------------------------------------------------
        # PHASE 1: Pre-turn dwell
        # -------------------------------------------------------------
        print(f"\n[PHASE 1/3] PRE-TURN DWELL ({self.pre_dwell_sec:.1f}s)")
        print(">>> KEEP PLATFORM COMPLETELY STATIONARY! <<<")
        dwell_start = time.monotonic()
        while time.monotonic() - dwell_start < self.pre_dwell_sec:
            rem = self.pre_dwell_sec - (time.monotonic() - dwell_start)
            sys.stdout.write(f"\r  Stationary baseline capturing... [{rem:3.1f}s remaining]")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r  Stationary baseline captured.                                \n")

        # -------------------------------------------------------------
        # PHASE 2: Active Turn Execution
        # -------------------------------------------------------------
        print("\n" + "#" * 65)
        print(f" [PHASE 2/3] ACTIVE MANEUVER: >>> EXECUTE TURN NOW! <<<")
        print(f" Target Duration: ~{duration_sec:.1f}s | Press [Ctrl+C] or [Enter] when turn is complete.")
        print("#" * 65 + "\n")

        turn_start_time = time.monotonic()
        accum_yaw_deg = 0.0
        last_t = turn_start_time
        last_report_t = turn_start_time

        try:
            while time.monotonic() - turn_start_time < duration_sec:
                now = time.monotonic()
                dt = now - last_t
                last_t = now

                # Real-time yaw integration from latest gyro reading
                if self.imu_records:
                    latest_gyro = self.imu_records[-1]["gyro"]
                    w_z = latest_gyro[2]
                    accum_yaw_deg += math.degrees(w_z * dt)
                else:
                    w_z = 0.0

                if now - last_report_t >= 0.5:
                    last_report_t = now
                    elapsed_turn = now - turn_start_time
                    imu_rate = len(self.imu_records) / (now - t_session_start)
                    lidar_rate = len(self.lidar_records) / (now - t_session_start)
                    pts_last = len(self.lidar_records[-1]["points"]) if self.lidar_records else 0

                    print(
                        f"  [Turn T+{elapsed_turn:4.1f}s] "
                        f"Yaw Rate: {math.degrees(w_z):+6.1f} deg/s | "
                        f"Turned: {accum_yaw_deg:+6.1f}° | "
                        f"IMU: {imu_rate:4.1f}Hz | "
                        f"LiDAR: {lidar_rate:4.1f}Hz ({pts_last} pts)",
                        flush=True,
                    )
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n  Turn phase ended by user.")

        turn_end_time = time.monotonic()
        actual_turn_duration = turn_end_time - turn_start_time

        # -------------------------------------------------------------
        # PHASE 3: Post-turn dwell
        # -------------------------------------------------------------
        print(f"\n[PHASE 3/3] POST-TURN DWELL ({self.post_dwell_sec:.1f}s)")
        print(">>> STOP PLATFORM - KEEP COMPLETELY STILL! <<<")
        post_start = time.monotonic()
        while time.monotonic() - post_start < self.post_dwell_sec:
            rem = self.post_dwell_sec - (time.monotonic() - post_start)
            sys.stdout.write(f"\r  Settling capture... [{rem:3.1f}s remaining]")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r  Post-turn baseline captured.                                 \n")

        # Stop recording
        self.is_recording = False
        self.stop_requested.set()
        t_imu.join(timeout=1.0)
        t_lidar.join(timeout=1.0)

        total_session_duration = time.monotonic() - t_session_start
        print(f"\n[Recorder] Session finished. Total time: {total_session_duration:.2f}s "
              f"(Turn: {actual_turn_duration:.2f}s).")

        return self.save_dataset(
            turn_start_time=turn_start_time,
            turn_end_time=turn_end_time,
            accum_yaw_deg=accum_yaw_deg,
        )

    def save_dataset(self, turn_start_time: float, turn_end_time: float, accum_yaw_deg: float):
        """Save recorded buffers to compressed .npz archive, metadata JSON, and preview PNG."""
        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_name = f"turn_{self.turn_type}_{timestamp_str}"
        npz_path = self.output_dir / f"{base_name}.npz"
        meta_path = self.output_dir / f"{base_name}_meta.json"
        plot_path = self.output_dir / f"{base_name}_preview.png"

        print(f"\n[Recorder] Packaging dataset: {base_name}...")
        print(f"  IMU samples captured:   {len(self.imu_records)}")
        print(f"  LiDAR scans captured: {len(self.lidar_records)}")

        if not self.imu_records or not self.lidar_records:
            print("[Recorder] ERROR: Insufficient sensor data captured to save dataset!")
            return None

        # 1. Format IMU arrays
        imu_t_mono = np.array([r["t_mono"] for r in self.imu_records], dtype=np.float64)
        imu_t_wall = np.array([r["t_wall"] for r in self.imu_records], dtype=np.float64)
        imu_gyro = np.array([r["gyro"] for r in self.imu_records], dtype=np.float32)
        imu_lin_accel = np.array([r["lin_accel"] for r in self.imu_records], dtype=np.float32)
        imu_raw_accel = np.array([r["raw_accel"] for r in self.imu_records], dtype=np.float32)
        imu_gravity = np.array([r["gravity"] for r in self.imu_records], dtype=np.float32)
        imu_quat = np.array([r["quaternion"] for r in self.imu_records], dtype=np.float32)
        imu_euler = np.array([r["euler"] for r in self.imu_records], dtype=np.float32)

        # 2. Format LiDAR arrays (flattened points + scan boundaries for zero-pickle portability)
        lidar_t_start = np.array([r["t_start"] for r in self.lidar_records], dtype=np.float64)
        lidar_t_end = np.array([r["t_end"] for r in self.lidar_records], dtype=np.float64)

        all_points = []
        scan_offsets = [0]
        for r in self.lidar_records:
            pts = r["points"]
            if len(pts) > 0:
                all_points.append(pts)
                scan_offsets.append(scan_offsets[-1] + len(pts))
            else:
                scan_offsets.append(scan_offsets[-1])

        lidar_points = np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=np.float32)
        lidar_scan_offsets = np.array(scan_offsets, dtype=np.int64)

        # 3. Save compressed NumPy archive
        np.savez_compressed(
            npz_path,
            # IMU channels
            imu_t_mono=imu_t_mono,
            imu_t_wall=imu_t_wall,
            imu_gyro=imu_gyro,
            imu_lin_accel=imu_lin_accel,
            imu_raw_accel=imu_raw_accel,
            imu_gravity=imu_gravity,
            imu_quaternion=imu_quat,
            imu_euler=imu_euler,
            # LiDAR channels
            lidar_t_start=lidar_t_start,
            lidar_t_end=lidar_t_end,
            lidar_points=lidar_points,
            lidar_scan_offsets=lidar_scan_offsets,
            # Phase boundaries
            turn_start_time=turn_start_time,
            turn_end_time=turn_end_time,
        )
        print(f"  [SAVED] NumPy archive:   {npz_path} ({npz_path.stat().st_size / 1024:.1f} KB)")

        # 4. Save metadata JSON
        duration_total = float(imu_t_mono[-1] - imu_t_mono[0])
        imu_hz = len(imu_t_mono) / duration_total if duration_total > 0 else 0.0
        lidar_hz = len(lidar_t_end) / duration_total if duration_total > 0 else 0.0

        meta = {
            "dataset_name": base_name,
            "recorded_at": datetime.now().isoformat(),
            "turn_type": self.turn_type,
            "mock_mode": self.mock_mode,
            "total_duration_sec": round(duration_total, 3),
            "turn_duration_sec": round(float(turn_end_time - turn_start_time), 3),
            "pre_dwell_sec": self.pre_dwell_sec,
            "post_dwell_sec": self.post_dwell_sec,
            "integrated_yaw_deg": round(accum_yaw_deg, 2),
            "imu_sample_count": len(imu_t_mono),
            "imu_effective_hz": round(imu_hz, 2),
            "lidar_scan_count": len(lidar_t_end),
            "lidar_point_count": len(lidar_points),
            "lidar_effective_hz": round(lidar_hz, 2),
            "gyro_z_peak_deg_s": round(float(np.max(np.abs(np.degrees(imu_gyro[:, 2])))), 2),
            "lateral_accel_peak_m_s2": round(float(np.max(np.abs(imu_lin_accel[:, 1]))), 3),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [SAVED] Metadata JSON:   {meta_path}")

        # 5. Generate preview plot if matplotlib is available
        self._generate_preview_plot(
            plot_path=plot_path,
            imu_t_mono=imu_t_mono,
            imu_gyro=imu_gyro,
            imu_lin_accel=imu_lin_accel,
            lidar_points=lidar_points,
            lidar_scan_offsets=lidar_scan_offsets,
            turn_start_time=turn_start_time,
            turn_end_time=turn_end_time,
            base_name=base_name,
        )

        return npz_path

    def _generate_preview_plot(
        self,
        plot_path,
        imu_t_mono,
        imu_gyro,
        imu_lin_accel,
        lidar_points,
        lidar_scan_offsets,
        turn_start_time,
        turn_end_time,
        base_name,
    ):
        """Generate a 4-panel diagnostic figure to quickly verify turning dynamics."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [INFO] matplotlib not installed; skipping preview plot generation.")
            return

        try:
            t_rel = imu_t_mono - imu_t_mono[0]
            t_start_rel = turn_start_time - imu_t_mono[0]
            t_end_rel = turn_end_time - imu_t_mono[0]

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle(f"Turn Dataset Diagnostic: {base_name}", fontsize=14, fontweight="bold")

            # 1. Angular Velocities & Integrated Heading
            ax1 = axes[0, 0]
            ax1.plot(t_rel, np.degrees(imu_gyro[:, 2]), label=r"Yaw Rate $\omega_z$ (deg/s)", color="crimson", lw=1.5)
            ax1.plot(t_rel, np.degrees(imu_gyro[:, 0]), label=r"Roll Rate $\omega_x$", color="gray", lw=0.8, alpha=0.6)
            ax1.plot(t_rel, np.degrees(imu_gyro[:, 1]), label=r"Pitch Rate $\omega_y$", color="lightblue", lw=0.8, alpha=0.6)
            ax1.axvspan(t_start_rel, t_end_rel, color="gold", alpha=0.15, label="Active Turn Window")
            ax1.set_title("Gyroscope Angular Velocities")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Rate (deg/s)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right", fontsize=8)

            # Secondary axis for integrated yaw
            ax1_twin = ax1.twinx()
            yaw_integrated = np.cumsum(imu_gyro[:, 2] * np.gradient(t_rel))
            ax1_twin.plot(t_rel, np.degrees(yaw_integrated), color="navy", lw=2, linestyle="--", label="Integrated Yaw (deg)")
            ax1_twin.set_ylabel("Total Heading (deg)", color="navy")
            ax1_twin.tick_params(axis="y", labelcolor="navy")

            # 2. Linear Accelerations & Centrifugal Force
            ax2 = axes[0, 1]
            ax2.plot(t_rel, imu_lin_accel[:, 0], label=r"Forward $a_x$ (m/s²)", color="forestgreen", lw=1.2)
            ax2.plot(t_rel, imu_lin_accel[:, 1], label=r"Lateral $a_y$ (m/s²)", color="darkorange", lw=1.5)
            ax2.axvspan(t_start_rel, t_end_rel, color="gold", alpha=0.15)
            ax2.set_title("Linear Acceleration (Gravity Removed)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Acceleration (m/s²)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right", fontsize=8)

            # 3. Lateral Acceleration vs Yaw Rate Scatter (Checks Lever Arm / Centrifugal coupling)
            ax3 = axes[1, 0]
            # Only during active turn window
            mask = (t_rel >= t_start_rel) & (t_rel <= t_end_rel)
            if np.any(mask):
                w_turn = imu_gyro[mask, 2]
                ay_turn = imu_lin_accel[mask, 1]
                ax3.scatter(np.degrees(w_turn), ay_turn, c=t_rel[mask], cmap="viridis", s=15, alpha=0.7)
                ax3.set_title(r"Lateral Coupling: $a_y$ vs $\omega_z$ (Color = Time)")
                ax3.set_xlabel(r"Yaw Rate $\omega_z$ (deg/s)")
                ax3.set_ylabel(r"Lateral Acceleration $a_y$ (m/s²)")
                ax3.grid(True, alpha=0.3)

            # 4. Top-Down LiDAR Scan Overlay (Pre-Turn vs Mid-Turn vs Post-Turn)
            ax4 = axes[1, 1]
            num_scans = len(lidar_scan_offsets) - 1
            if num_scans >= 3:
                scans_to_plot = [
                    (0, "Pre-Turn Scan", "blue"),
                    (num_scans // 2, "Mid-Turn Scan", "darkorange"),
                    (num_scans - 1, "Post-Turn Scan", "red"),
                ]
                for scan_idx, label, color in scans_to_plot:
                    s_idx = lidar_scan_offsets[scan_idx]
                    e_idx = lidar_scan_offsets[scan_idx + 1]
                    pts = lidar_points[s_idx:e_idx]
                    if len(pts) > 0:
                        angles_deg = pts[:, 1]
                        ranges_m = pts[:, 2] / 1000.0
                        angles_rep103 = np.radians((360.0 - angles_deg) % 360.0)
                        xs = ranges_m * np.cos(angles_rep103)
                        ys = ranges_m * np.sin(angles_rep103)
                        ax4.scatter(xs, ys, s=2, c=color, alpha=0.6, label=label)

            ax4.set_title("2D LiDAR Point Cloud Overlay (Sensor Frame)")
            ax4.set_xlabel("X Forward (m)")
            ax4.set_ylabel("Y Left (m)")
            ax4.axis("equal")
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"  [SAVED] Preview plot:    {plot_path}")
        except Exception as e:
            print(f"  [WARN] Failed to generate preview plot: {type(e).__name__}: {e}")

    def cleanup(self):
        """Clean teardown of hardware peripherals."""
        if self.lidar:
            print("[Recorder] Stopping RPLiDAR motor and closing serial port...")
            for step in (self.lidar.stop, self.lidar.stop_motor, self.lidar.disconnect):
                try:
                    step()
                except Exception:
                    pass
        print("[Recorder] Teardown complete.")


def interactive_menu():
    """Present interactive menu to select the turn maneuver."""
    print("\n" + "=" * 65)
    print(" SELECT TURNING MANEUVER TO RECORD:")
    print("=" * 65)
    for key, (tag, desc) in TURN_PRESETS.items():
        print(f"  [{key}] {desc}")
    print("=" * 65)

    choice = input("Enter choice [1-9, 0] (default 1): ").strip()
    if choice not in TURN_PRESETS:
        choice = "1"

    tag, _ = TURN_PRESETS[choice]
    dur_str = input("Target turn duration in seconds (default 8.0s): ").strip()
    try:
        dur = float(dur_str) if dur_str else 8.0
    except ValueError:
        dur = 8.0

    return tag, dur


def main():
    parser = argparse.ArgumentParser(description="Record synchronized IMU and LiDAR data during robot turns.")
    parser.add_argument("--turn-type", type=str, default=None, help="Tag for the turn maneuver (e.g. pivot_left_90)")
    parser.add_argument("--duration", type=float, default=8.0, help="Target turn duration in seconds (default: 8.0)")
    parser.add_argument("--pre-dwell", type=float, default=3.0, help="Stationary baseline seconds before turn (default: 3.0)")
    parser.add_argument("--post-dwell", type=float, default=3.0, help="Stationary baseline seconds after turn (default: 3.0)")
    parser.add_argument("--lidar-port", type=str, default="/dev/ttyUSB0", help="LiDAR serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--lidar-baud", type=int, default=256000, help="LiDAR baudrate (default: 256000)")
    parser.add_argument("--output-dir", type=str, default="data/turns", help="Directory to save datasets (default: data/turns)")
    parser.add_argument("--mock", action="store_true", help="Simulate hardware for testing without physical sensors")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive menu even if args provided")

    args = parser.parse_args()

    # Determine turn type and duration
    if args.turn_type is None or args.interactive:
        turn_type, duration = interactive_menu()
    else:
        turn_type = args.turn_type
        duration = args.duration

    recorder = TurnDatasetRecorder(
        turn_type=turn_type,
        lidar_port=args.lidar_port,
        lidar_baud=args.lidar_baud,
        output_dir=args.output_dir,
        pre_dwell_sec=args.pre_dwell,
        post_dwell_sec=args.post_dwell,
        mock_mode=args.mock,
    )

    # Handle Ctrl+C gracefully
    def _sig_handler(sig, frame):
        print("\n[Recorder] Stop signal received...")
        recorder.stop_requested.set()
        recorder.is_recording = False

    signal.signal(signal.SIGINT, _sig_handler)

    try:
        ok = recorder.initialize_sensors()
        if not ok:
            print("[Recorder] Hardware initialization failed. Aborting.")
            return 1

        recorder.record_session(duration_sec=duration)
    finally:
        recorder.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
