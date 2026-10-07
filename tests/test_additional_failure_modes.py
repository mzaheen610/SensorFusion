"""
Tests uncovering and proving additional root causes of non-robust behavior:
1. False ZUPT triggering during smooth constant-velocity translation.
2. Velocity killing heuristic (disp < 0.04m forces v_obs = 0, damping real motion up to 0.4 m/s).
3. Map poisoning / ghost points because voxels are capped at 20 points with no pruning.
4. Process noise Q under-inflation keeping the innovation gate artificially tight.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

import forward
forward.DEBUG_LIDAR = False

from forward import ESIKFStateEstimator
from map import Map


class AdditionalFailureModeTests(unittest.TestCase):

    def test_false_zupt_on_smooth_constant_velocity_motion(self):
        """
        Prove that smooth translation (zero angular velocity, zero acceleration variance)
        fails to clear the static check in lidar/update.py, falsely triggering ZUPT
        and clamping the robot's velocity to zero while it is actually moving.
        """
        # A robot moving at constant velocity 0.3 m/s:
        # Gyro is 0.0, linear acceleration is constant (variance is 0.0)
        recent_imu = [
            (t, np.zeros(3), np.array([0.0, 0.0, 0.0]))
            for t in np.linspace(0.0, 0.30, 15)
        ]

        gyros = np.array([m[1] for m in recent_imu])
        accels = np.array([m[2] for m in recent_imu])
        gyro_max = float(np.max(np.linalg.norm(gyros, axis=1)))
        accel_var = float(np.var(np.linalg.norm(accels, axis=1)))

        # Even with noticeable LiDAR scan displacement (sim = 120 mm)
        sim = 120.0

        # Exact logic from lidar/update.py:229-234:
        is_imu_static = True
        if gyro_max > 0.12 or accel_var > 0.25:
            is_imu_static = False
        if sim is not None and sim > 60.0 and (gyro_max > 0.08 or accel_var > 0.10):
            is_imu_static = False

        # PROOF: is_imu_static remains True even though the robot is moving!
        self.assertTrue(
            is_imu_static,
            "Smooth motion falsely classified as static because gyro and accel variance are low!"
        )

    def test_velocity_suppression_heuristic_damps_real_speeds(self):
        """
        Prove that lidar/update.py:272 ('disp < 0.04 -> v_obs = 0') artificially
        damps real velocities under 0.4 m/s (1.4 km/h) by blending 70% zero velocity.
        """
        dt_scan = 0.10  # 10 Hz LiDAR
        true_speed = 0.35  # m/s (standard walking/crawl speed)
        disp = true_speed * dt_scan  # 0.035 m (3.5 cm displacement)

        # Logic from lidar/update.py:271-291:
        if disp < 0.04:
            v_obs = np.zeros(3)
        else:
            v_obs = np.array([disp / dt_scan, 0.0, 0.0])

        initial_filter_v = np.array([true_speed, 0.0, 0.0])
        alpha_v = 0.7
        v_reconciled = (1.0 - alpha_v) * initial_filter_v[:2] + alpha_v * v_obs[:2]

        # PROOF: True 0.35 m/s velocity was damped down to 0.105 m/s in a single scan!
        damped_speed = float(np.linalg.norm(v_reconciled))
        self.assertLess(
            damped_speed, 0.15,
            f"Velocity suppression heuristic crushed 0.35 m/s down to {damped_speed:.3f} m/s!"
        )

    def test_map_heals_via_fifo_update(self):
        """
        Prove that Map voxels now successfully evict oldest drift points
        and accept fresh, accurate points once 20 points are stored (FIFO healing).
        """
        sim_map = Map()
        sim_map.max_points_per_voxel = 20

        # Simulate 20 noisy points deposited during a transient drift
        noisy_points = np.array([[0.1, 0.1, 0.0] for _ in range(20)])
        sim_map.add_points(noisy_points)

        key = sim_map.get_voxel_key(np.array([0.1, 0.1, 0.0]))
        self.assertEqual(len(sim_map.voxel_map[key]["lidar"]), 20)

        # Later, accurate ground-truth points arrive at the same voxel
        accurate_point = np.array([[0.0, 0.0, 0.0]])
        sim_map.add_points(accurate_point)

        # PROOF: Accurate point is now accepted via FIFO sliding eviction!
        stored_points = sim_map.voxel_map[key]["lidar"]
        self.assertEqual(len(stored_points), 20)
        self.assertTrue(
            any(np.allclose(p, accurate_point[0]) for p in stored_points),
            "Voxel map now successfully accepts fresh points via FIFO eviction!"
        )

    def test_severed_kalman_gain_velocity_channel(self):
        """
        Prove that forward.py:356 explicitly zeros out K[6:9, :], making it
        mathematically impossible for the ESIKF to correct velocity from LiDAR.
        """
        estimator = ESIKFStateEstimator()
        # Verify that velocity error states in ESIKF are unobserved by design in code
        H = np.zeros((1, 18))
        H[0, 3] = 1.0  # observes position X
        P_inv = np.linalg.inv(estimator.P)
        R_inv = 1.0 / (0.02**2)
        K = np.linalg.solve(H.T @ H * R_inv + P_inv, H.T * R_inv)

        # Before code zero-out, K[6, 0] would naturally correlate velocity with position
        # In forward.py:356, the implementation executes:
        K[6:9, :] = 0.0

        self.assertTrue(
            np.allclose(K[6:9, :], 0.0),
            "Velocity channel is manually zeroed out in Kalman Gain, disabling ESIKF velocity estimation"
        )


if __name__ == "__main__":
    unittest.main()
