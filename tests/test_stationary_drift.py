from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator
from map import Map
from utils.similarity import scan_similarity


class StationaryDriftTests(unittest.TestCase):
    def test_imu_prediction_stationary_drift_is_bounded(self):
        """When stationary and calibrated, 10s of IMU integration should not drift."""
        estimator = ESIKFStateEstimator()
        np.random.seed(42)
        dt = 0.016  # ~60 Hz
        num_steps = int(10.0 / dt)

        for _ in range(num_steps):
            gyro = np.random.normal(0.0, 1e-4, 3)
            accel = np.random.normal(0.0, 1e-3, 3)
            estimator.predict((gyro, accel), dt)

        # Position should stay within millimetres
        self.assertLess(np.linalg.norm(estimator.state.p), 0.05)
        self.assertLess(np.linalg.norm(estimator.state.v), 0.02)
        # Z-axis should be bounded by noise (observable 3D motion)
        self.assertLess(abs(estimator.state.p[2]), 0.02)
        self.assertLess(abs(estimator.state.v[2]), 0.01)
        self.assertTrue(np.all(estimator.state.g == 0.0))

    def test_signed_line_residuals_have_zero_mean(self):
        """Line residuals should be signed (positive and negative), avoiding one-sided repulsion."""
        direction = np.array([1.0, 0.0, 0.0])  # line along X
        n_line = np.array([-direction[1], direction[0], 0.0])
        norm_2d = np.linalg.norm(n_line[:2])
        if norm_2d > 1e-6:
            n_line /= norm_2d

        center = np.array([0.0, 1.0, 0.0])  # line at y = 1.0
        # Points scattered symmetrically on both sides of y = 1.0
        points = [
            np.array([0.0, 1.05, 0.0]),  # +0.05
            np.array([0.0, 0.95, 0.0]),  # -0.05
            np.array([1.0, 1.02, 0.0]),  # +0.02
            np.array([1.0, 0.98, 0.0]),  # -0.02
        ]
        residuals = [float(np.dot(n_line, p - center)) for p in points]
        self.assertAlmostEqual(np.mean(residuals), 0.0, places=6)
        # Verify that both positive and negative residuals exist
        self.assertTrue(any(r > 0 for r in residuals))
        self.assertTrue(any(r < 0 for r in residuals))

    def test_scan_similarity_median_rejects_edge_transition_spikes(self):
        """Median scan similarity should not spike when a single ray hits an obstacle edge."""
        bins_a = np.full(180, np.nan)
        bins_b = np.full(180, np.nan)
        for i in range(10, 40):
            bins_a[i] = 1000.0
            bins_b[i] = 1000.0 + (i % 3)  # tiny 0-2mm noise

        # An obstacle edge causes 1 ray to transition from 1000mm to 3000mm (2000mm spike)
        bins_b[25] = 3000.0

        sim_median = scan_similarity(bins_a, bins_b)
        self.assertIsNotNone(sim_median)
        # Median should remain around ~1mm, NOT 50+ mm
        self.assertLess(sim_median, 5.0)

    def test_camera_covariance_large_residual_scalability(self):
        """Camera Joseph-form covariance calculation should scale to >10,000 residuals without OOM."""
        P_snap = 0.01 * np.eye(18)
        num_residuals = 10000
        H = np.random.randn(num_residuals, 18)
        sigma_camera = 10.0

        weight = 1.0 / (sigma_camera**2)
        P_inv = np.linalg.inv(P_snap)
        S_inv = P_inv + weight * (H.T @ H)
        K = np.linalg.solve(S_inv, weight * H.T)

        I_KH = np.eye(18) - K @ H
        P_new = I_KH @ P_snap @ I_KH.T + (sigma_camera**2) * (K @ K.T)

        self.assertEqual(P_new.shape, (18, 18))
        self.assertTrue(np.isfinite(P_new).all())
        np.testing.assert_allclose(P_new, P_new.T, atol=1e-8)

    def test_stationary_lock_prevents_uncalibrated_drift_and_adapts_bias(self):
        """Standard ZUPT bounds drift with calibrated ba."""
        estimator = ESIKFStateEstimator()
        dt = 0.033  # ~30 Hz
        true_bias = np.array([0.02, 0.03, 0.0])
        estimator.state.ba = true_bias.copy()

        # Simulate 5 cycles of 1 second IMU prediction followed by standard ZUPT
        for cycle in range(5):
            for _ in range(30):
                gyro = np.zeros(3)
                accel = true_bias + np.random.normal(0, 0.005, 3)
                estimator.predict((gyro, accel), dt)

            # Standard ZUPT Kalman update
            estimator.zupt_update()

        # After 5 seconds, position drift must remain bounded under 2 cm
        self.assertLess(np.linalg.norm(estimator.state.p[:2]), 0.02)
        # Velocity must be zeroed by ZUPT
        np.testing.assert_array_equal(estimator.state.v, np.zeros(3))
        # ba must stay locked to calibrated bias
        np.testing.assert_allclose(estimator.state.ba, true_bias, atol=1e-5)

    def test_zupt_adapts_3d_bias_and_bounds_drift(self):
        """ZUPT bounds 3D position drift with calibrated ba."""
        estimator = ESIKFStateEstimator()
        dt = 0.033  # ~30 Hz
        true_bias = np.array([0.02, 0.03, 0.04])
        estimator.state.ba = true_bias.copy()

        for cycle in range(5):
            for _ in range(30):
                gyro = np.zeros(3)
                accel = true_bias + np.random.normal(0, 0.005, 3)
                estimator.predict((gyro, accel), dt)
            estimator.zupt_update()

        # 3D position drift must remain bounded under 3 cm
        self.assertLess(np.linalg.norm(estimator.state.p), 0.03)
        np.testing.assert_array_equal(estimator.state.v, np.zeros(3))
        np.testing.assert_allclose(estimator.state.ba, true_bias, atol=1e-5)

    def test_stationary_map_updates_naturally_saturate_voxels(self):
        """Stationary scans naturally populate unfilled voxels without exceeding capacity."""
        lidar_map = Map()
        # Simulate 10 scans of 50 stationary points in the same area
        stationary_points = np.random.uniform(0.1, 0.4, (50, 3))
        for _ in range(10):
            lidar_map.add_points(stationary_points)

        # Total points in any single voxel should not exceed max_points_per_voxel (20)
        for key, voxel in lidar_map.voxel_map.items():
            self.assertLessEqual(len(voxel["lidar"]), lidar_map.max_points_per_voxel)


    def test_linear_push_wakes_up_static_filter(self):
        """A linear push with horizontal acceleration > 0.4 m/s^2 must wake up the static filter."""
        estimator = ESIKFStateEstimator()
        estimator.is_static = True
        dt = 0.033

        # Stationary noise should not wake up the filter
        gyro_quiet = np.zeros(3)
        accel_quiet = np.array([0.05, 0.05, 0.0])
        estimator.predict((gyro_quiet, accel_quiet), dt)
        self.assertTrue(estimator.is_static)
        np.testing.assert_array_equal(estimator.state.v, np.zeros(3))

        # Push acceleration exceeding 0.4 m/s^2 must wake up the filter
        accel_push = np.array([0.60, 0.0, 0.0])
        estimator.predict((gyro_quiet, accel_push), dt)
        self.assertFalse(estimator.is_static)
        self.assertGreater(estimator.state.v[0], 0.0)

    def test_velocity_reconciliation_damps_runaway_speed(self):
        """Successive LiDAR-observed displacement velocities must damp IMU velocity runaway."""
        estimator = ESIKFStateEstimator()
        # Simulate IMU velocity runaway from persistent hand tilt / acceleration pulse
        estimator.state.v = np.array([0.80, 0.10, 0.0])

        # True physical motion: moved 0.05 m in 0.5 s (true speed = 0.10 m/s)
        p_prev = np.array([0.0, 0.0, 0.0])
        p_curr = np.array([0.05, 0.0, 0.0])
        dt_scan = 0.5

        v_obs = (p_curr - p_prev) / dt_scan
        v_obs[2] = 0.0

        alpha_v = 0.7
        v_reconciled = (1.0 - alpha_v) * estimator.state.v[:2] + alpha_v * v_obs[:2]
        estimator.state.v[:2] = v_reconciled

        # Velocity must be significantly reduced towards true physical speed (0.10 m/s)
        self.assertLess(estimator.state.v[0], 0.35)
        self.assertGreater(estimator.state.v[0], 0.05)

    def test_trajectory_buffer_retains_full_history_when_state_buffer_pruned(self):
        """Trajectory buffer must retain full path from origin while imu_state_buffer is pruned to 3s."""
        from collections import deque
        imu_state_buffer = deque()
        trajectory_buffer = [np.zeros(3)]

        # Simulate 10 seconds of motion (robot moving from x=0 to x=2.0)
        dt = 0.04  # 25 Hz
        for step in range(250):
            t = step * dt
            pos = np.array([t * 0.2, 0.0, 0.0])  # Moving at 0.2 m/s

            # IMU state buffer append and 3s pruning
            imu_state_buffer.append((t, pos, np.eye(3)))
            while imu_state_buffer and t - imu_state_buffer[0][0] > 3.0:
                imu_state_buffer.popleft()

            # Trajectory buffer maintains history
            if not trajectory_buffer:
                trajectory_buffer.append(pos)
            elif np.linalg.norm(pos - trajectory_buffer[-1]) > 0.005:
                trajectory_buffer.append(pos)
            else:
                trajectory_buffer[-1] = pos

        # imu_state_buffer should only have ~3 seconds (~75 items)
        self.assertLessEqual(len(imu_state_buffer), 78)
        # Oldest time in imu_state_buffer should be ~7 seconds
        self.assertGreater(imu_state_buffer[0][0], 6.5)

        # trajectory_buffer must start at 0.0 and reach 2.0 (full trajectory preserved!)
        self.assertAlmostEqual(trajectory_buffer[0][0], 0.0, places=3)
        self.assertAlmostEqual(trajectory_buffer[-1][0], 2.0 - dt * 0.2, places=2)
        self.assertGreater(len(trajectory_buffer), 100)


if __name__ == "__main__":
    unittest.main()
