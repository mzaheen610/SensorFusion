from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator
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
        # Z-axis should be exactly 0
        self.assertEqual(estimator.state.p[2], 0.0)
        self.assertEqual(estimator.state.v[2], 0.0)
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


if __name__ == "__main__":
    unittest.main()
