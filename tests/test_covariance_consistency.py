from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator
from map import Map


class CovarianceConsistencyTests(unittest.TestCase):
    def test_zupt_row_zeroing_preserves_unobservable_covariances(self):
        """ZUPT must not shrink position, attitude, or gyro bias covariance."""
        estimator = ESIKFStateEstimator()
        # Set distinctive initial covariances
        estimator.P = np.diag([
            0.05**2, 0.05**2, 0.05**2,  # rot
            0.10**2, 0.10**2, 0.10**2,  # pos
            0.20**2, 0.20**2, 0.20**2,  # vel
            0.01**2, 0.01**2, 0.01**2,  # bg
            0.05**2, 0.05**2, 0.05**2,  # ba
            0.01**2, 0.01**2, 0.01**2,  # g
        ])
        # Inject velocity residual
        estimator.state.v = np.array([0.15, -0.10, 0.0])
        P_before = estimator.P.copy()

        estimator.zupt_update(sigma_zupt=0.05)

        # Attitude, position, gyro bias, and accel bias covariances must be identical (not shrunk)
        np.testing.assert_allclose(estimator.P[0:3, 0:3], P_before[0:3, 0:3], atol=1e-12)
        np.testing.assert_allclose(estimator.P[3:6, 3:6], P_before[3:6, 3:6], atol=1e-12)
        np.testing.assert_allclose(estimator.P[9:12, 9:12], P_before[9:12, 9:12], atol=1e-12)
        np.testing.assert_allclose(estimator.P[12:15, 12:15], P_before[12:15, 12:15], atol=1e-12)
        np.testing.assert_allclose(estimator.P[15:18, 15:18], P_before[15:18, 15:18], atol=1e-12)

        # Velocity covariance must be reduced by measurement
        self.assertLess(np.trace(estimator.P[6:9, 6:9]), np.trace(P_before[6:9, 6:9]))

        # Velocity must be zeroed
        np.testing.assert_allclose(estimator.state.v, np.zeros(3))

    def test_zupt_adapts_bias_in_residual_direction_bounded(self):
        """When velocity accumulates from persistent bias, ZUPT must adapt ba towards true bias."""
        estimator = ESIKFStateEstimator()
        dt = 0.01  # 100 Hz
        true_bias = np.array([0.10, -0.08, 0.0])
        estimator.state.ba = np.zeros(3)

        # Integrate IMU with constant acceleration (simulating unmodeled bias)
        for _ in range(50):
            estimator.predict((np.zeros(3), true_bias), dt)

        self.assertGreater(np.linalg.norm(estimator.state.v[:2]), 0.03)

        ba_before = estimator.state.ba.copy()
        estimator.zupt_update(sigma_zupt=0.05, adapt_ba=True)

        # ba must have adapted in the direction of true_bias
        delta_ba = estimator.state.ba - ba_before
        self.assertGreater(np.dot(delta_ba[:2], true_bias[:2]), 0.0)

        # Single step must obey slew limit (<= 0.05 m/s^2)
        self.assertLessEqual(np.max(np.abs(delta_ba)), 0.05 + 1e-6)

    def test_lidar_row_zeroing_preserves_accel_and_gyro_bias_covariance(self):
        """LiDAR updates must not artificially shrink ba or bg covariance."""
        estimator = ESIKFStateEstimator()
        estimator.P = np.diag([
            0.05**2, 0.05**2, 0.05**2,
            0.10**2, 0.10**2, 0.10**2,
            0.20**2, 0.20**2, 0.20**2,
            0.01**2, 0.01**2, 0.01**2,
            0.05**2, 0.05**2, 0.05**2,
            0.01**2, 0.01**2, 0.01**2,
        ])
        lidar_map = Map()
        wall_points = np.asarray([[x, 2.0, z] for x in np.linspace(-1, 1, 10) for z in np.linspace(-0.2, 0.2, 5)])
        lidar_map.add_points(wall_points)

        scan = [(15, 90.0, 2000.0)]
        points = np.asarray([[x, 1.95, 0.0] for x in np.linspace(-0.5, 0.5, 8)])
        state = estimator.state

        P_before = estimator.P.copy()
        _, applied, P_new = estimator.lidar_update(
            scan, state, estimator.P.copy(), points, lidar_map
        )

        if applied:
            np.testing.assert_allclose(P_new[6:9, 6:9], P_before[6:9, 6:9], atol=1e-12)
            np.testing.assert_allclose(P_new[9:12, 9:12], P_before[9:12, 9:12], atol=1e-12)
            np.testing.assert_allclose(P_new[12:15, 12:15], P_before[12:15, 12:15], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
