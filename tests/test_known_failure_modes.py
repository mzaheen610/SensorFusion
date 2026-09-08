from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator
from lidar.backward import backprop
from map import Map



def make_pose(position=None, velocity=None):
    return SimpleNamespace(
        R=np.eye(3),
        p=np.zeros(3) if position is None else np.asarray(position, dtype=float),
        v=np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
        bg=np.zeros(3),
        ba=np.zeros(3),
        g=np.zeros(3),
    )


class KnownFailureModeTests(unittest.TestCase):
    def test_scan_wrap_should_follow_measurement_order(self):
        """A wrapped scan must not assign time from absolute angle."""
        pose = make_pose(velocity=[1.0, 0.0, 0.0])
        scan = [
            (15, 358.0, 1000.0),
            (15, 0.0, 1000.0),
            (15, 2.0, 1000.0),
        ]
        compensated = backprop(
            10.10,
            10.0,
            pose,
            scan,
            [(10.0, np.zeros(3), np.zeros(3))],
        )

        # Remove the static angular geometry and inspect only the translation.
        static_points = np.asarray([
            [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0]
            for angle in [358.0, 0.0, 2.0]
        ])
        translation = compensated - static_points
        self.assertTrue(np.all(np.diff(translation[:, 0]) >= -1e-8))

    def test_multiple_wrapped_scans_have_continuous_compensation(self):
        pose = make_pose(velocity=[0.8, -0.2, 0.0])
        wrapped_angles = [357.0, 359.0, 0.5, 2.0, 4.0]

        for scan_period in (0.06, 0.10, 0.18, 0.22):
            scan = [(15, angle, 1_000.0) for angle in wrapped_angles]
            imu_buffer = [
                (20.0 + offset, np.zeros(3), np.zeros(3))
                for offset in np.arange(0.0, scan_period, 0.02)
            ]
            compensated = backprop(
                20.0 + scan_period,
                20.0,
                pose,
                scan,
                imu_buffer,
            )
            self.assertEqual(compensated.shape, (len(scan), 3))
            self.assertLess(np.max(np.linalg.norm(np.diff(compensated, axis=0), axis=1)), 0.1)

    def test_nonwrapped_scan_compensation_is_bounded(self):
        pose = make_pose(velocity=[0.8, -0.2, 0.0])
        scan = [(15, angle, 1_000.0) for angle in [2.0, 20.0, 40.0, 60.0]]
        compensated = backprop(
            30.11,
            30.0,
            pose,
            scan,
            [
                (30.0, np.zeros(3), np.zeros(3)),
                (30.04, np.zeros(3), np.zeros(3)),
                (30.08, np.zeros(3), np.zeros(3)),
                (30.10, np.zeros(3), np.zeros(3)),
            ],
        )

        self.assertEqual(compensated.shape, (4, 3))
        self.assertTrue(np.isfinite(compensated).all())
        self.assertLess(np.linalg.norm(compensated, axis=1).max(), 2.0)

    def test_map_query_should_return_nearest_neighbors(self):
        """Voxel associations should prioritize geometry, not insertion order."""
        lidar_map = Map()
        lidar_map.add_points(np.asarray([
            [0.49, 0.49, 0.49],
            [0.01, 0.01, 0.01],
        ]))

        neighbors = lidar_map.query(np.array([0.01, 0.01, 0.01]))
        nearest_distance = np.linalg.norm(neighbors[0] - [0.01, 0.01, 0.01])
        self.assertLess(nearest_distance, 0.05)

    def test_map_query_should_not_mix_adjacent_surfaces(self):
        lidar_map = Map()
        floor = np.asarray([
            [x, y, 0.0]
            for x in np.linspace(0.05, 0.45, 5)
            for y in np.linspace(0.05, 0.45, 5)
        ])
        wall = np.asarray([
            [x, 0.5, z]
            for x in np.linspace(0.05, 0.45, 5)
            for z in np.linspace(0.05, 0.45, 5)
        ])
        lidar_map.add_points(np.vstack([floor, wall]))

        neighbors = lidar_map.query(np.array([0.25, 0.25, 0.01]))
        self.assertLess(np.max(np.abs(neighbors[:, 2])), 0.05)

    def test_explicit_constant_rotation_stays_on_so3(self):
        estimator = ESIKFStateEstimator()
        gyro = np.array([0.0, 0.0, 0.25])
        accel = np.zeros(3)

        for _ in range(400):
            estimator.predict((gyro, accel), 0.01)

        np.testing.assert_allclose(
            estimator.state.R.T @ estimator.state.R,
            np.eye(3),
            atol=1e-11,
        )
        self.assertAlmostEqual(np.linalg.det(estimator.state.R), 1.0, places=11)

    def test_prediction_covariance_remains_symmetric_positive(self):
        estimator = ESIKFStateEstimator()
        gyro = np.array([0.002, -0.001, 0.003])
        accel = np.array([0.01, -0.02, 0.005])

        for _ in range(100):
            estimator.predict((gyro, accel), 0.016)

        covariance = estimator.P
        self.assertTrue(np.isfinite(covariance).all())
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-10)
        np.testing.assert_allclose(
            estimator.state.R.T @ estimator.state.R,
            np.eye(3),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
