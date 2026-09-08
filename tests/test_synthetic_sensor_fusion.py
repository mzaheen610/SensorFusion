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
from lidar.backward import backprop
from map import Map
from utils.so3_rotation import exp



def make_pose(position=None, velocity=None, gyro_bias=None, accel_bias=None,
              gravity=None, rotation=None):
    return SimpleNamespace(
        R=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
        p=np.zeros(3) if position is None else np.asarray(position, dtype=float),
        v=np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
        bg=np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float),
        ba=np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=float),
        g=np.zeros(3) if gravity is None else np.asarray(gravity, dtype=float),
    )


def make_scan():
    """A small realistic scan: quality, angle in degrees, range in millimetres."""
    measurements = []
    for angle, distance in [
        (2.0, 208.0),
        (8.0, 211.0),
        (15.0, 216.0),
        (32.0, 245.0),
        (90.0, 1200.0),
        (180.0, 2400.0),
        (270.0, 900.0),
        (358.0, 209.0),
    ]:
        measurements.append((15, angle, distance))
    return measurements


class SyntheticBackpropagationTests(unittest.TestCase):
    def test_scan_window_and_invalid_timing(self):
        pose = make_pose()
        scan = make_scan()
        imu_buffer = [(10.0, np.zeros(3), np.zeros(3))]

        too_short = backprop(10.04, 10.0, pose, scan, imu_buffer)
        too_long = backprop(10.26, 10.0, pose, scan, imu_buffer)

        self.assertEqual(too_short.shape, (0, 3))
        self.assertEqual(too_long.shape, (0, 3))

    def test_realistic_ranges_convert_to_metres_and_remain_finite(self):
        pose = make_pose()
        scan = make_scan()
        compensated = backprop(
            scan_end_time=10.10,
            prev_scan_time=10.0,
            imu_pose=pose,
            scan=scan,
            imu_measurement_buffer=[(10.0, np.zeros(3), np.zeros(3))],
        )

        self.assertGreaterEqual(len(compensated), 7)
        self.assertTrue(np.isfinite(compensated).all())
        self.assertLess(np.linalg.norm(compensated, axis=1).max(), 15.0)
        np.testing.assert_allclose(
            compensated[0],
            np.array([0.207873, 0.00725, 0.0]),
            atol=2e-3,
        )

    def test_constant_angular_rate_preserves_rotation_structure(self):
        pose = make_pose()
        scan = [(15, 0.0, 1000.0), (15, 180.0, 1000.0)]
        gyro = np.array([0.0, 0.0, -np.pi / 2.0])
        imu_buffer = [
            (1.045, gyro, np.zeros(3)),
            (1.090, gyro, np.zeros(3)),
            (1.135, gyro, np.zeros(3)),
            (1.180, gyro, np.zeros(3)),
        ]

        compensated = backprop(1.225, 1.045, pose, scan, imu_buffer)

        self.assertEqual(compensated.shape, (2, 3))
        np.testing.assert_allclose(
            compensated[0],
            np.array([np.cos(0.09 * np.pi), np.sin(0.09 * np.pi), 0.0]),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            compensated[1],
            np.array([-1.0, 0.0, 0.0]),
            atol=1e-8,
        )

    def test_constant_velocity_compensation_uses_timestamped_motion(self):
        pose = make_pose(velocity=[1.0, 0.0, 0.0])
        scan = [(15, 0.0, 1000.0), (15, 180.0, 1000.0)]
        imu_buffer = [
            (20.05, np.zeros(3), np.zeros(3)),
            (20.10, np.zeros(3), np.zeros(3)),
            (20.15, np.zeros(3), np.zeros(3)),
            (20.20, np.zeros(3), np.zeros(3)),
        ]

        compensated = backprop(20.25, 20.05, pose, scan, imu_buffer)

        self.assertEqual(compensated.shape, (2, 3))
        np.testing.assert_allclose(compensated[0], [0.80, 0.0, 0.0], atol=0.03)
        np.testing.assert_allclose(compensated[1], [-1.0, 0.0, 0.0], atol=0.03)

    def test_nonfinite_and_far_points_are_discarded(self):
        pose = make_pose()
        scan = [
            (15, 0.0, 1000.0),
            (15, 90.0, np.nan),
            (15, 180.0, 20000.0),
        ]

        compensated = backprop(
            30.09,
            30.0,
            pose,
            scan,
            [(30.0, np.zeros(3), np.zeros(3))],
        )

        self.assertEqual(compensated.shape, (1, 3))
        np.testing.assert_allclose(compensated[0], [1.0, 0.0, 0.0])


class SyntheticLidarResidualTests(unittest.TestCase):
    def make_plane_map(self):
        lidar_map = Map()
        grid = []
        for x in np.linspace(0.05, 0.45, 6):
            for y in np.linspace(-0.2, 0.2, 6):
                grid.append([x, y, 0.0])
        lidar_map.add_points(np.asarray(grid))
        return lidar_map

    def test_zero_residual_scan_does_not_move_state(self):
        estimator = ESIKFStateEstimator()
        estimator.P = 1e-3 * np.eye(18)
        lidar_map = self.make_plane_map()
        scan = [(15, angle, 100.0) for angle in np.linspace(-60.0, 60.0, 25)]
        points = np.asarray([
            [0.1, -0.2, 0.0],
            [0.1, -0.1, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.1, 0.2, 0.0],
        ])
        state = make_pose()
        state_before = make_pose()

        _, applied, _ = estimator.lidar_update(
            scan, state, estimator.P.copy(), points, lidar_map
        )

        self.assertFalse(applied)
        np.testing.assert_allclose(state.R, state_before.R, atol=1e-12)
        np.testing.assert_allclose(state.p, state_before.p, atol=1e-12)

    def test_plane_residual_update_is_finite_and_bounded(self):
        estimator = ESIKFStateEstimator()
        estimator.P = 1e-2 * np.eye(18)
        lidar_map = self.make_plane_map()
        scan = [(15, angle, 100.0) for angle in np.linspace(-60.0, 60.0, 25)]
        points = np.asarray([
            [x, y, 0.1]
            for x in np.linspace(0.05, 0.45, 5)
            for y in np.linspace(-0.2, 0.2, 5)
        ])
        state = make_pose()

        points_world, applied, covariance = estimator.lidar_update(
            scan, state, estimator.P.copy(), points, lidar_map
        )

        self.assertTrue(applied)
        self.assertTrue(np.isfinite(points_world).all())
        self.assertTrue(np.isfinite(covariance).all())
        self.assertLess(np.linalg.norm(state.p), 1.0)
        self.assertLess(
            np.linalg.norm(state.R.T @ state.R - np.eye(3)),
            1e-12,
        )
        self.assertGreaterEqual(estimator.last_lidar_residual_count, 10)


if __name__ == "__main__":
    unittest.main()
