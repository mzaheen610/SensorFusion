from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from lidar.backward import backprop


def make_pose():
    return SimpleNamespace(
        R=np.eye(3),
        p=np.zeros(3),
        v=np.zeros(3),
        bg=np.zeros(3),
        ba=np.zeros(3),
        g=np.zeros(3),
    )


class BackpropTests(unittest.TestCase):
    def test_backprop_identity_pose_projects_points_directly(self):
        imu_pose = make_pose()
        scan = [
            (1, 0.0, 1000.0),
            (1, 90.0, 2000.0),
        ]

        compensated = backprop(
            scan_end_time=10.10,
            prev_scan_time=10.0,
            imu_pose=imu_pose,
            scan=scan,
            imu_measurement_buffer=[(10.0, np.zeros(3), np.zeros(3))],
        )

        self.assertEqual(len(compensated), 2)
        np.testing.assert_allclose(compensated[0], np.array([1.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(compensated[1], np.array([0.0, 2.0, 0.0]), atol=1e-8)

    def test_backprop_uses_timestamped_imu_history(self):
        imu_pose = make_pose()
        scan = [
            (1, 0.0, 1000.0),
            (1, 180.0, 1000.0),
        ]
        gyro = np.array([0.0, 0.0, -np.pi / 2.0])
        imu_measurement_buffer = [
            (1.045, gyro, np.zeros(3)),
            (1.090, gyro, np.zeros(3)),
            (1.135, gyro, np.zeros(3)),
            (1.180, gyro, np.zeros(3)),
        ]

        compensated = backprop(
            scan_end_time=1.225,
            prev_scan_time=1.045,
            imu_pose=imu_pose,
            scan=scan,
            imu_measurement_buffer=imu_measurement_buffer,
        )

        expected_first = np.array([
            np.cos(0.09 * np.pi),
            np.sin(0.09 * np.pi),
            0.0,
        ])
        expected_second = np.array([
            -np.cos(0.045 * np.pi),
            -np.sin(0.045 * np.pi),
            0.0,
        ])

        self.assertEqual(len(compensated), 2)
        np.testing.assert_allclose(compensated[0], expected_first, atol=1e-8)
        np.testing.assert_allclose(compensated[1], expected_second, atol=1e-8)
        self.assertTrue(np.isfinite(np.asarray(compensated)).all())


if __name__ == "__main__":
    unittest.main()