from pathlib import Path
import re
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator
from lidar.backward import backprop
from map import Map
from utils.projections import (
    calculate_photometric_error,
    project,
    project_points_to_frame,
    project_points_world,
)
from utils.similarity import scan_similarity, scan_to_bins
from utils.so3_rotation import exp, skew


LOG_PATH = PROJECT_ROOT / "logs" / "sep_7.log"
ORIGINAL_POINT_RE = re.compile(
    r"Original LiDAR points:\s*\(([-+0-9.eE]+),\s*"
    r"([-+0-9.eE]+),\s*([-+0-9.eE]+)\)"
)
COMPENSATED_POINT_RE = re.compile(
    r"Compensated LiDAR points:\s*\[\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]"
)
ROTATION_RE = re.compile(
    r"R_error\s*=\s*([-+0-9.eE]+)\s+det\(R\)\s*=\s*([-+0-9.eE]+)"
)
DIAGNOSTIC_RE = re.compile(
    r"LiDAR update diagnostics: compensated=(\d+) \| map_points=(\d+) \| "
    r"associations=(\d+) \| applied=(True|False) \| residuals=(\d+)"
)


def load_log():
    return LOG_PATH.read_text(encoding="utf-8")


def parse_logged_points(pattern, text):
    return [tuple(float(value) for value in match.groups())
            for match in pattern.finditer(text)]


def make_pose(position=None, velocity=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        R=np.eye(3),
        p=np.zeros(3) if position is None else np.asarray(position, dtype=float),
        v=np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
        bg=np.zeros(3),
        ba=np.zeros(3),
        g=np.zeros(3),
    )


class SavedLogFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = load_log()
        cls.raw_points = parse_logged_points(ORIGINAL_POINT_RE, cls.text)
        cls.compensated_points = parse_logged_points(
            COMPENSATED_POINT_RE, cls.text
        )
        cls.rotation_diagnostics = [
            tuple(float(value) for value in match.groups())
            for match in ROTATION_RE.finditer(cls.text)
        ]
        cls.update_diagnostics = [
            (
                int(compensated),
                int(map_points),
                int(associations),
                applied == "True",
                int(residuals),
            )
            for compensated, map_points, associations, applied, residuals
            in DIAGNOSTIC_RE.findall(cls.text)
        ]

    def test_log_contains_realistic_lidar_samples(self):
        self.assertGreaterEqual(len(self.raw_points), 100)
        raw = np.asarray(self.raw_points)
        qualities, angles, distances = raw.T

        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue(((qualities >= 0) & (qualities <= 15)).all())
        self.assertTrue(((angles >= 0) & (angles < 360)).all())
        self.assertTrue((distances > 0).all())
        self.assertGreater(distances.max(), 2000.0)
        self.assertLess(distances.min(), 250.0)

    def test_logged_compensated_points_obey_backprop_output_contract(self):
        compensated = np.asarray(self.compensated_points)

        self.assertGreaterEqual(len(compensated), 100)
        self.assertTrue(np.isfinite(compensated).all())
        self.assertTrue((np.linalg.norm(compensated, axis=1) <= 15.0).all())

    def test_logged_rotation_diagnostics_remain_finite(self):
        diagnostics = np.asarray(self.rotation_diagnostics)
        errors = diagnostics[:, 0]
        determinants = diagnostics[:, 1]

        self.assertGreaterEqual(len(diagnostics), 100)
        self.assertTrue(np.isfinite(diagnostics).all())
        self.assertTrue((determinants > 0.99).all())
        self.assertLess(errors.max(), 0.01)

    def test_logged_update_diagnostics_have_consistent_counts(self):
        self.assertGreaterEqual(len(self.update_diagnostics), 10)
        for compensated, map_points, associations, applied, residuals in self.update_diagnostics:
            self.assertGreaterEqual(compensated, 0)
            self.assertGreaterEqual(map_points, 0)
            self.assertGreaterEqual(associations, 0)
            self.assertGreaterEqual(residuals, 0)
            self.assertLessEqual(associations, compensated)
            self.assertLessEqual(residuals, associations)
            if applied:
                self.assertGreaterEqual(residuals, 10)

    def test_first_logged_scan_replays_through_backprop(self):
        scan = [
            (int(quality), angle, distance)
            for quality, angle, distance in self.raw_points[:5]
        ]
        compensated = backprop(
            scan_end_time=100.10,
            prev_scan_time=100.0,
            imu_pose=make_pose(),
            scan=scan,
            imu_measurement_buffer=[(100.0, np.zeros(3), np.zeros(3))],
        )

        self.assertEqual(compensated.shape, (5, 3))
        self.assertTrue(np.isfinite(compensated).all())
        expected_ranges = np.asarray([point[2] for point in scan]) / 1000.0
        np.testing.assert_allclose(
            np.linalg.norm(compensated, axis=1),
            expected_ranges,
            atol=1e-10,
        )


class PureModuleTests(unittest.TestCase):
    def test_so3_exponential_is_rotation_for_multiple_axes(self):
        for theta in ([0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [1.0, 0.0, -0.5]):
            rotation = exp(np.asarray(theta, dtype=float))
            np.testing.assert_allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-12,
            )
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)

    def test_skew_matches_cross_product(self):
        vector = np.array([0.2, -0.4, 0.7])
        other = np.array([-0.5, 0.1, 0.9])
        np.testing.assert_allclose(skew(vector) @ other, np.cross(vector, other))

    def test_scan_binning_wraps_angles_and_similarity_is_symmetric(self):
        scan_a = [(15, -1.0, 200.0)]
        scan_b = [(15, -1.0, 205.0)]
        for index in range(20):
            angle = float(index * 2 + 1)
            scan_a.append((15, angle, 200.0 + index))
            scan_b.append((15, angle, 205.0 + index))
        scan_a.append((15, 359.0, 220.0))
        scan_b.append((15, 359.0, 225.0))
        bins_a = scan_to_bins(scan_a)
        bins_b = scan_to_bins(scan_b)

        self.assertEqual(bins_a[0], 200.0)
        self.assertEqual(bins_a[179], 220.0)
        self.assertEqual(scan_similarity(bins_a, bins_b), 5.0)
        self.assertIsNone(scan_similarity(np.full(180, np.nan), bins_b))

    def test_map_voxels_cap_points_and_return_neighbors(self):
        lidar_map = Map()
        points = np.asarray([[0.01 * i, 0.0, 0.0] for i in range(30)])
        lidar_map.add_points(points)

        self.assertEqual(lidar_map.num_points(), 20)
        neighbors = lidar_map.query(np.array([0.1, 0.0, 0.0]))
        self.assertIsNotNone(neighbors)
        self.assertGreaterEqual(len(neighbors), 3)
        self.assertEqual(
            lidar_map.get_voxel_key(np.array([0.49, 0.49, 0.49])),
            (0.0, 0.0, 0.0),
        )

    def test_projection_filters_invalid_depth_and_applies_pose(self):
        state = make_pose(position=[1.0, 2.0, 3.0])
        world_points = project_points_world(
            np.asarray([[1.0, 0.0, 0.0]]),
            state,
            np.eye(4),
        )
        np.testing.assert_allclose(world_points[0], [2.0, 2.0, 3.0])
        np.testing.assert_allclose(project([0.0, 0.0, 1.0]), [320.0, 240.0])

        projected = project_points_to_frame(
            np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
            np.eye(4),
            np.eye(4),
        )
        self.assertEqual(len(projected), 1)

    def test_photometric_error_returns_pixelwise_differences(self):
        current = np.asarray([[1, 2], [3, 4]])
        reference = np.asarray([[0, 1], [4, 6]])
        self.assertEqual(calculate_photometric_error(current, reference), [1, 1, -1, -2])

    def test_predict_with_logged_scale_imu_values_preserves_so3(self):
        estimator = ESIKFStateEstimator()
        gyro = np.array([0.00267, -0.00154, 0.00109])
        accel = np.array([0.004, -0.002, -0.02])

        for _ in range(20):
            estimator.predict((gyro, accel), 0.016)

        self.assertTrue(np.isfinite(estimator.state.R).all())
        np.testing.assert_allclose(
            estimator.state.R.T @ estimator.state.R,
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(np.linalg.det(estimator.state.R), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
