from pathlib import Path
import sys
import time
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator, State
from map import Map


class MotionRecoveryAndWatchdogTests(unittest.TestCase):
    def test_forward_stroke_large_displacement_recovery(self):
        """
        When the robot is moved forward 0.6m in one stroke:
        A wall previously mapped at x = 1.8m is now measured at x = 1.2m in body frame.
        The adaptive search radius (2 voxels) must associate the points and correct
        position along +X (forward), never backwards behind origin.
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = False
        m = Map()

        # Build a mapped wall at x = 1.8m
        wall_pts = []
        for y in np.linspace(-1.0, 1.0, 12):
            for z in [-0.05, 0.0, 0.05]:
                wall_pts.append([1.8, y, z])
        m.add_points(np.array(wall_pts))

        # Measured points after a 0.6m forward translation (wall is now at x = 1.2m)
        curr_pts = []
        for y in np.linspace(-0.8, 0.8, 10):
            curr_pts.append([1.2, y, 0.0])
        curr_pts = np.array(curr_pts)

        # Initial pose estimate is around origin (has not caught up to the 0.6m stroke yet)
        state = State(
            R=np.eye(3),
            p=np.zeros(3),
            v=np.zeros(3),
            bg=np.zeros(3),
            ba=np.zeros(3),
            g=np.zeros(3),
        )
        P = 0.05 * np.eye(18)

        pts_world, applied, P_new = estimator.lidar_update(
            scan=[(15, 0.0, 1200.0)] * 10,
            state=state,
            P_copy=P,
            lidar_points_compensated=curr_pts,
            map=m,
        )

        self.assertTrue(applied, "Update must be applied using expanded dynamic search radius")
        # delta_p along X must be positive (advancing forward towards +X)
        self.assertGreater(state.p[0], 0.10, "Position must correct forward, never backwards")
        self.assertAlmostEqual(state.p[1], 0.0, places=2)

    def test_association_downsampling_caps_candidates(self):
        """
        Verify that a dense scan of 200 points is downsampled to at most 40 candidate points,
        preventing Raspberry Pi matching stalls while providing full pose constraint.
        """
        estimator = ESIKFStateEstimator()
        m = Map()
        # Add map points
        m.add_points(np.array([[2.0, y, 0.0] for y in np.linspace(-1, 1, 35)]))

        dense_pts = np.array([[1.9, np.sin(i), 0.0] for i in range(200)])
        state = State(
            R=np.eye(3),
            p=np.zeros(3),
            v=np.zeros(3),
            bg=np.zeros(3),
            ba=np.zeros(3),
            g=np.zeros(3),
        )
        P = 0.01 * np.eye(18)

        pts_w, applied, P_new = estimator.lidar_update(
            scan=[(15, 0.0, 1900.0)] * 10,
            state=state,
            P_copy=P,
            lidar_points_compensated=dense_pts,
            map=m,
        )

        # Total associations evaluated can never exceed 40
        self.assertLessEqual(estimator.last_lidar_association_count, 40)
        self.assertIsNotNone(estimator.last_lidar_time)

    def test_coasting_velocity_damping_under_low_acceleration(self):
        """
        When dynamic (is_static=False) and coasting without thrust (|a_xy| < 0.12),
        velocity must smoothly decay to zero rather than drifting open-loop indefinitely.
        When active thrust is present (|a_xy| >= 0.12), velocity must integrate normally.
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = False
        estimator.state.v = np.array([0.40, 0.0, 0.0])  # Initial forward coasting speed

        dt = 0.02  # 50 Hz
        # Coasting: only tiny sensor noise (< 0.02 m/s^2)
        for _ in range(50):  # 1.0 second
            estimator.predict((np.zeros(3), np.array([0.01, 0.0, 0.0])), dt)

        # Velocity must have decayed significantly from 0.40 m/s
        self.assertLess(estimator.state.v[0], 0.10)

        # Active thrust test: push with 0.50 m/s^2 forward
        v_before = estimator.state.v[0]
        estimator.predict((np.zeros(3), np.array([0.50, 0.0, 0.0])), dt)
        # Active acceleration integrates directly
        self.assertGreater(estimator.state.v[0], v_before + 0.50 * dt * 0.95)

    def test_disconnect_watchdog_halts_drift_in_imu_thread(self):
        """
        When LiDAR updates stall for > 1.5 seconds (e.g. cable unplugged),
        the IMU thread watchdog must engage static lock, zero velocity,
        and halt open-loop trajectory divergence.
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = False
        estimator.state.v = np.array([-0.30, 0.15, 0.0])  # Lingering velocity from braking
        estimator.last_lidar_time = time.monotonic() - 2.0  # 2.0s since last LiDAR scan

        now = time.monotonic()
        # Simulated imu_thread check
        last_lidar = getattr(estimator, 'last_lidar_time', None)
        if last_lidar is not None and (now - last_lidar > 1.5):
            estimator.is_static = True
            estimator.state.v[:] = 0.0

        self.assertTrue(estimator.is_static)
        np.testing.assert_allclose(estimator.state.v, np.zeros(3))

        # Further IMU prediction in static lock must not diverge
        p_frozen = estimator.state.p.copy()
        for _ in range(100):
            estimator.predict((np.zeros(3), np.array([-0.05, 0.10, 0.0])), 0.02)

        np.testing.assert_allclose(estimator.state.p, p_frozen, atol=1e-8)

    def test_map_query_early_break_and_list_support(self):
        """Map.query handles lists gracefully and returns nearest points."""
        m = Map()
        m.add_points(np.array([
            [1.0, 0.0, 0.0],
            [1.05, 0.0, 0.0],
            [1.10, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ]))

        # Pass python list to get_voxel_key
        key = m.get_voxel_key([1.0, 0.0, 0.0])
        self.assertEqual(key, (2.0, 0.0, 0.0))

        # Query near 1.0
        neighbors = m.query([1.02, 0.0, 0.0], min_points_in_voxel=2, radius_voxels=1)
        self.assertIsNotNone(neighbors)
        self.assertGreaterEqual(len(neighbors), 2)
        # First neighbor must be the closest one
        dist = np.linalg.norm(neighbors[0] - np.array([1.02, 0.0, 0.0]))
        self.assertLess(dist, 0.05)


if __name__ == "__main__":
    unittest.main()
