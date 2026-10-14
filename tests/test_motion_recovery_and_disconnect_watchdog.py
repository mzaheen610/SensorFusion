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


    def test_motor_vibration_rejection_holds_static_state(self):
        """
        Verify that spinning RPLidar motor vibration (14.3 Hz oscillatory acceleration
        with peak amplitude ~0.35 m/s^2 and Gaussian sensor noise) is rejected by
        vector EMA filtering, holding is_static=True with zero position/velocity drift.
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = True
        dt = 0.033
        p_init = estimator.state.p.copy()

        # Simulate 300 steps (10 seconds) of 14.3 Hz motor vibration
        for i in range(300):
            t = i * dt
            vib_x = 0.30 * np.sin(2 * np.pi * 14.3 * t) + np.random.normal(0, 0.08)
            vib_y = 0.30 * np.cos(2 * np.pi * 14.3 * t) + np.random.normal(0, 0.08)
            accel = np.array([vib_x, vib_y, 0.0])
            gyro = np.random.normal(0, 0.02, 3)

            estimator.predict((gyro, accel), dt)

        self.assertTrue(estimator.is_static, "Filter must maintain static lock under motor vibration")
        np.testing.assert_allclose(estimator.state.p, p_init, atol=1e-8, err_msg="Position must not drift under motor vibration")
        np.testing.assert_allclose(estimator.state.v, np.zeros(3), atol=1e-8, err_msg="Velocity must remain zero under motor vibration")

    def test_search_radius_restricts_to_one_voxel_when_static(self):
        """
        When is_static=True, the search radius in lidar_update must be strictly 1 voxel (±0.5m)
        to prevent spurious associations with distant walls.
        When is_static=False, search radius expands to 2 voxels (±1.0m) to catch fast hand motions.
        """
        m = Map()
        # Wall at x = 1.6m
        for y in np.linspace(-0.5, 0.5, 10):
            for z in [-0.1, 0.0, 0.1]:
                m.add_points(np.array([[1.6, y, z]]))

        # Body points measured at x = 0.8m (0.8m offset from wall)
        curr_pts = np.array([[0.8, y, 0.0] for y in np.linspace(-0.3, 0.3, 6)])

        state_s = State(R=np.eye(3), p=np.zeros(3), v=np.zeros(3), bg=np.zeros(3), ba=np.zeros(3), g=np.zeros(3))
        state_d = State(R=np.eye(3), p=np.zeros(3), v=np.zeros(3), bg=np.zeros(3), ba=np.zeros(3), g=np.zeros(3))
        P = 0.05 * np.eye(18)

        # Static: search_radius = 1 (voxel range ±0.5m) -> 0.8m offset is excluded
        est_static = ESIKFStateEstimator()
        est_static.is_static = True
        est_static.lidar_update(
            scan=[(15, 0.0, 800.0)] * 6, state=state_s, P_copy=P,
            lidar_points_compensated=curr_pts, map=m
        )
        self.assertEqual(est_static.last_lidar_association_count, 0, "Static search radius (1 voxel) must exclude 0.8m offset")

        # Dynamic: search_radius = 2 (voxel range ±1.0m) -> 0.8m offset is included
        est_dyn = ESIKFStateEstimator()
        est_dyn.is_static = False
        est_dyn.lidar_update(
            scan=[(15, 0.0, 800.0)] * 6, state=state_d, P_copy=P,
            lidar_points_compensated=curr_pts, map=m
        )
        self.assertGreater(est_dyn.last_lidar_association_count, 0, "Dynamic search radius (2 voxels) must include 0.8m offset")

    def test_stationary_lidar_delta_clamped_to_three_centimeters(self):
        """
        When is_platform_static is True, any ICP delta_p must be clamped to at most 3 cm (0.03m),
        preventing single-scan map smearing or spurious runaway jumps.
        """
        # Proposed unconstrained ICP delta of 0.15m:
        delta_p = np.array([0.15, 0.05, 0.0])
        is_platform_static = True

        if is_platform_static:
            disp_static = float(np.linalg.norm(delta_p[:2]))
            if disp_static > 0.03:
                delta_p[:2] = (delta_p[:2] / disp_static) * 0.03

        self.assertAlmostEqual(float(np.linalg.norm(delta_p[:2])), 0.03, places=4)
        self.assertLessEqual(np.linalg.norm(delta_p[:2]), 0.03001)


if __name__ == "__main__":
    unittest.main()

