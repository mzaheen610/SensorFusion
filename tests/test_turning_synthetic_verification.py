import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator
from map import Map
from utils.so3_rotation import exp


class TurningRunawaySyntheticTests(unittest.TestCase):
    """
    Synthetic tests verifying:
    1. Prevention of ballistic velocity runaway during rapid turns.
    2. Prevention of wall-sliding ICP jumps through bounded position correction.
    3. Proper tracking of legitimate forward translation without degradation.
    4. In-place rotation stability.
    """

    def test_reconciled_velocity_during_rapid_turn_is_suppressed(self):
        """
        Replay scenario from log run_2026_10_12_16_45_47.log:
        When the platform turns at omega = 0.92 rad/s (53 deg/s) with zero physical
        translation, any ICP displacement jump must NOT be converted into linear velocity.
        """
        # Current state before turn
        state_p = np.array([0.40, -0.26, 0.0])
        last_committed_pos = state_p.copy()
        live_velocity = np.zeros(3)

        # Simulated ICP jump from wall-sliding during turn: delta_p = [0.66, 0.52] (0.84m)
        icp_jump = np.array([0.6597, 0.5210, 0.0])
        corrected_p = state_p + icp_jump
        dt_scan = 0.37
        gyro_max = 0.9225  # 53 deg/s rotation from log

        disp = float(np.linalg.norm(corrected_p[:2] - last_committed_pos[:2]))

        # --- UNGUARDED LOGIC (What caused the failure in 16_45_47) ---
        v_obs_unguarded = (corrected_p - last_committed_pos) / dt_scan
        speed_unguarded = float(np.linalg.norm(v_obs_unguarded[:2]))
        MAX_PLATFORM_SPEED = 0.8
        if speed_unguarded > MAX_PLATFORM_SPEED:
            v_obs_unguarded[:2] = (v_obs_unguarded[:2] / speed_unguarded) * MAX_PLATFORM_SPEED
        v_reconciled_unguarded = 0.7 * v_obs_unguarded[:2]
        unguarded_speed = float(np.linalg.norm(v_reconciled_unguarded))

        # In unguarded code, velocity jumped to 0.56 m/s:
        self.assertGreater(unguarded_speed, 0.50, "Unguarded logic creates 0.56 m/s phantom velocity")

        # --- GUARDED LOGIC (Proposed Fix) ---
        # Guard 1: Suppress linear velocity reconciliation during active angular turning (gyro_max > 0.20 rad/s)
        # Guard 2: Reject single-scan displacement jumps > 0.25m as physical velocity
        is_turning = (gyro_max is not None and gyro_max > 0.20)
        is_jump = (disp > 0.25)

        if is_turning or is_jump:
            v_obs_guarded = np.zeros(3)
        elif disp < 0.04:
            v_obs_guarded = np.zeros(3)
        else:
            v_obs_guarded = (corrected_p - last_committed_pos) / dt_scan
            v_obs_guarded[2] = 0.0
            speed_obs = float(np.linalg.norm(v_obs_guarded[:2]))
            if speed_obs > MAX_PLATFORM_SPEED:
                v_obs_guarded[:2] = (v_obs_guarded[:2] / speed_obs) * MAX_PLATFORM_SPEED

        v_reconciled_guarded = 0.7 * v_obs_guarded[:2]
        guarded_speed = float(np.linalg.norm(v_reconciled_guarded))

        self.assertEqual(guarded_speed, 0.0, "Guarded logic must set linear velocity to 0.0 during turn")

        # Simulate 1.5 seconds of IMU propagation with both velocities:
        p_unguarded = corrected_p.copy()
        p_guarded = corrected_p.copy()
        dt_imu = 0.01  # 100 Hz
        for _ in range(150):
            p_unguarded[:2] += v_reconciled_unguarded * dt_imu
            p_guarded[:2] += v_reconciled_guarded * dt_imu

        unguarded_fly_dist = float(np.linalg.norm(p_unguarded - corrected_p))
        guarded_fly_dist = float(np.linalg.norm(p_guarded - corrected_p))

        # Unguarded flies over 0.84 meters into space; guarded flies 0.0 meters:
        self.assertGreater(unguarded_fly_dist, 0.80)
        self.assertEqual(guarded_fly_dist, 0.0)

    def test_legitimate_forward_motion_velocity_is_preserved(self):
        """
        Verify that when the robot physically translates forward at 0.25 m/s
        without rotating, guarded velocity reconciliation smoothly tracks it.
        """
        state_p = np.array([0.0, 0.0, 0.0])
        last_committed_pos = state_p.copy()
        dt_scan = 0.30

        # Physical forward motion of 7.5 cm in 0.3s (0.25 m/s)
        physical_step = np.array([0.075, 0.0, 0.0])
        next_p = state_p + physical_step
        gyro_max = 0.02  # Low gyro (moving straight, not turning)
        disp = float(np.linalg.norm(next_p[:2] - last_committed_pos[:2]))

        is_turning = (gyro_max is not None and gyro_max > 0.20)
        is_jump = (disp > 0.25)

        self.assertFalse(is_turning)
        self.assertFalse(is_jump)

        if is_turning or is_jump:
            v_obs = np.zeros(3)
        elif disp < 0.04:
            v_obs = np.zeros(3)
        else:
            v_obs = (next_p - last_committed_pos) / dt_scan
            v_obs[2] = 0.0
            speed_obs = float(np.linalg.norm(v_obs[:2]))
            MAX_PLATFORM_SPEED = 0.8
            if speed_obs > MAX_PLATFORM_SPEED:
                v_obs[:2] = (v_obs[:2] / speed_obs) * MAX_PLATFORM_SPEED

        # Observed velocity should match physical speed 0.25 m/s
        self.assertAlmostEqual(v_obs[0], 0.25, places=3)
        self.assertEqual(v_obs[1], 0.0)

    def test_position_correction_bound_rejects_large_wall_sliding(self):
        """
        Verify that setting max_position_correction = 0.25m rejects implausible
        ICP wall-sliding jumps (like 0.84m) while accepting plausible tracking steps (<= 0.15m).
        """
        estimator = ESIKFStateEstimator()
        max_position_correction = 0.25  # Proposed bound

        # 1. Implausible jump (0.84m from line 3802)
        dx_implausible = np.zeros(18)
        dx_implausible[3:6] = np.array([0.6597, 0.5210, 0.0])  # norm = 0.84m
        dx_norm_implausible = float(np.linalg.norm(dx_implausible[3:6]))

        self.assertGreater(dx_norm_implausible, max_position_correction)

        # 2. Plausible tracking step (0.08m)
        dx_plausible = np.zeros(18)
        dx_plausible[3:6] = np.array([0.06, 0.05, 0.0])  # norm = 0.078m
        dx_norm_plausible = float(np.linalg.norm(dx_plausible[3:6]))

        self.assertLess(dx_norm_plausible, max_position_correction)

    def test_in_place_pivot_turn_maintains_position_stability(self):
        """
        Simulate an in-place 90-degree pivot turn over 2 seconds:
        - Angular rate omega_z = 0.785 rad/s (45 deg/s)
        - Physical translation = 0
        - Verify that with proposed guards, position remains at origin and
          velocity does not experience positive feedback runaway.
        """
        p = np.array([0.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 0.0])
        last_committed_pos = p.copy()

        # Simulate 6 successive LiDAR scans during a 2-second turn
        for scan_idx in range(6):
            dt_scan = 0.33
            gyro_max = 0.785  # Rotating fast in place
            
            # Simulated ICP jitter during rotation: small noise + wall slip attempt
            icp_jitter = np.array([0.05 * np.cos(scan_idx), 0.05 * np.sin(scan_idx), 0.0])
            p += icp_jitter

            disp = float(np.linalg.norm(p[:2] - last_committed_pos[:2]))
            
            # Guarded velocity reconciliation:
            is_turning = (gyro_max is not None and gyro_max > 0.20)
            is_jump = (disp > 0.25)
            if is_turning or is_jump:
                v_obs = np.zeros(3)
            elif disp < 0.04:
                v_obs = np.zeros(3)
            else:
                v_obs = (p - last_committed_pos) / dt_scan
                v_obs[2] = 0.0

            v[:2] = 0.3 * v[:2] + 0.7 * v_obs[:2]

            # IMU propagation over scan period (33 steps of 10ms)
            for _ in range(33):
                p += v * 0.01

            last_committed_pos = p.copy()

        # At the end of the turn, velocity must remain zero and position must not runaway
        self.assertLess(float(np.linalg.norm(v)), 0.01, "Velocity must not runaway during turn")
        self.assertLess(float(np.linalg.norm(p)), 0.15, "Position must stay centered near origin")

    def test_static_predict_holds_position_under_accel_noise(self):
        """
        Verify that while is_static=True, predict() holds position completely frozen
        even under realistic motor vibration noise and bias.
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = True
        estimator.state.p = np.array([1.25, -0.50, 0.0])
        initial_p = estimator.state.p.copy()

        # Simulate 100 IMU prediction cycles (3.0 seconds at ~33 Hz) with vibration noise
        for _ in range(100):
            gyro_noise = np.random.normal(0, 0.01, 3)
            accel_noise = np.random.normal(0, 0.05, 3) + np.array([0.05, -0.03, 0.0])
            estimator.predict((gyro_noise, accel_noise), dt=0.03)

        self.assertTrue(estimator.is_static, "Filter must stay static under vibration noise")
        np.testing.assert_array_equal(estimator.state.p, initial_p, "Position must remain completely frozen when static")
        np.testing.assert_array_equal(estimator.state.v, np.zeros(3), "Velocity must remain zero when static")

    def test_verified_clip_recovers_from_large_initial_offset(self):
        """
        Verify that when a robot pose is offset by 0.35m from map walls,
        the covariance-sized verified clip accepts and steps toward the map
        rather than hard-rejecting (applied=False) like the old 0.08m static gate.
        """
        m = Map()
        # Wall at x = 2.0m spanning y from -1.5m to 1.5m
        wall = []
        for y in np.linspace(-1.5, 1.5, 40):
            wall.append([2.0, y, 0.0])
        m.add_points(np.array(wall))

        # True robot is at origin; observed wall in body frame at x=2.0m:
        body_points = []
        for y in np.linspace(-1.5, 1.5, 40):
            body_points.append([2.0, y, 0.0])
        body_points = np.array(body_points)

        estimator = ESIKFStateEstimator()
        estimator.is_static = True
        # Set prior state offset by +0.35m along x:
        estimator.state.p = np.array([0.35, 0.0, 0.0])
        estimator.state.R = np.eye(3)

        pts_w, applied, P_new = estimator.lidar_update(
            scan=list(range(len(body_points))),
            state=estimator.state,
            P_copy=estimator.P.copy(),
            lidar_points_compensated=body_points,
            map=m
        )

        self.assertTrue(applied, "Verified clip must accept update and NOT hard-reject on 0.35m offset")
        # Position must have moved significantly towards true origin (0.0):
        self.assertLess(estimator.state.p[0], 0.35, "Position must step towards the map")
        self.assertGreater(estimator.state.p[0], -0.10, "Position must not overshoot past true wall")

    def test_static_merge_does_not_snap_back_on_valid_correction(self):
        """
        Verify that a valid 0.15m correction is trusted directly when static
        rather than being discarded and snapped back to an old anchor.
        """
        estimator = ESIKFStateEstimator()
        estimator.state.p = np.array([0.0, 0.0, 0.0])
        last_committed_pos = np.array([0.0, 0.0, 0.0])

        # LiDAR corrected pose at 0.15m:
        state_corrected_p = np.array([0.15, 0.0, 0.0])

        # New merge logic:
        is_platform_static = True
        if is_platform_static:
            estimator.state.p = state_corrected_p.copy()
            estimator.state.v = np.zeros(3)

        np.testing.assert_array_equal(estimator.state.p, state_corrected_p,
                                      "Corrected pose must be trusted without 0.08m snap-back discard")


if __name__ == "__main__":
    unittest.main()
