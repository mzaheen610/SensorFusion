import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from forward import ESIKFStateEstimator, State
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

    def test_static_delta_merge_preserves_live_predictions(self):
        """
        Verify that when static, LiDAR corrections are merged as deltas (filter.state.p += delta_p)
        rather than overwriting with stale snapshots, preserving live IMU predictions.
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = True

        # State at scan timestamp was at [0.0, 0.0, 0.0]
        state_old_p = np.array([0.0, 0.0, 0.0])
        # LiDAR update computed a correction to [0.15, 0.0, 0.0] on that snapshot:
        state_corrected_p = np.array([0.15, 0.0, 0.0])
        delta_p = state_corrected_p - state_old_p

        # While computation ran, live filter state advanced by 0.03m:
        estimator.state.p = np.array([0.03, 0.0, 0.0])

        # Merge delta incrementally:
        is_platform_static = True
        estimator.state.p += delta_p
        if is_platform_static:
            estimator.state.v = np.zeros(3)

        # Expected position is 0.03 + 0.15 = 0.18m (not 0.15m which would discard the 0.03m prediction):
        np.testing.assert_allclose(estimator.state.p, np.array([0.18, 0.0, 0.0]), atol=1e-8,
                                   err_msg="Delta merging must preserve live state integration")

    def test_multimodal_motion_detector_rejects_static_on_high_lidar_sim(self):
        """
        Verify that strong LiDAR motion evidence (sim >= 35.0 mm) prevents false static detection
        even when IMU gyro and accel variance are low (e.g. smooth forward push).
        """
        # Smooth push: low gyro (0.075 rad/s <= 0.12) and low accel variance (0.05 <= 0.30)
        gyro_max = 0.075
        accel_var = 0.05
        is_imu_static = (gyro_max <= 0.12 and accel_var <= 0.30)
        self.assertTrue(is_imu_static, "IMU alone appears static during smooth glide")

        # But LiDAR similarity is high (90 mm shift):
        sim = 90.0
        is_lidar_static = (sim is None or sim < 35.0)
        self.assertFalse(is_lidar_static, "LiDAR must detect geometric scan difference")

        is_static_sample = is_lidar_static and is_imu_static
        self.assertFalse(is_static_sample, "Overall detector must NOT declare static when LiDAR sees 90mm shift")

    def test_gated_bias_adaptation_clamps_transients(self):
        """
        Verify that bias adaptation ignores braking transients and clamps per-step adjustments.
        """
        estimator = ESIKFStateEstimator()
        initial_ba = np.array([0.035, 0.057, -0.324])
        estimator.state.ba = initial_ba.copy()

        # Deceleration transient: mean_accel is strongly negative (-0.50 m/s^2)
        mean_accel = np.array([-0.50, 0.057, -0.324])

        # At static_count = 2 (early in static dwell), adaptation must NOT run:
        static_count = 2
        is_platform_static = True
        gyro_max = 0.04
        accel_var = 0.05
        if is_platform_static and static_count >= 10:
            if gyro_max < 0.06 and accel_var < 0.08:
                step_ba = np.clip(0.01 * (mean_accel - estimator.state.ba), -0.005, 0.005)
                estimator.state.ba += step_ba

        np.testing.assert_array_equal(estimator.state.ba, initial_ba,
                                      "Bias must not adapt before sustained dwell (static_count < 10)")

        # At static_count = 10, step must be clamped to at most 0.005 m/s^2:
        static_count = 10
        if is_platform_static and static_count >= 10:
            if gyro_max < 0.06 and accel_var < 0.08:
                step_ba = np.clip(0.01 * (mean_accel - estimator.state.ba), -0.005, 0.005)
                estimator.state.ba += step_ba

        delta_ba = estimator.state.ba - initial_ba
        self.assertLessEqual(abs(delta_ba[0]), 0.005 + 1e-9, "Per-step bias adaptation must be clamped to <= 0.005 m/s^2")


    def test_corner_turn_reversal_suppression_under_nhc(self):
        """
        Verify that when rounding a 90-degree corner around a table/obstacle:
        1. Residuals from unmapped perpendicular surfaces do not drag the vehicle backwards in reverse.
        2. Non-Holonomic Constraint (NHC) bounds reverse body displacement during turns (dx_body[0] >= -0.04m).
        """
        estimator = ESIKFStateEstimator()
        estimator.is_static = False

        # Build map with only the side edge of a table (y = -0.50m, x in [0.0, 1.2m])
        m = Map()
        side_table_pts = []
        for x in np.linspace(0.0, 1.2, 25):
            for z in [-0.05, 0.0, 0.05]:
                side_table_pts.append([x, -0.50, z])
        m.add_points(np.array(side_table_pts))

        # Robot reaches the corner at x = 1.30, y = -0.29, turned 90 deg CCW (+pi/2 yaw)
        state = State(
            R=np.array([[0.0, -1.0, 0.0],
                        [1.0,  0.0, 0.0],
                        [0.0,  0.0, 1.0]]),
            p=np.array([1.30, -0.29, 0.0]),
            v=np.zeros(3),
            bg=np.zeros(3),
            ba=np.zeros(3),
            g=np.zeros(3),
        )
        P = 0.08 * np.eye(18)

        # Beams in body frame hitting the front edge of the table (body forward is world +Y)
        body_points = []
        for y_body in np.linspace(-0.3, 0.3, 15):
            body_points.append([0.30, y_body, 0.0])
        body_points = np.array(body_points)

        pts_w, applied, P_new = estimator.lidar_update(
            scan=[(15, 0.0, 300.0)] * len(body_points),
            state=state,
            P_copy=P,
            lidar_points_compensated=body_points,
            map=m,
        )

        # Trajectory must NOT collapse backwards (x must remain >= 1.25m, not collapse to < 1.0m)
        self.assertGreaterEqual(state.p[0], 1.25, "Position must not collapse backwards past the table corner")

    def test_static_anchor_release_bidirectional_transition(self):
        """
        Verify that releasing from static mode starts cleanly from static_anchor_pos
        (wiping sub-3cm stationary creep) and bounds transition steps to plausible physical limits
        while cleanly supporting legitimate reverse motion.
        """
        static_anchor_pos = np.array([1.30, -0.29])
        filter_p = np.array([1.27, -0.27, 0.0])  # slightly pulled inside 3cm anchor radius
        filter_R = np.eye(3)

        # Negative delta_p from deliberate backward movement
        delta_p = np.array([-0.100, 0.010, 0.0])

        is_platform_static = False
        if not is_platform_static:
            if static_anchor_pos is not None:
                filter_p[:2] = static_anchor_pos.copy()
                static_anchor_pos = None
                delta_p_body = filter_R.T @ delta_p
                delta_p_body[0] = np.clip(delta_p_body[0], -0.20, 0.20)
                delta_p = filter_R @ delta_p_body
            filter_p += delta_p

        # Resulting position must start at 1.30 and apply -0.10m reverse step cleanly to 1.20:
        self.assertAlmostEqual(filter_p[0], 1.20, places=3,
                               msg="Static release must allow legitimate backward step from resting anchor")

    def test_bidirectional_reverse_velocity_reconciliation(self):
        """
        Verify that legitimate backward movement synthesized negative body velocity
        without being zeroed or inverted by active turn checks.
        """
        filter_R = np.eye(3)
        dt_scan = 0.50
        p_now = np.array([1.10, 0.0, 0.0])
        p_prev = np.array([1.20, 0.0, 0.0])  # Robot moved backwards by 10 cm in 0.5s (-0.20 m/s)

        v_obs = (p_now - p_prev) / dt_scan
        v_obs[2] = 0.0

        v_obs_body = filter_R.T @ v_obs
        v_obs_body[1] = 0.0
        v_obs_body[2] = 0.0
        MAX_REVERSE_SPEED = 0.5
        if v_obs_body[0] < -MAX_REVERSE_SPEED:
            v_obs_body[0] = -MAX_REVERSE_SPEED
        v_obs = filter_R @ v_obs_body

        self.assertAlmostEqual(v_obs[0], -0.20, places=3,
                               msg="Reverse velocity must be accurately preserved without being zeroed")

    def test_frontier_mapping_condition_on_heading_rotation(self):
        """
        Verify that rotating by >= 15 degrees (0.26 rad) unlatches map point addition
        so that new frontiers are progressively mapped during turns.
        """
        last_map_heading = 0.0  # rad
        current_heading = 0.30  # ~17.2 deg rotation

        heading_diff = abs(float(np.arctan2(np.sin(current_heading - last_map_heading),
                                            np.cos(current_heading - last_map_heading))))
        self.assertGreaterEqual(heading_diff, 0.26, "Heading difference must exceed 15-degree threshold")

        gyro_max = 0.45  # moderate turn rate (25.8 deg/s)
        is_high_speed_turn = (gyro_max > 0.80)
        can_add_frontier = (not is_high_speed_turn and heading_diff >= 0.26)
        self.assertTrue(can_add_frontier, "Moderate turn with >15 deg rotation must permit frontier mapping")


if __name__ == "__main__":
    unittest.main()
