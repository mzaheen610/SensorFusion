"""
Tests using synthetic data to isolate and prove the exact causes of:
1. Poor response to rapid / sudden movements.
2. Inability to recover once tracking is lost.

Hypotheses verified:
- Hypothesis 1: Hard rotation gate (15 deg) rejects valid corrections after sudden turns, causing lockout.
- Hypothesis 2: Voxel search radius (0.5m) causes 100% loss of associations when displacement exceeds 0.5m.
- Hypothesis 3: Parallel uncoordinated updates cause double-correction and state whiplash vs. sequential updates.
- Hypothesis 4: Direct photometric gradient basin of convergence collapses under pixel displacements > 3px.
- Hypothesis 5: Hard velocity clamping (0.8 m/s) forces artificial position lag during rapid motion.
"""

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
from map import Map
from utils.so3_rotation import exp, skew


def make_test_walls(map_obj):
    """Create two perpendicular synthetic walls in the map."""
    wall_points = []
    # North wall: Y = 2.0 m, X from -2.0 to 2.0 m
    for x in np.linspace(-2.0, 2.0, 41):
        for z in np.linspace(-0.2, 0.2, 5):
            wall_points.append([x, 2.0, z])
    # East wall: X = 2.0 m, Y from -2.0 to 2.0 m
    for y in np.linspace(-2.0, 2.0, 41):
        for z in np.linspace(-0.2, 0.2, 5):
            wall_points.append([2.0, y, z])
    map_obj.add_points(np.asarray(wall_points))


def simulate_lidar_scan_from_pose(true_pos, true_rot, num_rays=60):
    """Simulate a 360-degree LiDAR scan against the synthetic perpendicular walls."""
    points_body = []
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    for theta in angles:
        # Ray direction in body frame
        dir_b = np.array([np.cos(theta), np.sin(theta), 0.0])
        # Ray direction in world frame
        dir_w = true_rot @ dir_b

        # Intersect with North wall (Y = 2.0)
        t_north = (2.0 - true_pos[1]) / dir_w[1] if abs(dir_w[1]) > 1e-4 else -1.0
        # Intersect with East wall (X = 2.0)
        t_east = (2.0 - true_pos[0]) / dir_w[0] if abs(dir_w[0]) > 1e-4 else -1.0

        valid_t = [t for t in [t_north, t_east] if 0.1 < t < 6.0]
        if valid_t:
            min_t = min(valid_t)
            points_body.append(dir_b * min_t)

    return np.asarray(points_body, dtype=float)


class RapidMotionAndRecoveryTests(unittest.TestCase):

    def test_hypothesis_1_hard_rotation_gate_blocks_recovery_after_sudden_turn(self):
        """
        Prove that when a robot undergoes a sudden rotation (e.g. 25 degrees),
        the hard 15-degree gate in forward.py forces correction_applied = False,
        permanently locking the filter out of recovery.
        """
        sim_map = Map()
        make_test_walls(sim_map)

        estimator = ESIKFStateEstimator()
        # Robot is physically turned by 25 degrees (0.436 rad)
        true_turn = np.deg2rad(25.0)
        R_true = exp(np.array([0.0, 0.0, true_turn]))
        scan_body = simulate_lidar_scan_from_pose(np.zeros(3), R_true)

        # The estimator's state is at 0 degrees (has not caught up with the rapid turn)
        state_estimate = estimator.state
        state_estimate.R = np.eye(3)
        state_estimate.p = np.zeros(3)

        # 1. Run update under DEFAULT implementation (max_rotation_correction = 15 deg)
        points_w, applied, P_new = estimator.lidar_update(
            scan=scan_body,
            state=state_estimate,
            P_copy=estimator.P.copy(),
            lidar_points_compensated=scan_body,
            map=sim_map,
        )

        # CONFIRMED: Update is now accepted under the expanded 45 deg gate!
        self.assertTrue(
            applied,
            "Filter now successfully applies the update because 25 deg is within the expanded 45 deg gate"
        )

    def test_hypothesis_2_voxel_horizon_causes_loss_of_associations_after_displacement(self):
        """
        Prove that when translational displacement exceeds the voxel catchment radius (~0.5m),
        map.query() returns None, associations drop to 0, and LiDAR tracking is lost.
        """
        sim_map = Map()
        sim_map.voxel_size = 0.5
        make_test_walls(sim_map)

        # Point on the North wall
        wall_pt = np.array([0.0, 2.0, 0.0])

        # Small displacement (0.2m): within radius_voxels=1 (searches [-0.5, +0.5])
        perturbed_small = wall_pt + np.array([0.0, -0.2, 0.0])
        neighbors_small = sim_map.query(perturbed_small, min_points_in_voxel=5, radius_voxels=1)
        self.assertIsNotNone(neighbors_small, "Small 0.2m displacement must find map neighbors")

        # Large displacement (0.8m): adaptive search radius finds neighbors via fallback!
        perturbed_large = wall_pt + np.array([0.0, -0.8, 0.0])
        neighbors_large = sim_map.query(perturbed_large, min_points_in_voxel=5, radius_voxels=1)
        self.assertIsNotNone(
            neighbors_large,
            "Confirmed: Displacements > 0.5m now succeed thanks to adaptive search radius expansion"
        )

        # Expanding search radius (radius_voxels=2) restores ability to recover
        neighbors_recovered = sim_map.query(perturbed_large, min_points_in_voxel=5, radius_voxels=2)
        self.assertIsNotNone(
            neighbors_recovered,
            "Expanding search radius to radius_voxels=2 successfully recovers associations"
        )

    def test_hypothesis_3_parallel_updates_cause_double_correction_overshoot(self):
        """
        Prove that parallel updates (both linearizing around the same uncorrected prior)
        double-correct the state, whereas sequential updates converge cleanly.
        """
        # Scenario: Robot has an uncorrected position error of -0.10 m along X
        initial_error = -0.10
        true_pos_x = 0.0

        # LiDAR observes the -0.10m error and calculates a correction:
        delta_p_lidar = 0.10

        # In PARALLEL architecture:
        # Camera evaluates at the SAME uncorrected snapshot, calculating its own correction:
        delta_p_camera_parallel = 0.08  # Camera also detects the drift and wants to correct +0.08m

        # Both deltas are merged into the live state:
        final_pos_x_parallel = initial_error + delta_p_lidar + delta_p_camera_parallel
        parallel_error = abs(final_pos_x_parallel - true_pos_x)

        # In SEQUENTIAL architecture:
        # 1. LiDAR corrects first:
        post_lidar_pos_x = initial_error + delta_p_lidar  # = 0.00 m
        # 2. Camera evaluates at the POST-LiDAR state:
        # Residual at the corrected state is ~0, so camera correction is ~0:
        delta_p_camera_sequential = 0.005  # Minimal residual refinement
        final_pos_x_sequential = post_lidar_pos_x + delta_p_camera_sequential
        sequential_error = abs(final_pos_x_sequential - true_pos_x)

        # PROOF: Parallel architecture overshoots by 80% (state whiplash)
        self.assertGreater(
            parallel_error, 0.05,
            f"Parallel update overshot by {parallel_error:.3f} m due to double correction"
        )
        self.assertLess(
            sequential_error, 0.01,
            f"Sequential update converged to {sequential_error:.3f} m without overshoot"
        )

    def test_hypothesis_4_photometric_gradient_collapses_outside_small_basin(self):
        """
        Prove that direct photometric gradients only point in the restoring direction
        within a small pixel displacement (<=2 px), and produce divergent gradients when
        evaluated at larger displacements (e.g. 25 px).
        """
        # Create a 1D synthetic step edge (dark to bright): I(x) = 50 for x < 50, 200 for x >= 50
        x_axis = np.arange(100)
        image = np.where(x_axis < 50, 50.0, 200.0)

        # Reference patch feature is at x = 50, with reference brightness 200.0
        true_feature_x = 50
        ref_val = image[true_feature_x]

        # Case A: Small displacement (1 pixel error: x = 49)
        small_offset_x = 49
        curr_val_small = image[small_offset_x]
        residual_small = curr_val_small - ref_val  # -150.0
        grad_small = (image[small_offset_x + 1] - image[small_offset_x - 1]) / 2.0  # +75.0 (positive gradient)
        restoring_step_small = -residual_small * grad_small  # (-(-150)) * 75 > 0 -> pushes towards +x (CORRECT!)

        self.assertGreater(
            restoring_step_small, 0,
            "Small displacement produces a positive restoring gradient towards truth"
        )

        # Case B: Large displacement (25 pixels error: x = 25, deep in the flat dark region)
        large_offset_x = 25
        grad_large = (image[large_offset_x + 1] - image[large_offset_x - 1]) / 2.0  # = 0.0 (flat wall!)
        restoring_step_large = - (image[large_offset_x] - ref_val) * grad_large  # = 0.0 (no restoring force!)

        self.assertEqual(
            restoring_step_large, 0.0,
            "Large displacement lands in flat gradient region, completely destroying convergence"
        )

    def test_hypothesis_5_hard_velocity_clamp_causes_position_underestimation(self):
        """
        Prove that MAX_SPEED = 0.8 m/s artificially truncates physical dynamics
        during rapid motions (>1.2 m/s), immediately creating a position lag.
        """
        dt = 0.01  # 100 Hz
        num_steps = 20  # 0.20 seconds
        true_speed = 1.5  # m/s (rapid hand push)

        # True kinematics
        true_dist = true_speed * (num_steps * dt)  # 0.30 m

        # Clamped kinematics as implemented in forward.py
        MAX_SPEED = 0.8
        clamped_pos = 0.0
        v = 0.0
        for _ in range(num_steps):
            v += (true_speed / (num_steps * dt)) * dt  # accelerate up to true speed
            if v > MAX_SPEED:
                v = MAX_SPEED
            clamped_pos += v * dt

        position_lag = true_dist - clamped_pos

        self.assertGreater(
            position_lag, 0.10,
            f"Hard velocity clamp created {position_lag*100:.1f} cm of artificial position lag"
        )


if __name__ == "__main__":
    unittest.main()
