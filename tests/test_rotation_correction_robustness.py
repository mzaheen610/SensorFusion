"""
Unit test companion for LiDAR rotation correction and IMU imperfection tolerance.

Validates:
1. Instantaneous heading error pull-back under ESIKF LiDAR update.
2. Moderate gyro bias resilience during turns (<0.05 rad/s).
3. Gyro scale factor tolerance (-35% to +20%).
4. Deskewing motion compensation output validity.
5. Stationary ZUPT bias adaptation limits.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from forward import ESIKFStateEstimator, State
from map import Map
from utils.so3_rotation import exp, skew, reorthonormalize
from lidar.backward import backprop
from benchmark_pipeline_rigorous import SimulationWorld
from benchmark_rotation_robustness import (
    test_instantaneous_heading_convergence,
    test_active_turn_gyro_biases,
    test_gyro_scale_factor_errors,
    test_deskewing_motion_compensation_sensitivity,
    test_zupt_bias_adaptation_limits
)


class TestRotationCorrectionRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def test_instantaneous_heading_pullback(self):
        """Verify that LiDAR update pulls back heading error up to 15 deg."""
        results = test_instantaneous_heading_convergence()
        # Find results for 8 degrees
        res_8deg = next(r for r in results if abs(r['initial_deg'] - 8.0) < 1e-3)
        self.assertTrue(res_8deg['applied'])
        self.assertGreater(res_8deg['corr_deg'], 3.0, "Should apply at least 3 deg correction for 8 deg error")

    def test_active_turn_low_bias_tolerance(self):
        """Verify that turn under small bias (<0.05 rad/s) has bounded yaw error."""
        results = test_active_turn_gyro_biases()
        res_0 = next(r for r in results if r['bias_rad_s'] == 0.0)
        res_002 = next(r for r in results if r['bias_rad_s'] == 0.02)
        self.assertLess(res_0['final_yaw_deg'], 8.0)
        self.assertLess(res_002['final_yaw_deg'], 8.0)

    def test_gyro_scale_factor_tolerance(self):
        """Verify that moderate scale factor errors (-35% to +20%) remain bounded."""
        results = test_gyro_scale_factor_errors()
        res_neg35 = next(r for r in results if r['scale_error'] == -0.35)
        res_pos20 = next(r for r in results if r['scale_error'] == 0.20)
        self.assertLess(res_neg35['final_yaw_deg'], 5.0)
        self.assertLess(res_pos20['final_yaw_deg'], 12.0)


    def test_deskewing_motion_compensation(self):
        """Verify that backprop produces valid non-empty arrays."""
        results = test_deskewing_motion_compensation_sensitivity()
        for r in results:
            self.assertEqual(r['num_points'], 180)
            self.assertNotEqual(r['status'], "FAILED")

    def test_zupt_bias_adaptation_limits(self):
        """Verify that ZUPT adapts resting bias up to 0.05 rad/s but guards against slow turns (>0.08 rad/s)."""
        results = test_zupt_bias_adaptation_limits()
        res_001 = next(r for r in results if abs(r['true_bg'] - 0.01) < 1e-3)
        res_005 = next(r for r in results if abs(r['true_bg'] - 0.05) < 1e-3)
        res_010 = next(r for r in results if abs(r['true_bg'] - 0.10) < 1e-3)
        self.assertTrue(res_001['adapted'], "Low bias 0.01 rad/s should be adapted")
        self.assertTrue(res_005['adapted'], "Medium bias 0.05 rad/s should be adapted via mean gyro")
        self.assertFalse(res_010['adapted'], "Potential turn 0.10 rad/s should be guarded (>0.08 rad/s limit)")



if __name__ == "__main__":
    unittest.main()
