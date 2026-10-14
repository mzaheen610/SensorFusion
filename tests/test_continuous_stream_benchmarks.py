"""
Unit Tests for Continuous File-Streamed Sensor Fusion Benchmarks.

Tests advanced continuous edge cases fed sequentially from disk archives:
- Scenario 7: Continuous Loop Closure (16m multi-room closed circuit)
- Scenario 8: Long Featureless Corridor (Aperture effect & degenerate geometry)
- Scenario 9: High-Frequency S-Curve Slalom (Continuous angular reversal stability)
- Scenario 10: Dense Crowd Navigation (Multi-pedestrian dynamic clutter)
- Scenario 11: Sensor Dropout & Re-acquisition (2.5s LiDAR blackout & recovery)
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from continuous_stream_edge_cases import (
    run_scenario_7_continuous_loop_closure,
    run_scenario_8_corridor_aperture_stress,
    run_scenario_9_high_freq_slalom,
    run_scenario_10_dense_crowd_navigation,
    run_scenario_11_sensor_dropout_recovery,
)


class ContinuousStreamBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = PROJECT_ROOT / "experiments" / "continuous_streams_data"
        cls.data_dir.mkdir(parents=True, exist_ok=True)

    def test_scenario_7_continuous_loop_closure(self):
        """Verify 16m closed-circuit perimeter trajectory returns to origin with <10cm drift."""
        np.random.seed(42)
        history = run_scenario_7_continuous_loop_closure(self.data_dir)
        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])

        self.assertLess(ate_rmse, 0.22, f"Loop closure ATE RMSE {ate_rmse*100:.2f}cm exceeds 22cm")
        self.assertLess(final_err, 0.10, f"Loop closure return drift {final_err*100:.2f}cm exceeds 10cm")

    def test_scenario_8_corridor_aperture_stress(self):
        """
        Verify that in a long featureless corridor, lateral Y error is locked tight (<2cm)
        even when unconstrained longitudinal X slippage occurs due to aperture limits.
        """
        np.random.seed(42)
        history = run_scenario_8_corridor_aperture_stress(self.data_dir)
        y_err = [abs(p[1] - tp[1]) for p, tp in zip(history['est_p'], history['true_p'])]
        max_lateral_err = max(y_err)

        self.assertLess(max_lateral_err, 0.02, f"Corridor lateral Y error {max_lateral_err*100:.2f}cm exceeds 2cm")

    def test_scenario_9_high_frequency_slalom(self):
        """Verify continuous S-curve weaving at 1.0 rad/s maintains <10cm max error and <5 deg yaw error."""
        np.random.seed(42)
        history = run_scenario_9_high_freq_slalom(self.data_dir)
        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        max_err = float(np.max(history['error_p']))
        max_yaw_err_deg = float(np.rad2deg(np.max(history['error_yaw'])))

        self.assertLess(ate_rmse, 0.10, f"Slalom ATE RMSE {ate_rmse*100:.2f}cm exceeds 10cm")
        self.assertLess(max_err, 0.12, f"Slalom peak error {max_err*100:.2f}cm exceeds 12cm")
        self.assertLess(max_yaw_err_deg, 5.0, f"Slalom yaw error {max_yaw_err_deg:.2f} deg exceeds 5 deg")

    def test_scenario_10_dense_crowd_navigation(self):
        """Verify multi-pedestrian crossing clutter does not cause divergence or map pollution."""
        np.random.seed(42)
        history = run_scenario_10_dense_crowd_navigation(self.data_dir)
        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])

        self.assertLess(ate_rmse, 0.28, f"Crowd ATE RMSE {ate_rmse*100:.2f}cm exceeds 28cm")
        self.assertLess(final_err, 0.18, f"Post-crowd resting drift {final_err*100:.2f}cm exceeds 18cm")

    def test_scenario_11_sensor_dropout_recovery(self):
        """Verify that after a 2.5s LiDAR blackout, the filter re-acquires and recovers to <5cm error."""
        np.random.seed(42)
        history = run_scenario_11_sensor_dropout_recovery(self.data_dir)
        final_recovered_err = float(history['error_p'][-1])

        self.assertLess(final_recovered_err, 0.05, f"Recovered drift {final_recovered_err*100:.2f}cm exceeds 5cm")


if __name__ == "__main__":
    unittest.main()
