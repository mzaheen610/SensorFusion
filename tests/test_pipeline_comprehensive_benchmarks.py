"""
Unit Tests for Comprehensive Sensor Fusion Pipeline Stress-Testing & Benchmarks.

Validates the full ESIKF LiDAR-IMU-Camera pipeline across all critical edge cases:
1. Sudden Movements & Jerky Motion (impulse accelerations, emergency stops, pivot turns)
2. Backward Movement (reverse translation and turning)
3. Sideways Movement (lateral crab strafe, aperture effect)
4. Dynamic Obstacles & Crossing Pedestrians (occlusion handling, dynamic rejection)
5. Low-Cost IMU Sensor Artifacts (biases, 14.3 Hz motor resonance harmonics, sample jitter)
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from benchmark_pipeline_rigorous import (
    SimulationWorld,
    PipelineRunner,
    TrajectoryGenerator,
    run_scenario_1_sudden_movements,
    run_scenario_2_backward_movement,
    run_scenario_3_sideways_movement,
    run_scenario_4_dynamic_pedestrians,
    run_scenario_5_low_cost_imu,
)


class ComprehensivePipelineBenchmarkTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_scenario_1_sudden_movements_and_jerk(self):
        """
        Verify pipeline stability under sudden forward acceleration bursts (2.0 m/s^2),
        emergency braking (-3.0 m/s^2), aggressive 90 deg/s pivot turns, and 5 Hz jerk.
        """
        np.random.seed(42)
        history = run_scenario_1_sudden_movements()

        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])
        final_v = float(np.linalg.norm(history['est_v'][-1]))
        max_yaw_err_deg = float(np.rad2deg(np.max(history['error_yaw'])))

        self.assertLess(ate_rmse, 0.12, f"ATE RMSE {ate_rmse*100:.2f}cm exceeds 12cm threshold")
        self.assertLess(final_err, 0.08, f"Final drift {final_err*100:.2f}cm exceeds 8cm threshold")
        self.assertLess(final_v, 1e-4, f"Final resting velocity {final_v:.4f}m/s must be zero")
        self.assertLess(max_yaw_err_deg, 5.0, f"Max yaw error {max_yaw_err_deg:.2f} deg exceeds 5 deg limit")

    def test_scenario_2_backward_movement(self):
        """
        Verify pipeline stability during steady reverse translation (-0.30 m/s)
        and reverse cornering.
        """
        np.random.seed(42)
        history = run_scenario_2_backward_movement()

        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])
        applied_ratio = sum(history['applied']) / len(history['applied'])

        self.assertLess(ate_rmse, 0.10, f"Reverse ATE RMSE {ate_rmse*100:.2f}cm exceeds 10cm threshold")
        self.assertLess(final_err, 0.10, f"Final reverse drift {final_err*100:.2f}cm exceeds 10cm threshold")
        self.assertGreater(applied_ratio, 0.70, f"Update applied ratio {applied_ratio*100:.1f}% below 70%")

    def test_scenario_3_sideways_movement(self):
        """
        Verify pure lateral translation (crab strafe) parallel to boundary walls
        does not cause aperture effect lock-in or false forward/backward drift coupling.
        """
        np.random.seed(42)
        history = run_scenario_3_sideways_movement()

        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])
        est_x = np.array([p[0] for p in history['est_p']])
        max_phantom_x_drift = float(np.max(np.abs(est_x)))

        self.assertLess(ate_rmse, 0.10, f"Sideways ATE RMSE {ate_rmse*100:.2f}cm exceeds 10cm threshold")
        self.assertLess(final_err, 0.06, f"Return to origin drift {final_err*100:.2f}cm exceeds 6cm threshold")
        self.assertLess(max_phantom_x_drift, 0.15, f"Phantom X-coupling {max_phantom_x_drift*100:.2f}cm exceeds 15cm limit")

    def test_scenario_4_dynamic_pedestrians_and_occlusion(self):
        """
        Verify robustness against pedestrians crossing the LiDAR FOV and severe
        close-range static occlusion (40% forward FOV blocked).
        Filter must reject dynamic points and cleanly recover upon departure.
        """
        np.random.seed(42)
        history = run_scenario_4_dynamic_pedestrians()

        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])

        self.assertLess(ate_rmse, 0.15, f"Dynamic ATE RMSE {ate_rmse*100:.2f}cm exceeds 15cm threshold")
        self.assertLess(final_err, 0.08, f"Post-occlusion recovery drift {final_err*100:.2f}cm exceeds 8cm threshold")

    def test_scenario_5_low_cost_imu_artifacts(self):
        """
        Verify resilience against severe low-cost sensor biases (5x gyro bias, 5x accel bias),
        14.3 Hz motor resonance harmonics (0.35 m/s^2 amplitude), and random sample jitter.
        """
        np.random.seed(42)
        history = run_scenario_5_low_cost_imu()

        ate_rmse = float(np.sqrt(np.mean(np.array(history['error_p'])**2)))
        final_err = float(history['error_p'][-1])
        final_yaw_err_deg = float(np.rad2deg(history['error_yaw'][-1]))

        self.assertLess(ate_rmse, 0.12, f"Low-cost IMU ATE RMSE {ate_rmse*100:.2f}cm exceeds 12cm threshold")
        self.assertLess(final_err, 0.15, f"Final drift {final_err*100:.2f}cm exceeds 15cm threshold")
        self.assertLess(final_yaw_err_deg, 3.0, f"Final yaw error {final_yaw_err_deg:.2f} deg exceeds 3.0 deg")


if __name__ == "__main__":
    unittest.main()
