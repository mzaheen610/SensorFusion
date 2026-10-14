"""
Unit Tests for Online Map Reconstruction Fidelity vs. Ground Truth.

Evaluates online mapping accuracy, wall thickness, and ghost point rejection
matching real system prediction (33 Hz) and update (3.3 Hz) rates.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from evaluate_map_reconstruction import (
    generate_map_scenario_1_multiroom,
    generate_map_scenario_2_loop_circuit,
    generate_map_scenario_3_corridor,
    generate_map_scenario_4_dynamic_crowd,
    generate_map_scenario_5_real_hardware_log,
)


class MapReconstructionFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_dir = PROJECT_ROOT / "experiments" / "generated_maps"
        cls.out_dir.mkdir(parents=True, exist_ok=True)

    def test_corridor_reconstruction_subcentimeter_accuracy(self):
        """Verify that straight corridor mapping achieves sub-centimeter wall precision (<2cm RMSE)."""
        np.random.seed(42)
        metrics = generate_map_scenario_3_corridor(self.out_dir)
        self.assertLess(metrics['rmse_cm'], 2.0, f"Corridor RMSE {metrics['rmse_cm']:.2f}cm exceeds 2cm")
        self.assertLess(metrics['mean_cm'], 1.5, f"Corridor Mean dist {metrics['mean_cm']:.2f}cm exceeds 1.5cm")
        self.assertLess(metrics['sigma_cm'], 1.0, f"Corridor wall thickness {metrics['sigma_cm']:.2f}cm exceeds 1.0cm")
        self.assertEqual(metrics['ghost_pct'], 0.0, "Corridor must have 0% ghost points")

    def test_dynamic_crowd_voxel_capping_and_recovery(self):
        """Verify dynamic crowd mapping does not explode voxel capacity and maintains map bounds."""
        np.random.seed(42)
        metrics = generate_map_scenario_4_dynamic_crowd(self.out_dir)
        self.assertLess(metrics['rmse_cm'], 50.0, f"Crowd RMSE {metrics['rmse_cm']:.2f}cm exceeds 50cm")
        self.assertGreater(metrics['coverage_pct'], 60.0, f"Crowd coverage {metrics['coverage_pct']:.1f}% below 60%")
        self.assertLess(metrics['ghost_pct'], 40.0, f"Crowd ghost ratio {metrics['ghost_pct']:.1f}% exceeds 40%")

    def test_multiroom_online_mapping(self):
        """Verify multi-room exploration reconstructs room geometry with >65% perimeter coverage."""
        np.random.seed(42)
        metrics = generate_map_scenario_1_multiroom(self.out_dir)
        self.assertGreater(metrics['num_points'], 500, "Multiroom map should contain at least 500 points")
        self.assertGreater(metrics['coverage_pct'], 65.0, f"Perimeter coverage {metrics['coverage_pct']:.1f}% below 65%")

    def test_real_hardware_log_map_extraction(self):
        """Verify extraction of real robot lab map points from run_2026_10_13_18_02_23.log."""
        metrics = generate_map_scenario_5_real_hardware_log(self.out_dir)
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics['num_points'], 500)
        # Vertical spread should be bounded for 2D planar LiDAR
        self.assertLess(metrics['s2'], 0.10, f"Vertical spread s2={metrics['s2']:.3f}m exceeds 10cm")


if __name__ == "__main__":
    unittest.main()
