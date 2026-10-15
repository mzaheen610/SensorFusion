import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from record_turn_dataset import TurnDatasetRecorder
from analyze_turn_dataset import analyze_dataset


class TurnDatasetRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mock_recording_produces_valid_npz_and_metadata(self):
        """Verify TurnDatasetRecorder creates valid .npz and .json files in mock mode."""
        recorder = TurnDatasetRecorder(
            turn_type="unit_test_turn",
            output_dir=str(self.output_dir),
            pre_dwell_sec=0.2,
            post_dwell_sec=0.2,
            mock_mode=True,
        )

        ok = recorder.initialize_sensors()
        self.assertTrue(ok)

        npz_path = recorder.record_session(duration_sec=0.5)
        self.assertIsNotNone(npz_path)
        self.assertTrue(npz_path.exists())

        # Check metadata JSON
        meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
        self.assertTrue(meta_path.exists())

        # Load and verify archive contents
        data = np.load(npz_path)
        expected_keys = [
            "imu_t_mono", "imu_gyro", "imu_lin_accel", "imu_raw_accel",
            "lidar_t_start", "lidar_t_end", "lidar_points", "lidar_scan_offsets",
            "turn_start_time", "turn_end_time"
        ]
        for key in expected_keys:
            self.assertIn(key, data)

        self.assertGreater(len(data["imu_t_mono"]), 20)
        self.assertGreater(len(data["lidar_t_end"]), 5)
        self.assertGreater(len(data["lidar_points"]), 50)
        self.assertTrue(np.all(np.isfinite(data["imu_gyro"])))
        self.assertTrue(np.all(np.isfinite(data["imu_lin_accel"])))

    def test_lidar_scan_slice_reconstruction(self):
        """Verify that scan offsets cleanly partition the continuous point cloud."""
        recorder = TurnDatasetRecorder(
            turn_type="slice_test",
            output_dir=str(self.output_dir),
            pre_dwell_sec=0.1,
            post_dwell_sec=0.1,
            mock_mode=True,
        )
        recorder.initialize_sensors()
        npz_path = recorder.record_session(duration_sec=0.3)

        data = np.load(npz_path)
        offsets = data["lidar_scan_offsets"]
        pts = data["lidar_points"]
        num_scans = len(data["lidar_t_end"])

        self.assertEqual(len(offsets), num_scans + 1)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], len(pts))

        # Check each scan slice
        for s in range(num_scans):
            scan_pts = pts[offsets[s]:offsets[s+1]]
            self.assertGreater(len(scan_pts), 0)
            self.assertEqual(scan_pts.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
