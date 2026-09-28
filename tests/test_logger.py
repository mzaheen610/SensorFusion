import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensorfusion"))

from utils.logger import setup_runtime_logger, teardown_runtime_logger


class RuntimeLoggerTests(unittest.TestCase):
    def tearDown(self):
        teardown_runtime_logger()

    def test_dual_logging_captures_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = setup_runtime_logger(logs_dir=tmp_dir, prefix="test_run_")
            self.assertTrue(os.path.isfile(log_file))

            test_message = "Fusion test output 12345"
            err_message = "Sensor warning 67890"

            print(test_message)
            sys.stderr.write(err_message + "\n")

            teardown_runtime_logger()

            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(test_message, content)
            self.assertIn(err_message, content)
            self.assertIn("[Logger] Runtime logging started", content)

    def test_timestamp_filename_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = setup_runtime_logger(logs_dir=tmp_dir, prefix="robot_run_")
            filename = os.path.basename(log_file)

            self.assertTrue(filename.startswith("robot_run_"))
            self.assertTrue(filename.endswith(".log"))
            # Expected format: robot_run_YYYY_MM_DD_HH_MM_SS.log (32 chars)
            self.assertGreaterEqual(len(filename), 25)
            teardown_runtime_logger()


if __name__ == "__main__":
    unittest.main()
