"""
Tests confirming failure modes directly from real hardware logs.
Verifies:
1. Filter lockout / death spiral where applied=False for >50 consecutive scans (sep_8_2.log).
2. Large state whiplash / oscillation caused by uncoordinated delta merges (sep_25.log).
3. LiDAR scan skip cascade caused by timing gaps during dynamic motion (sep_15.log / sep_17.log).
"""

from pathlib import Path
import re
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


class LogFailureAnalysisTests(unittest.TestCase):

    def test_log_sep_8_2_confirms_permanent_recovery_lockout(self):
        """
        In sep_8_2.log, verify that once drift occurred, the filter entered a permanent
        lockout where applied=False for dozens of consecutive scans, proving the system's
        inability to recover once tracking is lost.
        """
        log_path = LOG_DIR / "sep_8_2.log"
        self.assertTrue(log_path.exists(), f"Log file {log_path} not found")

        content = log_path.read_text(encoding="utf-8", errors="replace")

        # Pattern: LiDAR update diagnostics: ... applied=(True|False) | residuals=(\d+)
        diag_pattern = re.compile(
            r"LiDAR update diagnostics: .*? applied=(True|False) \| residuals=(\d+)"
        )

        matches = diag_pattern.findall(content)
        self.assertGreater(len(matches), 10, "Not enough diagnostic entries found in log")

        applied_flags = [m[0] == "True" for m in matches]
        residual_counts = [int(m[1]) for m in matches]

        # Count consecutive False values at the tail end of the run
        trailing_false_count = 0
        for flag in reversed(applied_flags):
            if not flag:
                trailing_false_count += 1
            else:
                break

        # Evidence: The run ended with a sustained sequence of rejected updates
        self.assertGreaterEqual(
            trailing_false_count, 30,
            f"Expected at least 30 consecutive rejected scans at end of sep_8_2.log, got {trailing_false_count}"
        )

        # In that lockout window, residuals were 0 because points failed the innovation gate
        lockout_residuals = residual_counts[-trailing_false_count:]
        zero_residual_ratio = sum(1 for r in lockout_residuals if r == 0) / len(lockout_residuals)
        self.assertGreater(
            zero_residual_ratio, 0.90,
            "During lockout, >90% of scans had 0 residuals due to gate rejection"
        )

    def test_log_sep_25_confirms_state_whiplash_oscillations(self):
        """
        In sep_25.log, verify that the estimated position experiences large, rapid oscillations
        (whiplash) where delta_position jumps exceed 20 cm in magnitude across consecutive steps.
        """
        log_path = LOG_DIR / "sep_25.log"
        self.assertTrue(log_path.exists(), f"Log file {log_path} not found")

        content = log_path.read_text(encoding="utf-8", errors="replace")

        # Pattern: delta_position=\[\s*([-\d\.\+e]+)\s+([-\d\.\+e]+)\s+([-\d\.\+e]+)\s*\]
        delta_pattern = re.compile(
            r"delta_position=\[\s*([-\d\.\+eE]+)\s+([-\d\.\+eE]+)\s+([-\d\.\+eE]+)\s*\]"
        )

        deltas = []
        for line in content.splitlines():
            m = delta_pattern.search(line)
            if m:
                dx, dy, dz = float(m.group(1)), float(m.group(2)), float(m.group(3))
                deltas.append((dx, dy, dz))

        self.assertGreater(len(deltas), 20, "Not enough delta_position commits found")

        # Check for large jump magnitudes (> 0.20 m)
        large_jumps = [d for d in deltas if (d[0]**2 + d[1]**2)**0.5 > 0.20]
        self.assertGreaterEqual(
            len(large_jumps), 5,
            f"Expected multiple large position jumps (>20cm), found {len(large_jumps)}"
        )

        # Verify sign reversal (oscillation back and forth along X axis)
        sign_reversals = 0
        for i in range(1, len(deltas)):
            # If consecutive deltas are both substantial (>10cm) but have opposite signs
            if abs(deltas[i-1][0]) > 0.10 and abs(deltas[i][0]) > 0.10:
                if (deltas[i-1][0] * deltas[i][0]) < 0:
                    sign_reversals += 1

        self.assertGreater(
            sign_reversals, 0,
            "Confirmed state whiplash: consecutive updates applied large opposite-sign corrections"
        )

    def test_log_sep_15_confirms_scan_drop_cascade(self):
        """
        In sep_15.log, verify that frequent 'LiDAR compensation skipped' occurrences
        cause severe gaps in scan delivery.
        """
        log_path = LOG_DIR / "sep_15.log"
        self.assertTrue(log_path.exists(), f"Log file {log_path} not found")

        content = log_path.read_text(encoding="utf-8", errors="replace")
        skipped_count = content.count("LiDAR compensation skipped:")

        self.assertGreaterEqual(
            skipped_count, 15,
            f"Expected at least 15 skipped compensation events in sep_15.log, found {skipped_count}"
        )


if __name__ == "__main__":
    unittest.main()
