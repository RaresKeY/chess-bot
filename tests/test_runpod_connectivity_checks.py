import os
import subprocess
import unittest
from pathlib import Path


class RunpodConnectivityChecksTests(unittest.TestCase):
    def test_runpod_wrapper_delegates_to_cloud_connectivity_checks(self):
        text = Path("scripts/runpod_connectivity_telemetry_checks.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CLOUD_CHECK_PROVIDER", text)
        self.assertIn("RUNPOD_CONNECTIVITY_TIMEOUT_SECONDS", text)
        self.assertIn("RUNPOD_ENABLE_LIVE_CONNECTIVITY_CHECKS", text)
        self.assertIn("cloud_connectivity_health_checks.sh", text)

    def test_connectivity_telemetry_checks_local_mode_completes(self):
        env = os.environ.copy()
        env["RUNPOD_ENABLE_LIVE_CONNECTIVITY_CHECKS"] = "0"
        env["RUNPOD_CONNECTIVITY_TIMEOUT_SECONDS"] = "8"
        env["RUNPOD_CYCLE_RUN_ID"] = "test-run"
        proc = subprocess.run(
            ["bash", "scripts/runpod_connectivity_telemetry_checks.sh"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("[cloud-check] checks completed", proc.stdout)

    def test_regression_checks_include_connectivity_category(self):
        text = Path("scripts/runpod_regression_checks.sh").read_text(encoding="utf-8")
        self.assertIn("RUN_CONNECTIVITY_TELEMETRY_CHECKS", text)
        self.assertIn("RUN_CONNECTIVITY_PROVIDER", text)
        self.assertIn("cloud_connectivity_health_checks.sh", text)


if __name__ == "__main__":
    unittest.main()
