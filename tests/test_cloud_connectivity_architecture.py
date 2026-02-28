import os
import subprocess
import unittest
from pathlib import Path


class CloudConnectivityArchitectureTests(unittest.TestCase):
    def test_cloud_connectivity_script_has_provider_model(self):
        text = Path("scripts/cloud_connectivity_health_checks.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CLOUD_CHECK_PROVIDER", text)
        self.assertIn("providers/${PROVIDER}.sh", text)
        self.assertIn("runpod|vast", text)
        self.assertIn("cloud_connectivity_checks", text)

    def test_provider_modules_exist(self):
        runpod = Path("scripts/cloud_checks/providers/runpod.sh")
        vast = Path("scripts/cloud_checks/providers/vast.sh")
        self.assertTrue(runpod.is_file())
        self.assertTrue(vast.is_file())
        self.assertIn("runpod_provider_local_checks()", runpod.read_text(encoding="utf-8"))
        self.assertIn("vast_provider_local_checks()", vast.read_text(encoding="utf-8"))

    def test_cloud_connectivity_local_runpod_mode_completes(self):
        env = os.environ.copy()
        env["CLOUD_CHECK_PROVIDER"] = "runpod"
        env["CLOUD_CHECK_ENABLE_LIVE"] = "0"
        env["CLOUD_CHECK_TIMEOUT_SECONDS"] = "8"
        env["CLOUD_CHECK_RUN_ID"] = "test-run"
        proc = subprocess.run(
            ["bash", "scripts/cloud_connectivity_health_checks.sh"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("[cloud-check] checks completed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
