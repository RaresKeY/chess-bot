import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class VastCycleScriptTests(unittest.TestCase):
    def _latest_registry_event(self, path: Path) -> dict:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return lines[-1]

    def test_cycle_scripts_exist(self):
        names = [
            "scripts/vast_provision.py",
            "scripts/vast_cycle_common.sh",
            "scripts/vast_cycle_start.sh",
            "scripts/vast_cycle_stop.sh",
            "scripts/vast_cycle_terminate.sh",
            "scripts/vast_cycle_status.sh",
            "scripts/vast_cli_doctor.sh",
            "scripts/vast_regression_checks.sh",
            "deploy/vast_cloud_training/README.md",
            "deploy/vast_cloud_training/PLAN.md",
            "deploy/vast_cloud_training/env.example",
            "deploy/vast_cloud_training/train_baseline_preset.sh",
        ]
        for name in names:
            self.assertTrue(Path(name).is_file(), name)

    def test_common_api_token_prefers_env_without_keyring_call(self):
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "source scripts/vast_cycle_common.sh && "
                "export VAST_API_KEY='env-token-123' && "
                "out=\"$(vast_cycle_api_token /definitely/not/a/python)\" && "
                "printf '%s' \"$out\"",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.stdout, "env-token-123")

    def test_common_api_token_falls_back_to_dotenv(self):
        with tempfile.TemporaryDirectory() as td:
            dotenv = Path(td) / ".env.vast"
            dotenv.write_text("VAST_API_KEY=dotenv-token-999\n", encoding="utf-8")
            env = os.environ.copy()
            env["VAST_DOTENV_PATH"] = str(dotenv)
            env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "source scripts/vast_cycle_common.sh && "
                    "unset VAST_API_KEY && "
                    "out=\"$(vast_cycle_api_token python3 /work)\" && "
                    "printf '%s' \"$out\"",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.stdout, "dotenv-token-999")

    def test_cycle_scripts_use_vast_namespaced_registry(self):
        text = Path("scripts/vast_cycle_common.sh").read_text(encoding="utf-8")
        self.assertIn("config/vast_tracked_instances.jsonl", text)
        self.assertNotIn("runpod_tracked_pods", text)

    def test_stop_and_terminate_use_vast_provision_helper(self):
        stop = Path("scripts/vast_cycle_stop.sh").read_text(encoding="utf-8")
        self.assertIn("manage-instance", stop)
        self.assertIn("--state stopped", stop)

        terminate = Path("scripts/vast_cycle_terminate.sh").read_text(encoding="utf-8")
        self.assertIn("destroy-instance", terminate)

    def test_registry_record_writer_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True, exist_ok=True)
            script = (
                "source scripts/vast_cycle_common.sh && "
                f"vast_cycle_registry_record '{root}' 'test.sh' 'start' 'RUNNING' '123' 'run-1' 'label-a' '1.2.3.4' '2222' 'note'"
            )
            subprocess.run(["bash", "-lc", script], check=True, capture_output=True, text=True)
            event = self._latest_registry_event(root / "config" / "vast_tracked_instances.jsonl")
            self.assertEqual(event["state"], "RUNNING")
            self.assertEqual(event["instance_id"], 123)
            self.assertEqual(event["ssh_port"], "2222")


if __name__ == "__main__":
    unittest.main()
