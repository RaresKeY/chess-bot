import unittest
from pathlib import Path


class RunpodEntrypointScriptTests(unittest.TestCase):
    def test_entrypoint_handles_non_git_repo_dir_and_skips_inference_without_torch(self):
        text = Path("deploy/runpod_cloud_training/entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn("REPO_DIR exists and is non-empty but not a git repo", text)
        self.assertIn("python_module_available()", text)
        self.assertIn("Skipping inference API start: torch is not installed", text)
        self.assertIn("ensure_runner_ssh_account()", text)
        self.assertIn("Unlocked ${RUNNER_USER} account for SSH public-key auth", text)
        self.assertIn("AuthorizedKeysFile .ssh/authorized_keys", text)


if __name__ == "__main__":
    unittest.main()
