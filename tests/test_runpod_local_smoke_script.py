import unittest
from pathlib import Path


class RunpodLocalSmokeScriptTests(unittest.TestCase):
    def test_uses_conventional_phase_timing_log_path_and_prep_phase(self):
        script_path = Path("scripts/runpod_local_smoke_test.sh")
        text = script_path.read_text(encoding="utf-8")
        self.assertIn("artifacts/timings/runpod_phase_times.jsonl", text)
        self.assertIn("prepare_timing_log_file()", text)
        self.assertIn('run_timed "prepare_timing_log_file" prepare_timing_log_file', text)
        self.assertIn("chmod 666", text)

    def test_can_pass_optional_progress_jsonl_to_train_preset(self):
        text = Path("scripts/runpod_local_smoke_test.sh").read_text(encoding="utf-8")
        self.assertIn('SMOKE_PROGRESS_JSONL_OUT="${SMOKE_PROGRESS_JSONL_OUT:-}"', text)
        self.assertIn('-e TRAIN_PROGRESS_JSONL_OUT="${SMOKE_PROGRESS_JSONL_OUT}"', text)
        self.assertIn('echo "[local-smoke] progress_jsonl=${SMOKE_PROGRESS_JSONL_OUT}"', text)


if __name__ == "__main__":
    unittest.main()
