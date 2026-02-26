import unittest
from pathlib import Path


class RunpodCycleScriptTests(unittest.TestCase):
    def test_cycle_scripts_exist(self):
        names = [
            "scripts/runpod_cycle_common.sh",
            "scripts/runpod_cycle_start.sh",
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_local_validate.sh",
            "scripts/runpod_cycle_stop.sh",
            "scripts/runpod_cycle_full_smoke.sh",
        ]
        for name in names:
            self.assertTrue(Path(name).is_file(), name)

    def test_full_smoke_calls_modular_steps(self):
        text = Path("scripts/runpod_cycle_full_smoke.sh").read_text(encoding="utf-8")
        for name in [
            "runpod_cycle_start.sh",
            "runpod_cycle_push_dataset.sh",
            "runpod_cycle_train.sh",
            "runpod_cycle_collect.sh",
            "runpod_cycle_local_validate.sh",
            "runpod_cycle_stop.sh",
        ]:
            self.assertIn(name, text)

    def test_stop_script_uses_graphql_pod_stop(self):
        text = Path("scripts/runpod_cycle_stop.sh").read_text(encoding="utf-8")
        self.assertIn("mutation StopPod", text)
        self.assertIn("podStop", text)


if __name__ == "__main__":
    unittest.main()
