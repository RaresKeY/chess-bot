import unittest
from pathlib import Path


class RunpodDirectApiGuardrailTests(unittest.TestCase):
    def test_direct_start_uses_direct_provision_component_only(self):
        text = Path("scripts/runpod_cycle_start.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/runpod_provision.py", text)
        self.assertNotIn("scripts/runpod_sdk_component.py", text)

    def test_direct_full_smoke_uses_direct_start_stop_only(self):
        text = Path("scripts/runpod_cycle_full_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("runpod_cycle_start.sh", text)
        self.assertIn("runpod_cycle_stop.sh", text)
        self.assertNotIn("runpod_sdk_cycle_start.sh", text)
        self.assertNotIn("runpod_sdk_cycle_stop.sh", text)

    def test_shared_cycle_scripts_do_not_bind_to_provision_backend(self):
        for path in (
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_local_validate.sh",
        ):
            text = Path(path).read_text(encoding="utf-8")
            self.assertNotIn("runpod_provision.py", text, path)
            self.assertNotIn("runpod_sdk_component.py", text, path)

