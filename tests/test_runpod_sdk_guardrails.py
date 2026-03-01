import unittest
from pathlib import Path


class RunpodSdkGuardrailTests(unittest.TestCase):
    def test_sdk_start_uses_sdk_component_only(self):
        text = Path("scripts/runpod_sdk_cycle_start.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/runpod_sdk_component.py", text)
        self.assertNotIn("scripts/runpod_provision.py", text)

    def test_sdk_full_smoke_uses_sdk_start_stop_only(self):
        text = Path("scripts/runpod_sdk_cycle_full_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("runpod_sdk_cycle_start.sh", text)
        self.assertIn("runpod_sdk_cycle_stop.sh", text)
        self.assertNotIn("runpod_cycle_start.sh", text)
        self.assertNotIn("runpod_cycle_stop.sh", text)

    def test_sdk_stop_uses_sdk_component_only(self):
        text = Path("scripts/runpod_sdk_cycle_stop.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/runpod_sdk_component.py", text)
        self.assertNotIn("scripts/runpod_provision.py", text)

