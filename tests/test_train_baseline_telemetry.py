import importlib.util
import pathlib
import unittest


def _load_train_baseline_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "train_baseline.py"
    spec = importlib.util.spec_from_file_location("train_baseline_script", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class TrainBaselineTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_train_baseline_module()

    def test_parse_nvidia_smi_csv_line(self):
        line = "0, NVIDIA GeForce RTX 2080 Ti, 67, 2100, 11264, 79, 118.25, 250.00"
        parsed = self.mod._parse_nvidia_smi_csv_line(line)
        self.assertEqual(parsed["gpu_index"], 0)
        self.assertEqual(parsed["gpu_name"], "NVIDIA GeForce RTX 2080 Ti")
        self.assertEqual(parsed["gpu_util_percent"], 67.0)
        self.assertEqual(parsed["vram_used_mib"], 2100.0)
        self.assertEqual(parsed["vram_total_mib"], 11264.0)
        self.assertEqual(parsed["gpu_temp_c"], 79.0)

    def test_summarize_telemetry_samples(self):
        samples = [
            {"ts_ms": 1000, "gpu_util_percent": 10, "vram_used_mib": 100, "proc_rss_kib": 200000, "epoch": 1},
            {"ts_ms": 3000, "gpu_util_percent": 30, "vram_used_mib": 300, "proc_rss_kib": 220000, "epoch": 2},
            {"ts_ms": 5000, "gpu_util_percent": 20, "vram_used_mib": 200, "proc_rss_kib": 210000, "epoch": 2},
        ]
        out = self.mod._summarize_telemetry_samples(samples)
        self.assertEqual(out["sample_count"], 3)
        self.assertEqual(out["duration_seconds"], 4.0)
        gpu_util = out["metrics"]["gpu_util_percent"]
        self.assertEqual(gpu_util["min"], 10.0)
        self.assertEqual(gpu_util["max"], 30.0)
        self.assertAlmostEqual(gpu_util["mean"], 20.0)


if __name__ == "__main__":
    unittest.main()

