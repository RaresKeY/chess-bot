import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - environment-dependent skip path
    torch = None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TrainDualSequenceCliTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_dual_sequence_help_exposes_flags(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "scripts/train_dual_sequence.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--side-mode", proc.stdout)
        self.assertIn("--horizon", proc.stdout)
        self.assertIn("--mate-in-x", proc.stdout)
        self.assertIn("--model-family", proc.stdout)
        self.assertIn("--architecture", proc.stdout)
        self.assertIn("--bootstrap-epochs", proc.stdout)
        self.assertIn("--out-model-white", proc.stdout)
        self.assertIn("--out-model-black", proc.stdout)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_dual_sequence_cli_both_sides(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.jsonl"
            val_path = tmp_path / "val.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5", "g1f3"], "winner_side": "W", "phase": "opening"},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"], "winner_side": "B", "phase": "opening"},
            ]
            _write_jsonl(train_path, rows)
            _write_jsonl(val_path, rows)

            out_white = tmp_path / "model_white.pt"
            out_black = tmp_path / "model_black.pt"
            out_metrics = tmp_path / "train_metrics_dual_sequence.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_dual_sequence.py",
                    "--train",
                    str(train_path),
                    "--val",
                    str(val_path),
                    "--side-mode",
                    "both",
                    "--model-family",
                    "dual_side_sequence_board_lstm",
                    "--horizon",
                    "4",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--embed-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-layers",
                    "1",
                    "--dropout",
                    "0.0",
                    "--mate-bias",
                    "--out-model-white",
                    str(out_white),
                    "--out-model-black",
                    str(out_black),
                    "--out-metrics",
                    str(out_metrics),
                    "--device",
                    "cpu",
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(out_white.is_file())
            self.assertTrue(out_black.is_file())
            self.assertTrue(out_metrics.is_file())
            payload = json.loads(out_metrics.read_text(encoding="utf-8"))
            self.assertEqual(payload["side_mode"], "both")
            self.assertEqual(payload["runtime"]["model_family"], "dual_side_sequence_board_lstm")
            self.assertIn("white", payload["sides"])
            self.assertIn("black", payload["sides"])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_dual_sequence_cli_allplay_bootstrap_architecture(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train_bootstrap.jsonl"
            val_path = tmp_path / "val_bootstrap.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5", "g1f3"], "winner_side": "W", "phase": "opening"},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"], "winner_side": "B", "phase": "opening"},
                {"context": ["c2c4"], "target": ["e7e6", "d2d4"], "winner_side": "D", "phase": "middlegame"},
            ]
            _write_jsonl(train_path, rows)
            _write_jsonl(val_path, rows)

            out_shared = tmp_path / "model_shared.pt"
            out_white = tmp_path / "model_white.pt"
            out_black = tmp_path / "model_black.pt"
            out_metrics = tmp_path / "train_metrics_bootstrap.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_dual_sequence.py",
                    "--train",
                    str(train_path),
                    "--val",
                    str(val_path),
                    "--architecture",
                    "allplay_bootstrap_side_finetune_curriculum",
                    "--side-mode",
                    "both",
                    "--model-family",
                    "dual_side_sequence_lstm",
                    "--horizon",
                    "4",
                    "--bootstrap-epochs",
                    "1",
                    "--finetune-epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--embed-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-layers",
                    "1",
                    "--dropout",
                    "0.0",
                    "--out-model-shared",
                    str(out_shared),
                    "--out-model-white",
                    str(out_white),
                    "--out-model-black",
                    str(out_black),
                    "--out-metrics",
                    str(out_metrics),
                    "--device",
                    "cpu",
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(out_shared.is_file())
            self.assertTrue(out_white.is_file())
            self.assertTrue(out_black.is_file())
            payload = json.loads(out_metrics.read_text(encoding="utf-8"))
            self.assertEqual(payload["architecture"], "allplay_bootstrap_side_finetune_curriculum")
            self.assertEqual(payload["training_mode"], "dual_bootstrap_curriculum")
            self.assertEqual(payload["runtime"]["model_family"], "allplay_bootstrap_dualhead_curriculum_lstm")
            self.assertIn("white", payload["sides"])
            self.assertIn("black", payload["sides"])


if __name__ == "__main__":
    unittest.main()
