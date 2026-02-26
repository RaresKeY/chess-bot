import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.chessbot.dataset_subset import build_jsonl_subset


class DatasetSubsetTests(unittest.TestCase):
    def _write_rows(self, path: Path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_build_jsonl_subset_filters_by_min_target_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            out = Path(tmp) / "out.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5"]},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"]},
                {"context": ["c2c4"], "target": ["e7e5", "b1c3", "g8f6"]},
            ]
            self._write_rows(src, rows)
            res = build_jsonl_subset(str(src), str(out), max_rows=10, min_target_len=2)
            self.assertEqual(res.rows_written, 2)
            self.assertGreaterEqual(res.rows_target_rejected, 1)
            written = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(len(written), 2)
            self.assertTrue(all(len(r["target"]) >= 2 for r in written))

    def test_build_jsonl_subset_exact_target_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            out = Path(tmp) / "out.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5"]},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"]},
                {"context": ["c2c4"], "target": ["e7e5", "b1c3"]},
            ]
            self._write_rows(src, rows)
            res = build_jsonl_subset(str(src), str(out), max_rows=10, min_target_len=1, exact_target_len=2)
            self.assertEqual(res.rows_written, 2)
            written = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(all(len(r["target"]) == 2 for r in written))

    def test_make_training_subset_cli_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train_in = base / "train_in.jsonl"
            val_in = base / "val_in.jsonl"
            out_dir = base / "subset"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5", "g1f3", "b8c6", "f1b5"]},
                {"context": ["d2d4"], "target": ["d7d5"]},
                {"context": ["c2c4"], "target": ["e7e5", "b1c3", "g8f6", "g2g3"]},
            ]
            self._write_rows(train_in, rows)
            self._write_rows(val_in, rows)
            cmd = [
                "python3",
                "scripts/make_training_subset.py",
                "--train-in",
                str(train_in),
                "--val-in",
                str(val_in),
                "--out-dir",
                str(out_dir),
                "--train-rows",
                "2",
                "--val-rows",
                "1",
                "--min-target-len",
                "4",
            ]
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            summary = json.loads((out_dir / "subset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train"]["rows_written"], 2)
            self.assertEqual(summary["val"]["rows_written"], 1)
            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "val.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
