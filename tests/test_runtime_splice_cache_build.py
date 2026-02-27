from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_runtime_splice_cache_smoke() -> None:
    rows = [
        {
            "game_id": "g1",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],
            "winner_side": "W",
        },
        {
            "game_id": "g2",
            "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"],
            "winner_side": "B",
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "dataset"
        _write_jsonl(ds / "train.jsonl", rows)
        _write_jsonl(ds / "val.jsonl", rows[:1])
        (ds / "stats.json").write_text(json.dumps({"dataset_format": "game_jsonl_runtime_splice_v1"}) + "\n", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/build_runtime_splice_cache.py",
            "--dataset-dir",
            str(ds),
            "--splits",
            "train,val",
            "--min-context",
            "4",
            "--min-target",
            "1",
            "--jobs",
            "2",
        ]
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

        manifest_path = ds / "runtime_splice_cache" / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["kind"] == "runtime_splice_cache"
        assert manifest["config"]["min_context"] == 4
        assert manifest["splits"]["train"]["game_rows_total"] == 2
        assert manifest["splits"]["val"]["game_rows_total"] == 1
        assert manifest["splits"]["train"]["sample_rows_total"] > 0
        assert (ds / "runtime_splice_cache" / "train" / "offsets.u64.bin").is_file()
        assert (ds / "runtime_splice_cache" / "train" / "sample_phase_ids.u8.bin").is_file()


def test_build_runtime_splice_cache_multiple_datasets() -> None:
    rows = [
        {
            "game_id": "g1",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],
            "winner_side": "W",
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds1 = root / "dataset_a"
        ds2 = root / "dataset_b"
        for ds in (ds1, ds2):
            _write_jsonl(ds / "train.jsonl", rows)
            _write_jsonl(ds / "val.jsonl", rows)
            _write_jsonl(ds / "test.jsonl", rows)
            (ds / "stats.json").write_text(
                json.dumps({"dataset_format": "game_jsonl_runtime_splice_v1"}) + "\n",
                encoding="utf-8",
            )

        cmd = [
            sys.executable,
            "scripts/build_runtime_splice_cache.py",
            "--dataset-dir",
            str(ds1),
            "--dataset-dir",
            str(ds2),
            "--jobs",
            "6",
            "--no-progress-bar",
            "--no-verbose",
        ]
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

        for ds in (ds1, ds2):
            manifest_path = ds / "runtime_splice_cache" / "manifest.json"
            assert manifest_path.is_file()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["dataset_dir"] == str(ds.resolve())
            assert set(manifest["splits"].keys()) == {"train", "val", "test"}
