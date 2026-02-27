from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_bytes(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * n)


def _create_valid_cache_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (dataset_dir / f"{split}.jsonl").write_text('{"game_id":"g1","moves":["e2e4","e7e5"]}\n', encoding="utf-8")
        split_dir = dataset_dir / "runtime_splice_cache" / split
        rows = 3
        _write_bytes(split_dir / "path_ids.u32.bin", rows * 4)
        _write_bytes(split_dir / "offsets.u64.bin", rows * 8)
        _write_bytes(split_dir / "splice_indices.u32.bin", rows * 4)
        _write_bytes(split_dir / "sample_phase_ids.u8.bin", rows)
        (split_dir / "paths.json").write_text(json.dumps([str((dataset_dir / f"{split}.jsonl").resolve())]), encoding="utf-8")
    (dataset_dir / "runtime_splice_cache" / "manifest.json").write_text(
        json.dumps({"kind": "runtime_splice_cache"}),
        encoding="utf-8",
    )


def test_validate_runtime_splice_cache_ok() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "elite_2025-11_game"
        _create_valid_cache_dataset(ds)
        cmd = [sys.executable, "scripts/validate_runtime_splice_cache.py", "--dataset-dir", str(ds)]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
        out = json.loads(proc.stdout)
        assert out["failed_count"] == 0
        assert out["ok_count"] == 1


def test_validate_runtime_splice_cache_detects_row_mismatch() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "elite_2025-11_game"
        _create_valid_cache_dataset(ds)
        # Break train split row alignment.
        _write_bytes(ds / "runtime_splice_cache" / "train" / "sample_phase_ids.u8.bin", 2)
        cmd = [sys.executable, "scripts/validate_runtime_splice_cache.py", "--dataset-dir", str(ds)]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
        assert proc.returncode != 0
        out = json.loads(proc.stdout)
        assert out["failed_count"] == 1
        all_errors = "\n".join(out["results"][0]["errors"])
        assert "row_count_mismatch" in all_errors

