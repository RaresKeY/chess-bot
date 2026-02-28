import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_runtime_splice_vocab_meta_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "tiny_game"
    ds.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        ds / "train.jsonl",
        [
            {"game_id": "g1", "moves": ["e2e4", "e7e5", "g1f3", "b8c6"]},
            {"game_id": "g2", "moves": ["d2d4", "d7d5", "c2c4", "e7e6"]},
        ],
    )
    _write_jsonl(
        ds / "val.jsonl",
        [
            {"game_id": "g3", "moves": ["c2c4", "e7e5", "b1c3", "g8f6"]},
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_runtime_splice_vocab_meta.py",
            "--dataset-dir",
            str(ds),
            "--jobs",
            "2",
            "--chunk-size-mb",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out_path = ds / "runtime_splice_cache" / "vocab_rows_meta.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "runtime_splice_vocab_rows_meta_v1"
    assert payload["splits"]["train"]["game_rows"] == 2
    assert payload["splits"]["val"]["game_rows"] == 1
    assert "e2e4" in payload["vocab"]
