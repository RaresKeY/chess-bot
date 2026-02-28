from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hf_dataset_publish import _resolve_dataset_dirs, _resolve_token, _validate_runtime_splice_cache


def _touch_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


def _write_valid_game_dataset(root: Path, name: str) -> Path:
    ds = root / name
    ds.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (ds / f"{split}.jsonl").write_text('{"game_id":"g1","moves":["e2e4","e7e5"]}\n', encoding="utf-8")
        split_dir = ds / "runtime_splice_cache" / split
        _touch_bytes(split_dir / "path_ids.u32.bin", 4)
        _touch_bytes(split_dir / "offsets.u64.bin", 8)
        _touch_bytes(split_dir / "splice_indices.u32.bin", 4)
        _touch_bytes(split_dir / "sample_phase_ids.u8.bin", 1)
        (split_dir / "paths.json").write_text(json.dumps([str((ds / f"{split}.jsonl").resolve())]), encoding="utf-8")
    (ds / "runtime_splice_cache" / "manifest.json").write_text(
        json.dumps({"kind": "runtime_splice_cache", "splits": {"train": {}, "val": {}, "test": {}}}),
        encoding="utf-8",
    )
    return ds


def test_resolve_dataset_dirs_single_and_root(tmp_path: Path) -> None:
    ds = _write_valid_game_dataset(tmp_path, "elite_2025-11_game")
    _write_valid_game_dataset(tmp_path, "elite_2025-12_game")
    out = _resolve_dataset_dirs(str(ds), str(tmp_path), "elite_*_game")
    assert str(ds.resolve()) in [str(p) for p in out]
    assert len(out) == 2


def test_validate_runtime_splice_cache_requires_manifest(tmp_path: Path) -> None:
    ds = tmp_path / "elite_2025-11_game"
    ds.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
        _validate_runtime_splice_cache(ds)


def test_validate_runtime_splice_cache_accepts_valid_layout(tmp_path: Path) -> None:
    ds = _write_valid_game_dataset(tmp_path, "elite_2025-11_game")
    _validate_runtime_splice_cache(ds)


def test_resolve_token_publish_uses_dotenv_fallback(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env.hf_dataset"
    dotenv.write_text("HF_WRITE_TOKEN=publish-dotenv-token\n", encoding="utf-8")
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("src.chessbot.secrets.token_from_keyring", lambda *_: "")
    monkeypatch.setattr("scripts.hf_dataset_publish.default_dotenv_paths", lambda **_: [dotenv])
    assert _resolve_token(None, "huggingface", "codex_hf_write_token") == "publish-dotenv-token"


def test_resolve_token_publish_missing_includes_dotenv_paths(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env.hf_dataset"
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("src.chessbot.secrets.token_from_keyring", lambda *_: "")
    monkeypatch.setattr("scripts.hf_dataset_publish.default_dotenv_paths", lambda **_: [dotenv])
    with pytest.raises(SystemExit) as exc_info:
        _resolve_token(None, "huggingface", "codex_hf_write_token")
    assert "dotenv paths" in str(exc_info.value)
