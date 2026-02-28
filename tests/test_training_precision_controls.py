import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.chessbot.training import _resolve_amp_autocast_dtype


def test_train_baseline_help_exposes_precision_flags() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/train_baseline.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--amp-dtype" in proc.stdout
    assert "--tf32" in proc.stdout


def test_resolve_amp_autocast_dtype_disabled_amp_returns_none() -> None:
    out = _resolve_amp_autocast_dtype("fp16", use_amp=False, device=torch.device("cpu"))
    assert out is None


def test_resolve_amp_autocast_dtype_fp16() -> None:
    out = _resolve_amp_autocast_dtype("fp16", use_amp=True, device=torch.device("cuda"))
    assert out == torch.float16


def test_resolve_amp_autocast_dtype_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_amp_autocast_dtype("wat", use_amp=True, device=torch.device("cuda"))
