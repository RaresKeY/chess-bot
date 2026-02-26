import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.chessbot.io_utils import ensure_parent


@dataclass
class SubsetBuildResult:
    input_path: str
    output_path: str
    rows_scanned: int
    rows_written: int
    rows_target_rejected: int
    rows_invalid_json: int
    min_target_len: int
    exact_target_len: Optional[int]


def _target_len_ok(row: Dict, min_target_len: int, exact_target_len: Optional[int]) -> bool:
    target = row.get("target", [])
    if not isinstance(target, list):
        return False
    tlen = len(target)
    if exact_target_len is not None and tlen != int(exact_target_len):
        return False
    if tlen < int(min_target_len):
        return False
    return True


def build_jsonl_subset(
    input_path: str,
    output_path: str,
    *,
    max_rows: int,
    min_target_len: int = 1,
    exact_target_len: Optional[int] = None,
) -> SubsetBuildResult:
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0")
    min_target_len = int(min_target_len)
    if min_target_len < 0:
        raise ValueError("min_target_len must be >= 0")
    if exact_target_len is not None:
        exact_target_len = int(exact_target_len)
        if exact_target_len < 0:
            raise ValueError("exact_target_len must be >= 0")

    rows_scanned = 0
    rows_written = 0
    rows_target_rejected = 0
    rows_invalid_json = 0

    in_path = Path(input_path)
    out_path = Path(output_path)
    ensure_parent(str(out_path))
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            rows_scanned += 1
            try:
                row = json.loads(line)
            except Exception:
                rows_invalid_json += 1
                continue
            if not isinstance(row, dict) or not _target_len_ok(row, min_target_len=min_target_len, exact_target_len=exact_target_len):
                rows_target_rejected += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows_written += 1
            if rows_written >= max_rows:
                break

    return SubsetBuildResult(
        input_path=str(in_path),
        output_path=str(out_path),
        rows_scanned=rows_scanned,
        rows_written=rows_written,
        rows_target_rejected=rows_target_rejected,
        rows_invalid_json=rows_invalid_json,
        min_target_len=min_target_len,
        exact_target_len=exact_target_len,
    )
