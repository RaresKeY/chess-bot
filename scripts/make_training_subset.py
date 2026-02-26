#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.dataset_subset import build_jsonl_subset
from src.chessbot.io_utils import ensure_parent, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small train/val JSONL subsets for faster local experiments")
    parser.add_argument("--train-in", required=True, help="Input train.jsonl")
    parser.add_argument("--val-in", required=True, help="Input val.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory for subset train/val files and summary")
    parser.add_argument("--train-rows", type=int, default=20000, help="Max train rows to write")
    parser.add_argument("--val-rows", type=int, default=2000, help="Max val rows to write")
    parser.add_argument("--min-target-len", type=int, default=1, help="Require target length >= this value")
    parser.add_argument(
        "--exact-target-len",
        type=int,
        default=0,
        help="Require exact target length (0 disables exact length filtering)",
    )
    parser.add_argument("--summary-out", default="", help="Optional explicit summary JSON path")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_out = out_dir / "train.jsonl"
    val_out = out_dir / "val.jsonl"
    summary_out = Path(args.summary_out) if args.summary_out else (out_dir / "subset_summary.json")
    exact_target_len = None if int(args.exact_target_len) <= 0 else int(args.exact_target_len)

    train_res = build_jsonl_subset(
        args.train_in,
        str(train_out),
        max_rows=int(args.train_rows),
        min_target_len=int(args.min_target_len),
        exact_target_len=exact_target_len,
    )
    val_res = build_jsonl_subset(
        args.val_in,
        str(val_out),
        max_rows=int(args.val_rows),
        min_target_len=int(args.min_target_len),
        exact_target_len=exact_target_len,
    )

    summary = {
        "train": train_res.__dict__,
        "val": val_res.__dict__,
        "out_dir": str(out_dir.resolve()),
        "filters": {
            "min_target_len": int(args.min_target_len),
            "exact_target_len": exact_target_len,
        },
    }
    ensure_parent(str(summary_out))
    write_json(str(summary_out), summary)
    print(summary)


if __name__ == "__main__":
    main()
