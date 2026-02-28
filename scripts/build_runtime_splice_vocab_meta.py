#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.training import RuntimeSpliceConfig, _moves_from_row, _runtime_splice_indices_for_moves  # noqa: E402


def _count_chunk(
    path: str,
    start: int,
    end: int,
    cfg: RuntimeSpliceConfig,
    collect_vocab: bool,
) -> Tuple[int, int, Set[str]]:
    game_rows = 0
    sample_rows = 0
    vocab: Set[str] = set()
    with open(path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        while True:
            offset = f.tell()
            if offset >= end:
                break
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            moves = _moves_from_row(row)
            game_rows += 1
            sample_rows += len(
                _runtime_splice_indices_for_moves(
                    moves=moves,
                    cfg=cfg,
                    game_id=str(row.get("game_id", "")),
                )
            )
            if collect_vocab:
                vocab.update(str(m) for m in moves if m)
    return game_rows, sample_rows, vocab


def _chunk_ranges(path: Path, chunk_mb: int) -> List[Tuple[int, int]]:
    size = int(path.stat().st_size)
    chunk = max(1, int(chunk_mb)) * 1024 * 1024
    out: List[Tuple[int, int]] = []
    start = 0
    while start < size:
        end = min(start + chunk, size)
        out.append((start, end))
        start = end
    if not out:
        out.append((0, 0))
    return out


def _process_split(path: Path, cfg: RuntimeSpliceConfig, jobs: int, chunk_mb: int, collect_vocab: bool) -> Tuple[int, int, Set[str]]:
    tasks = _chunk_ranges(path, chunk_mb)
    total_game_rows = 0
    total_sample_rows = 0
    tokens: Set[str] = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = [
            ex.submit(
                _count_chunk,
                str(path),
                start,
                end,
                cfg,
                collect_vocab,
            )
            for start, end in tasks
        ]
        for fut in concurrent.futures.as_completed(futures):
            g, s, t = fut.result()
            total_game_rows += int(g)
            total_sample_rows += int(s)
            if collect_vocab:
                tokens.update(t)
    return total_game_rows, total_sample_rows, tokens


def main() -> None:
    p = argparse.ArgumentParser(description="Build runtime-splice vocab+row-count metadata for game datasets.")
    p.add_argument("--dataset-dir", action="append", required=True, help="Game dataset directory (repeatable).")
    p.add_argument("--min-context", type=int, default=8)
    p.add_argument("--min-target", type=int, default=1)
    p.add_argument("--max-samples-per-game", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count()))
    p.add_argument("--chunk-size-mb", type=int, default=16)
    args = p.parse_args()

    cfg = RuntimeSpliceConfig(
        min_context=int(args.min_context),
        min_target=int(args.min_target),
        max_samples_per_game=int(args.max_samples_per_game),
        seed=int(args.seed),
    )
    jobs = max(1, int(args.jobs))

    for raw_dir in args.dataset_dir:
        dataset_dir = Path(raw_dir).resolve()
        train_path = dataset_dir / "train.jsonl"
        val_path = dataset_dir / "val.jsonl"
        if not train_path.is_file() or not val_path.is_file():
            raise SystemExit(f"Missing train/val JSONL in {dataset_dir}")

        train_games, train_samples, train_tokens = _process_split(
            train_path, cfg=cfg, jobs=jobs, chunk_mb=int(args.chunk_size_mb), collect_vocab=True
        )
        val_games, val_samples, _ = _process_split(
            val_path, cfg=cfg, jobs=jobs, chunk_mb=int(args.chunk_size_mb), collect_vocab=False
        )

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for tok in sorted(train_tokens):
            if tok not in vocab:
                vocab[tok] = len(vocab)

        out = {
            "kind": "runtime_splice_vocab_rows_meta_v1",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_dir": str(dataset_dir),
            "config": {
                "min_context": int(cfg.min_context),
                "min_target": int(cfg.min_target),
                "max_samples_per_game": int(cfg.max_samples_per_game),
                "seed": int(cfg.seed),
            },
            "splits": {
                "train": {"path": "train.jsonl", "game_rows": int(train_games), "sample_rows": int(train_samples)},
                "val": {"path": "val.jsonl", "game_rows": int(val_games), "sample_rows": int(val_samples)},
            },
            "vocab": vocab,
            "vocab_size": int(len(vocab)),
        }

        out_path = dataset_dir / "runtime_splice_cache" / "vocab_rows_meta.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "out_path": str(out_path),
                    "train_game_rows": int(train_games),
                    "train_sample_rows": int(train_samples),
                    "val_game_rows": int(val_games),
                    "val_sample_rows": int(val_samples),
                    "vocab_size": int(len(vocab)),
                    "jobs": int(jobs),
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
