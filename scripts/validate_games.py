#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.validation import iter_validation_events, resolve_inputs


def _fmt_rate(n: int, elapsed_s: float) -> str:
    if elapsed_s <= 0:
        return "0.0/s"
    return f"{(n / elapsed_s):.1f}/s"


class _ProgressLine:
    def __init__(self, enabled: bool, total: int = 0, unit: str = "games") -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.total = total
        self.unit = unit
        self.start = time.monotonic()
        self.last_draw = 0.0
        self.tick = 0
        self.active = False

    def _bar(self, frac: float, width: int) -> str:
        filled = max(0, min(width, int(width * frac)))
        return "#" * filled + "-" * (width - filled)

    def _pulse(self, width: int) -> str:
        width = max(8, width)
        pos = self.tick % width
        chars = ["-"] * width
        chars[pos] = "#"
        if pos > 0:
            chars[pos - 1] = "="
        return "".join(chars)

    def update(self, processed: int, valid: int, invalid: int, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self.last_draw) < 0.15:
            return
        self.last_draw = now
        self.tick += 1
        self.active = True
        elapsed = now - self.start
        ratio = (valid / processed) if processed else 0.0

        term_w = shutil.get_terminal_size((100, 20)).columns
        if self.total > 0:
            frac = processed / max(self.total, 1)
            bar_w = max(10, min(36, term_w - 70))
            bar = self._bar(frac, bar_w)
            msg = (
                f"\r[validate] [{bar}] {processed}/{self.total} {self.unit} "
                f"valid={valid} invalid={invalid} rate={_fmt_rate(processed, elapsed)}"
            )
        else:
            bar_w = max(10, min(24, term_w - 78))
            bar = self._pulse(bar_w)
            msg = (
                f"\r[validate] [{bar}] processed={processed} valid={valid} invalid={invalid} "
                f"valid_ratio={ratio:.4f} rate={_fmt_rate(processed, elapsed)}"
            )
        sys.stderr.write(msg)
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled and self.active:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.active = False


def _validate_file_to_shards(path: str, min_plies: int, shard_dir: str, shard_idx: int) -> Dict:
    valid_path = os.path.join(shard_dir, f"valid_{shard_idx:04d}.jsonl")
    invalid_path = os.path.join(shard_dir, f"invalid_{shard_idx:04d}.csv")
    reason_counts: Dict[str, int] = {}
    valid_count = 0
    invalid_count = 0

    with open(valid_path, "w", encoding="utf-8") as valid_f, open(
        invalid_path, "w", encoding="utf-8", newline=""
    ) as invalid_f:
        invalid_writer = csv.DictWriter(
            invalid_f,
            fieldnames=["game_id", "source_file", "reason", "offending_ply", "result"],
        )
        invalid_writer.writeheader()

        for kind, row in iter_validation_events(paths=[path], min_plies=min_plies):
            if kind == "valid":
                valid_f.write(json.dumps(row, ensure_ascii=True) + "\n")
                valid_count += 1
            else:
                invalid_writer.writerow(row)
                invalid_count += 1
                reason = row.get("reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "path": path,
        "valid_path": valid_path,
        "invalid_path": invalid_path,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "reason_counts": reason_counts,
    }


def _merge_shards(results: List[Dict], valid_out: str, invalid_out: str) -> Tuple[int, int, Dict[str, int]]:
    ensure_parent(valid_out)
    ensure_parent(invalid_out)
    total_valid = 0
    total_invalid = 0
    merged_reasons: Dict[str, int] = {}

    with open(valid_out, "w", encoding="utf-8") as valid_f, open(
        invalid_out, "w", encoding="utf-8", newline=""
    ) as invalid_f:
        invalid_writer = csv.DictWriter(
            invalid_f,
            fieldnames=["game_id", "source_file", "reason", "offending_ply", "result"],
        )
        invalid_writer.writeheader()

        for res in sorted(results, key=lambda x: x["path"]):
            with open(res["valid_path"], "r", encoding="utf-8") as vf:
                shutil.copyfileobj(vf, valid_f)
            with open(res["invalid_path"], "r", encoding="utf-8", newline="") as inf:
                reader = csv.DictReader(inf)
                for row in reader:
                    invalid_writer.writerow(row)

            total_valid += int(res["valid_count"])
            total_invalid += int(res["invalid_count"])
            for reason, count in res["reason_counts"].items():
                merged_reasons[reason] = merged_reasons.get(reason, 0) + int(count)

    return total_valid, total_invalid, merged_reasons


def _split_single_pgn_for_parallel(input_path: str, out_dir: str, num_shards: int) -> Tuple[List[str], int]:
    """
    Split a PGN file into round-robin shard PGNs by game start markers.

    This targets Lichess-style PGNs where each game begins with an [Event "..."]
    header line and preserves game text verbatim.
    """
    if num_shards < 2:
        return [input_path], 0

    out_paths = [os.path.join(out_dir, f"chunk_{i + 1:02d}.pgn") for i in range(num_shards)]
    current_idx: Optional[int] = None
    game_count = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as src:
        files = [open(p, "w", encoding="utf-8") for p in out_paths]
        try:
            for line in src:
                if line.startswith("[Event "):
                    current_idx = game_count % num_shards
                    game_count += 1
                if current_idx is not None:
                    files[current_idx].write(line)
        finally:
            for f in files:
                f.close()

    shard_paths = [p for p in out_paths if os.path.getsize(p) > 0]
    return shard_paths, game_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PGN games by legal replay.")
    parser.add_argument("--input", required=True, help="PGN file, glob, or directory")
    parser.add_argument("--valid-out", default="data/validated/valid_games.jsonl")
    parser.add_argument("--invalid-out", default="data/validated/invalid_games.csv")
    parser.add_argument("--summary-out", default="data/validated/summary.json")
    parser.add_argument("--min-plies", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Worker process count for validating multiple input PGN files concurrently "
            "(0 uses all CPU cores; single-file inputs remain sequential)."
        ),
    )
    parser.add_argument(
        "--all-cores",
        action="store_true",
        help="Convenience alias for --workers 0 (use all available CPU cores).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N processed games (0 disables periodic progress logs).",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a live terminal progress bar/status line (TTY only).",
    )
    parser.add_argument(
        "--auto-shard-single-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a single PGN file is provided and workers>1, split it into temporary PGN shards "
            "so file-level multiprocessing can use all cores."
        ),
    )
    args = parser.parse_args()
    if args.all_cores:
        args.workers = 0

    paths = resolve_inputs(args.input)
    if not paths:
        raise SystemExit("No PGN inputs found")

    cpu_count = os.cpu_count() or 1
    worker_count = cpu_count if args.workers <= 0 else args.workers
    input_paths_for_run = list(paths)
    auto_sharded = False
    auto_shard_game_count: Optional[int] = None

    with tempfile.TemporaryDirectory(prefix="validate_runtime_", dir="/tmp") as runtime_tmp:
        if (
            args.auto_shard_single_file
            and worker_count > 1
            and len(paths) == 1
            and os.path.isfile(paths[0])
            and paths[0].lower().endswith(".pgn")
        ):
            chunk_dir = os.path.join(runtime_tmp, "input_chunks")
            os.makedirs(chunk_dir, exist_ok=True)
            input_paths_for_run, auto_shard_game_count = _split_single_pgn_for_parallel(
                input_path=paths[0], out_dir=chunk_dir, num_shards=worker_count
            )
            auto_sharded = len(input_paths_for_run) > 1
            if auto_sharded:
                suffix = (
                    f" (detected_games={auto_shard_game_count})"
                    if auto_shard_game_count is not None
                    else ""
                )
                print(
                    f"[validate] auto-sharded single PGN into {len(input_paths_for_run)} chunks "
                    f"for {worker_count} workers{suffix}"
                )

        use_parallel_files = worker_count > 1 and len(input_paths_for_run) > 1
        if worker_count > 1 and len(input_paths_for_run) <= 1 and not auto_sharded:
            print(
                "[validate] note: single input PGN is validated as one streaming parse; "
                "file-level parallel workers require multiple PGN files."
            )

        if use_parallel_files:
            file_progress = _ProgressLine(enabled=args.progress_bar, total=len(input_paths_for_run), unit="files")
            shard_dir = os.path.join(runtime_tmp, "output_shards")
            os.makedirs(shard_dir, exist_ok=True)
            results: List[Dict] = []
            with ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = [
                    ex.submit(_validate_file_to_shards, path, args.min_plies, shard_dir, i)
                    for i, path in enumerate(input_paths_for_run)
                ]
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    file_progress.update(
                        processed=len(results),
                        valid=sum(int(r["valid_count"]) for r in results),
                        invalid=sum(int(r["invalid_count"]) for r in results),
                    )
                    if args.progress_every >= 0:
                        print(
                            f"[validate] finished file: {res['path']} "
                            f"(valid={res['valid_count']}, invalid={res['invalid_count']}) "
                            f"[{len(results)}/{len(input_paths_for_run)} files]"
                        )
            valid_count, invalid_count, reason_counts = _merge_shards(
                results=results, valid_out=args.valid_out, invalid_out=args.invalid_out
            )
            file_progress.close()
        else:
            ensure_parent(args.valid_out)
            ensure_parent(args.invalid_out)
            reason_counts = {}
            valid_count = 0
            invalid_count = 0

            with open(args.valid_out, "w", encoding="utf-8") as valid_f, open(
                args.invalid_out, "w", encoding="utf-8", newline=""
            ) as invalid_f:
                invalid_writer = csv.DictWriter(
                    invalid_f,
                    fieldnames=["game_id", "source_file", "reason", "offending_ply", "result"],
                )
                invalid_writer.writeheader()

                processed = 0
                progress = _ProgressLine(enabled=args.progress_bar)
                for kind, row in iter_validation_events(paths=input_paths_for_run, min_plies=args.min_plies):
                    processed += 1
                    if kind == "valid":
                        valid_f.write(json.dumps(row, ensure_ascii=True) + "\n")
                        valid_count += 1
                    else:
                        invalid_writer.writerow(row)
                        invalid_count += 1
                        reason = row.get("reason", "unknown")
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    progress.update(processed=processed, valid=valid_count, invalid=invalid_count)
                    if args.progress_every > 0 and processed % args.progress_every == 0:
                        print(
                            f"[validate] processed={processed} valid={valid_count} invalid={invalid_count} "
                            f"valid_ratio={(valid_count / processed):.4f}"
                        )
                progress.update(processed=processed, valid=valid_count, invalid=invalid_count, force=True)
                progress.close()

    total = valid_count + invalid_count
    summary = {
        "input_files": input_paths_for_run,
        "input_source": args.input,
        "auto_sharded_single_file": auto_sharded,
        "auto_shard_game_count": auto_shard_game_count if auto_sharded else None,
        "total_games": total,
        "valid_games": valid_count,
        "invalid_games": invalid_count,
        "valid_ratio": (valid_count / total) if total else 0.0,
        "invalid_reason_counts": reason_counts,
        "outputs": {
            "valid_games": args.valid_out,
            "invalid_games": args.invalid_out,
        },
    }
    write_json(args.summary_out, summary)

    print(f"Valid games: {valid_count}")
    print(f"Invalid games: {invalid_count}")
    if use_parallel_files:
        print(f"Workers: {worker_count} (processes, file-level)")
    elif worker_count > 1:
        print(f"Workers requested: {worker_count} (single-file input ran sequentially)")
    print(f"Summary: {args.summary_out}")


if __name__ == "__main__":
    main()
