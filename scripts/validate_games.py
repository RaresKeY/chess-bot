#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.validation import iter_validation_events, resolve_inputs


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
        default=1,
        help="Thread count for validating multiple input PGN files concurrently (single-file inputs remain sequential).",
    )
    args = parser.parse_args()

    paths = resolve_inputs(args.input)
    if not paths:
        raise SystemExit("No PGN inputs found")

    use_threaded = args.workers > 1 and len(paths) > 1
    if use_threaded:
        with tempfile.TemporaryDirectory(prefix="validate_shards_", dir="/tmp") as shard_dir:
            results: List[Dict] = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(_validate_file_to_shards, path, args.min_plies, shard_dir, i)
                    for i, path in enumerate(paths)
                ]
                for fut in as_completed(futures):
                    results.append(fut.result())
            valid_count, invalid_count, reason_counts = _merge_shards(
                results=results, valid_out=args.valid_out, invalid_out=args.invalid_out
            )
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

            for kind, row in iter_validation_events(paths=paths, min_plies=args.min_plies):
                if kind == "valid":
                    valid_f.write(json.dumps(row, ensure_ascii=True) + "\n")
                    valid_count += 1
                else:
                    invalid_writer.writerow(row)
                    invalid_count += 1
                    reason = row.get("reason", "unknown")
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

    total = valid_count + invalid_count
    summary = {
        "input_files": paths,
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
    if use_threaded:
        print(f"Workers: {args.workers} (threaded, file-level)")
    print(f"Summary: {args.summary_out}")


if __name__ == "__main__":
    main()
