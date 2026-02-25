#!/usr/bin/env python3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Iterator, List, Tuple

from src.chessbot.io_utils import ensure_parent, read_jsonl, write_json
from src.chessbot.splicing import (
    SpliceConfig,
    game_has_splice_samples,
    is_game_eligible,
    iter_game_splice_samples,
    split_game_ids,
)


def _iter_batches(rows: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _process_game_to_lines(game: Dict, cfg: SpliceConfig, split: Dict[str, set]) -> Tuple[str, str, List[str], bool]:
    if not is_game_eligible(game, cfg):
        return "", "", [], False
    gid = game["game_id"]
    if gid in split["train_games"]:
        split_name = "train"
    elif gid in split["val_games"]:
        split_name = "val"
    elif gid in split["test_games"]:
        split_name = "test"
    else:
        return "", "", [], False

    lines: List[str] = []
    for sample in iter_game_splice_samples(game, cfg):
        lines.append(json.dumps(sample, ensure_ascii=True) + "\n")
    return split_name, gid, lines, bool(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build splice dataset from validated games")
    parser.add_argument("--input", required=True, help="Path to valid_games.jsonl")
    parser.add_argument("--output-dir", default="data/dataset")
    parser.add_argument("--k", type=int, default=4, help="Target horizon")
    parser.add_argument("--min-context", type=int, default=8)
    parser.add_argument("--min-target", type=int, default=1)
    parser.add_argument("--max-samples-per-game", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--decisive-only", action="store_true", default=True)
    parser.add_argument("--allow-draws", action="store_true", help="Include draw games")
    parser.add_argument("--workers", type=int, default=1, help="Threads for batch sample generation in pass 2")
    parser.add_argument("--batch-size", type=int, default=256, help="Games per pass-2 processing batch")
    args = parser.parse_args()

    decisive_only = args.decisive_only and not args.allow_draws
    cfg = SpliceConfig(
        k=args.k,
        min_context=args.min_context,
        min_target=args.min_target,
        max_samples_per_game=args.max_samples_per_game,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        decisive_only=decisive_only,
    )

    # Pass 1: collect split-eligible game ids only (streaming read).
    input_games_total = 0
    filtered_games_total = 0
    spliceable_game_ids = []
    for game in read_jsonl(args.input):
        input_games_total += 1
        if not is_game_eligible(game, cfg):
            continue
        filtered_games_total += 1
        if game_has_splice_samples(game, cfg):
            spliceable_game_ids.append(game["game_id"])

    split = split_game_ids(spliceable_game_ids, cfg)

    train_path = f"{args.output_dir}/train.jsonl"
    val_path = f"{args.output_dir}/val.jsonl"
    test_path = f"{args.output_dir}/test.jsonl"
    stats_path = f"{args.output_dir}/stats.json"

    ensure_parent(train_path)
    ensure_parent(val_path)
    ensure_parent(test_path)

    split_sample_counts = {"train": 0, "val": 0, "test": 0}
    routed_games_seen = {"train": set(), "val": set(), "test": set()}

    with open(train_path, "w", encoding="utf-8") as train_f, open(
        val_path, "w", encoding="utf-8"
    ) as val_f, open(test_path, "w", encoding="utf-8") as test_f:
        if args.workers <= 1:
            for game in read_jsonl(args.input):
                split_name, gid, lines, wrote_any = _process_game_to_lines(game, cfg, split)
                if not split_name:
                    continue
                out_f = train_f if split_name == "train" else val_f if split_name == "val" else test_f
                for line in lines:
                    out_f.write(line)
                split_sample_counts[split_name] += len(lines)
                if wrote_any:
                    routed_games_seen[split_name].add(gid)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for batch in _iter_batches(read_jsonl(args.input), args.batch_size):
                    for split_name, gid, lines, wrote_any in ex.map(
                        lambda g: _process_game_to_lines(g, cfg, split), batch
                    ):
                        if not split_name:
                            continue
                        out_f = train_f if split_name == "train" else val_f if split_name == "val" else test_f
                        for line in lines:
                            out_f.write(line)
                        split_sample_counts[split_name] += len(lines)
                        if wrote_any:
                            routed_games_seen[split_name].add(gid)

    stats = {
        "input_games_total": input_games_total,
        "input_games_after_filters": filtered_games_total,
        "spliceable_games": len(spliceable_game_ids),
        "split_games": {
            "train": len(split["train_games"]),
            "val": len(split["val_games"]),
            "test": len(split["test_games"]),
        },
        "split_games_with_samples_written": {
            "train": len(routed_games_seen["train"]),
            "val": len(routed_games_seen["val"]),
            "test": len(routed_games_seen["test"]),
        },
        "split_samples": {
            "train": split_sample_counts["train"],
            "val": split_sample_counts["val"],
            "test": split_sample_counts["test"],
        },
        "params": {
            "k": args.k,
            "min_context": args.min_context,
            "min_target": args.min_target,
            "max_samples_per_game": args.max_samples_per_game,
            "decisive_only": decisive_only,
            "workers": args.workers,
            "batch_size": args.batch_size,
        },
        "outputs": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        },
    }
    write_json(stats_path, stats)

    print(f"Input games total/after_filters/spliceable: {input_games_total}/{filtered_games_total}/{len(spliceable_game_ids)}")
    print(
        "Samples train/val/test: "
        f"{split_sample_counts['train']}/{split_sample_counts['val']}/{split_sample_counts['test']}"
    )
    if args.workers > 1:
        print(f"Workers: {args.workers} (threaded pass-2 batches of {args.batch_size})")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
