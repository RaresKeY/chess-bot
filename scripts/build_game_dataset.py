#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.io_utils import ensure_parent, read_jsonl, write_json
from src.chessbot.splicing import SpliceConfig, game_has_splice_samples, is_game_eligible, split_game_ids


def _canonical_game_row(game: Dict, keep_headers: bool = False) -> Dict:
    moves = list(game.get("moves") or game.get("moves_uci") or [])
    row = {
        "schema": "game_dataset_runtime_splice_v1",
        "game_id": game.get("game_id", ""),
        "winner_side": game.get("winner_side", "?"),
        "result": game.get("result", "*"),
        "plies": int(game.get("plies", len(moves))),
        "moves": moves,
    }
    source_file = game.get("source_file")
    if source_file:
        row["source_file"] = source_file
    if keep_headers and isinstance(game.get("headers"), dict):
        row["headers"] = game["headers"]
    return row


def _iter_games(input_path: str) -> Iterable[Dict]:
    for row in read_jsonl(input_path):
        if isinstance(row, dict):
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact game-level train/val/test dataset from validated games (runtime-splicing architecture)."
    )
    parser.add_argument("--input", required=True, help="Path to validated valid_games.jsonl")
    parser.add_argument("--output-dir", default="data/dataset", help="Output directory for train/val/test JSONL files")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--decisive-only", action="store_true", default=True)
    parser.add_argument("--allow-draws", action="store_true", help="Include draw games")
    parser.add_argument(
        "--runtime-min-context",
        type=int,
        default=8,
        help="Eligibility floor used to exclude games with no runtime splice samples.",
    )
    parser.add_argument(
        "--runtime-min-target",
        type=int,
        default=1,
        help="Eligibility floor used to exclude games with no runtime splice samples.",
    )
    parser.add_argument(
        "--keep-headers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preserve PGN headers in output rows (larger files; training does not require this).",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    decisive_only = bool(args.decisive_only and not args.allow_draws)
    cfg = SpliceConfig(
        k=4,  # unused for game-level writing; only runtime splice eligibility matters below
        min_context=int(args.runtime_min_context),
        min_target=int(args.runtime_min_target),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        decisive_only=decisive_only,
    )

    # Pass 1: determine split-eligible game ids only.
    input_games_total = 0
    eligible_games_total = 0
    spliceable_game_ids: List[str] = []
    for game in _iter_games(args.input):
        input_games_total += 1
        if not is_game_eligible(game, cfg):
            continue
        eligible_games_total += 1
        if game_has_splice_samples(game, cfg):
            gid = str(game.get("game_id", ""))
            if gid:
                spliceable_game_ids.append(gid)
        if args.progress_every > 0 and input_games_total % args.progress_every == 0:
            print(
                f"[game-dataset pass1] processed={input_games_total} eligible={eligible_games_total} "
                f"spliceable={len(spliceable_game_ids)}",
                flush=True,
            )

    split = split_game_ids(spliceable_game_ids, cfg)

    out_dir = Path(args.output_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    stats_path = out_dir / "stats.json"
    ensure_parent(str(train_path))
    ensure_parent(str(val_path))
    ensure_parent(str(test_path))

    split_game_counts = {"train": 0, "val": 0, "test": 0}
    split_plies_total = {"train": 0, "val": 0, "test": 0}
    split_winner_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    with open(train_path, "w", encoding="utf-8") as train_f, open(
        val_path, "w", encoding="utf-8"
    ) as val_f, open(test_path, "w", encoding="utf-8") as test_f:
        pass2_games = 0
        for game in _iter_games(args.input):
            pass2_games += 1
            if not is_game_eligible(game, cfg) or not game_has_splice_samples(game, cfg):
                if args.progress_every > 0 and pass2_games % args.progress_every == 0:
                    print(
                        f"[game-dataset pass2] processed={pass2_games} "
                        f"games(train/val/test)={split_game_counts['train']}/{split_game_counts['val']}/{split_game_counts['test']}",
                        flush=True,
                    )
                continue
            gid = str(game.get("game_id", ""))
            if gid in split["train_games"]:
                split_name = "train"
                out_f = train_f
            elif gid in split["val_games"]:
                split_name = "val"
                out_f = val_f
            elif gid in split["test_games"]:
                split_name = "test"
                out_f = test_f
            else:
                continue

            row = _canonical_game_row(game, keep_headers=bool(args.keep_headers))
            out_f.write(json.dumps(row, ensure_ascii=True) + "\n")
            split_game_counts[split_name] += 1
            split_plies_total[split_name] += int(row.get("plies", 0))
            split_winner_counts[split_name][str(row.get("winner_side", "?"))] += 1

            if args.progress_every > 0 and pass2_games % args.progress_every == 0:
                print(
                    f"[game-dataset pass2] processed={pass2_games} "
                    f"games(train/val/test)={split_game_counts['train']}/{split_game_counts['val']}/{split_game_counts['test']}",
                    flush=True,
                )

    stats = {
        "dataset_format": "game_jsonl_runtime_splice_v1",
        "input_path": str(Path(args.input)),
        "output_dir": str(out_dir.resolve()),
        "schema_move_field": "moves",
        "source_validated_move_alias_supported": ["moves", "moves_uci"],
        "keep_headers": bool(args.keep_headers),
        "runtime_splice_defaults": {
            "min_context": int(args.runtime_min_context),
            "min_target": int(args.runtime_min_target),
        },
        "split_seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "decisive_only": decisive_only,
        "input_games_total": int(input_games_total),
        "input_games_after_filters": int(eligible_games_total),
        "spliceable_games": int(len(spliceable_game_ids)),
        "split_games": {k: int(v) for k, v in split_game_counts.items()},
        "split_plies_total": {k: int(v) for k, v in split_plies_total.items()},
        "split_avg_plies": {
            k: (float(split_plies_total[k]) / split_game_counts[k] if split_game_counts[k] else 0.0)
            for k in split_game_counts
        },
        "split_winner_counts": {k: dict(v) for k, v in split_winner_counts.items()},
    }
    write_json(str(stats_path), stats)
    print(json.dumps({"game_dataset_built": stats}, ensure_ascii=True))


if __name__ == "__main__":
    main()

