import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Set, Tuple

import chess

from src.chessbot.phase import (
    PHASE_RULE_VERSION,
    classify_board_phase,
    remaining_plies_bucket,
    relative_progress_bucket,
)

@dataclass
class SpliceConfig:
    k: int = 4
    min_context: int = 8
    min_target: int = 1
    max_samples_per_game: int = 0
    seed: int = 7
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    decisive_only: bool = True


def compute_split_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def is_game_eligible(game: Dict, cfg: SpliceConfig) -> bool:
    if cfg.decisive_only and game.get("winner_side") not in {"W", "B"}:
        return False
    return True


def game_has_splice_samples(game: Dict, cfg: SpliceConfig) -> bool:
    moves = game.get("moves_uci", [])
    n = len(moves)
    start_i = cfg.min_context - 1
    end_i = n - cfg.min_target - 1
    return end_i >= start_i


def _stable_game_seed(game_id: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{game_id}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def iter_game_splice_samples(game: Dict, cfg: SpliceConfig) -> Iterator[Dict]:
    game_id = game["game_id"]
    winner_side = game.get("winner_side", "?")
    moves = game.get("moves_uci", [])
    n = len(moves)

    start_i = cfg.min_context - 1
    end_i = n - cfg.min_target - 1
    if end_i < start_i:
        return

    local_samples: List[Dict] = []
    board = chess.Board()
    board_ok = True
    for i, uci in enumerate(moves):
        if board_ok:
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                board_ok = False
            else:
                if mv in board.legal_moves:
                    board.push(mv)
                else:
                    board_ok = False
        if i < start_i or i > end_i:
            continue
        target = moves[i + 1 : i + 1 + cfg.k]
        if len(target) < cfg.min_target:
            continue
        ply = i + 1
        plies_remaining = max(0, n - ply)
        phase_info = (
            classify_board_phase(board, ply=ply)
            if board_ok
            else {
                "phase": "unknown",
                "phase_reason": "invalid_moves_uci",
                "phase_rule_version": PHASE_RULE_VERSION,
            }
        )
        local_samples.append(
            {
                "game_id": game_id,
                "winner_side": winner_side,
                "splice_index": i,
                "ply": ply,
                "plies_remaining": plies_remaining,
                "plies_remaining_bucket": remaining_plies_bucket(plies_remaining),
                "relative_progress_bucket": relative_progress_bucket(ply, n),
                "phase": phase_info.get("phase", "unknown"),
                "phase_reason": phase_info.get("phase_reason", "unknown"),
                "phase_rule_version": phase_info.get("phase_rule_version", PHASE_RULE_VERSION),
                "context": moves[: i + 1],
                "target": target,
                "next_move": target[0],
                "plies_total": n,
            }
        )

    if cfg.max_samples_per_game > 0 and len(local_samples) > cfg.max_samples_per_game:
        rnd = random.Random(_stable_game_seed(game_id, cfg.seed))
        rnd.shuffle(local_samples)
        local_samples = local_samples[: cfg.max_samples_per_game]

    for sample in local_samples:
        yield sample


def split_game_ids(game_ids: List[str], cfg: SpliceConfig) -> Dict[str, Set[str]]:
    # Deduplicate while preserving first-seen order to avoid false leakage if IDs repeat.
    ordered = list(dict.fromkeys(game_ids))
    rnd = random.Random(cfg.seed)
    rnd.shuffle(ordered)

    n_train, n_val, _ = compute_split_counts(len(ordered), cfg.train_ratio, cfg.val_ratio)
    train_games = set(ordered[:n_train])
    val_games = set(ordered[n_train : n_train + n_val])
    test_games = set(ordered[n_train + n_val :])

    overlap = (train_games & val_games) or (train_games & test_games) or (val_games & test_games)
    if overlap:
        raise RuntimeError("Split leakage detected in game IDs")

    return {
        "train_games": train_games,
        "val_games": val_games,
        "test_games": test_games,
    }


def build_splice_samples(games: List[Dict], cfg: SpliceConfig) -> Dict[str, List[Dict]]:
    # Legacy in-memory helper kept for compatibility/tests; prefer iter_game_splice_samples + split_game_ids.
    samples_by_game: Dict[str, List[Dict]] = defaultdict(list)

    for game in games:
        if not is_game_eligible(game, cfg):
            continue
        game_id = game["game_id"]
        samples_by_game[game_id].extend(iter_game_splice_samples(game, cfg))

    return samples_by_game


def split_by_game(samples_by_game: Dict[str, List[Dict]], cfg: SpliceConfig):
    game_ids = list(samples_by_game.keys())
    rnd = random.Random(cfg.seed)
    rnd.shuffle(game_ids)

    n_train, n_val, _ = compute_split_counts(len(game_ids), cfg.train_ratio, cfg.val_ratio)
    train_games = set(game_ids[:n_train])
    val_games = set(game_ids[n_train : n_train + n_val])
    test_games = set(game_ids[n_train + n_val :])

    train_rows: List[Dict] = []
    val_rows: List[Dict] = []
    test_rows: List[Dict] = []

    for gid, rows in samples_by_game.items():
        if gid in train_games:
            train_rows.extend(rows)
        elif gid in val_games:
            val_rows.extend(rows)
        else:
            test_rows.extend(rows)

    overlap = (train_games & val_games) or (train_games & test_games) or (val_games & test_games)
    if overlap:
        raise RuntimeError("Split leakage detected in game IDs")

    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_games": train_games,
        "val_games": val_games,
        "test_games": test_games,
    }
