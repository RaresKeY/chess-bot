#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.pgn
import torch

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.inference import (
    artifact_rollout_horizon,
    artifact_training_objective,
    infer_first_move_auto_from_artifact_on_device,
    infer_first_move_dual_artifacts_on_device,
)
from src.chessbot.io_utils import ensure_parent, write_json


def _find_latest_model(artifacts_dir: Path) -> Path:
    candidates = [p for p in artifacts_dir.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise SystemExit(f"No model artifacts found under {artifacts_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _resolve_model_path(text: str) -> Path:
    if text == "latest":
        return _find_latest_model(REPO_ROOT / "artifacts")
    return Path(text).resolve()


def _resolve_optional_model_path(text: str) -> Optional[Path]:
    value = str(text or "").strip()
    if not value:
        return None
    return _resolve_model_path(value)


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _parse_topk_multipliers(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            v = int(piece)
        except Exception:
            continue
        if v > 0 and v not in out:
            out.append(v)
    return out or [1, 2, 5]


class LoadedMoveModelRuntime:
    def __init__(
        self,
        *,
        alias: str,
        device_str: str,
        model_path: Optional[Path],
        model_white_path: Optional[Path],
        model_black_path: Optional[Path],
        sequence_decode_policy: str,
        fallback_topk_multipliers: List[int],
    ):
        self.alias = alias
        self.device = torch.device(device_str)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

        self.model_path = model_path
        self.model_white_path = model_white_path
        self.model_black_path = model_black_path
        self.sequence_decode_policy = str(sequence_decode_policy)
        self.fallback_topk_multipliers = [int(x) for x in fallback_topk_multipliers]

        if model_white_path is not None and model_black_path is not None:
            self.mode = "dual_pair"
            self.white_artifact = torch.load(str(model_white_path), map_location="cpu")
            self.black_artifact = torch.load(str(model_black_path), map_location="cpu")
            self.single_artifact = None
            self.training_objective = "dual_pair_side_routed"
            self.artifact_rollout_horizon = 0
            self.policy_mode_default = "sequence"
            return

        if model_path is not None:
            self.mode = "single"
            self.single_artifact = torch.load(str(model_path), map_location="cpu")
            self.white_artifact = None
            self.black_artifact = None
            self.training_objective = artifact_training_objective(self.single_artifact)
            self.artifact_rollout_horizon = artifact_rollout_horizon(self.single_artifact)
            self.policy_mode_default = "rollout" if self.training_objective.startswith("multistep_") else "next"
            return

        raise RuntimeError(
            f"Participant {alias} must provide --model or both --model-<x>-white and --model-<x>-black"
        )

    def infer(self, context: List[str], winner_side: str, topk: int) -> Dict:
        if self.mode == "dual_pair":
            out = infer_first_move_dual_artifacts_on_device(
                white_artifact=self.white_artifact,
                black_artifact=self.black_artifact,
                context=context,
                topk=topk,
                device_str=str(self.device),
                sequence_decode_policy=self.sequence_decode_policy,
                fallback_topk_multipliers=self.fallback_topk_multipliers,
            )
            topk_tokens = list(out.get("topk_step1") or [])
            return {
                "topk": topk_tokens,
                "best_legal": str(out.get("move_uci") or out.get("best_legal") or ""),
                "predicted_uci": str(topk_tokens[0] if topk_tokens else ""),
                "fallback": bool(out.get("fallback", False)),
                "policy_mode_used": str(out.get("policy_mode_used") or "sequence"),
                "device": str(self.device),
            }

        out = infer_first_move_auto_from_artifact_on_device(
            artifact=self.single_artifact,
            context=context,
            winner_side=winner_side,
            topk=topk,
            device_str=str(self.device),
            policy_mode="auto",
            rollout_plies=0,
            rollout_fallback_legal=True,
            sequence_decode_policy=self.sequence_decode_policy,
            fallback_topk_multipliers=self.fallback_topk_multipliers,
        )
        topk_tokens = list(out.get("topk") or out.get("topk_step1") or [])
        if not topk_tokens:
            step_debug = list(out.get("step_debug") or [])
            if step_debug:
                topk_tokens = list(step_debug[0].get("topk") or [])
        predicted_uci = str(out.get("predicted_uci") or (topk_tokens[0] if topk_tokens else ""))
        return {
            "topk": topk_tokens,
            "best_legal": str(out.get("move_uci") or out.get("best_legal") or out.get("first_move") or ""),
            "predicted_uci": predicted_uci,
            "fallback": bool(out.get("fallback", False)),
            "policy_mode_used": str(out.get("policy_mode_used") or self.policy_mode_default),
            "device": str(self.device),
        }


def _model_move(
    runtime: LoadedMoveModelRuntime,
    context: List[str],
    winner_side: str,
    topk: int,
    board: chess.Board,
) -> Dict:
    out = runtime.infer(context=context, winner_side=winner_side, topk=topk)
    topk_tokens = list(out.get("topk") or [])
    predicted_uci = str(out.get("predicted_uci") or (topk_tokens[0] if topk_tokens else ""))
    uci = str(out.get("best_legal") or "")
    fallback_flag = bool(out.get("fallback", False))
    if uci:
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            mv = None
        if mv is not None and mv in board.legal_moves:
            return {
                "move_uci": uci,
                "fallback": fallback_flag,
                "predicted_uci": predicted_uci,
                "topk": topk_tokens,
            }

    fallback = next(iter(board.legal_moves), None)
    if fallback is None:
        return {
            "move_uci": "",
            "fallback": True,
            "predicted_uci": predicted_uci,
            "topk": topk_tokens,
        }
    return {
        "move_uci": fallback.uci(),
        "fallback": True,
        "predicted_uci": predicted_uci,
        "topk": topk_tokens,
    }


def _result_for_player(board: chess.Board, player_color: chess.Color) -> str:
    result = board.result(claim_draw=True)
    if result == "1-0":
        return "win" if player_color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if player_color == chess.BLACK else "loss"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def _render_progress(i: int, total: int, wins: int, draws: int, losses: int) -> None:
    width = 28
    frac = min(1.0, max(0.0, i / max(total, 1)))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[selfplay] [{bar}] {i}/{total} A W/D/L={wins}/{draws}/{losses}")
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play head-to-head matches between two trained model artifacts")
    parser.add_argument("--model-a", default="", help="Model A artifact path or 'latest'")
    parser.add_argument("--model-b", default="", help="Model B artifact path or 'latest'")
    parser.add_argument("--model-a-white", default="", help="Optional white-side artifact path for model A")
    parser.add_argument("--model-a-black", default="", help="Optional black-side artifact path for model A")
    parser.add_argument("--model-b-white", default="", help="Optional white-side artifact path for model B")
    parser.add_argument("--model-b-black", default="", help="Optional black-side artifact path for model B")
    parser.add_argument("--alias-a", default="model_a", help="Display name for model A")
    parser.add_argument("--alias-b", default="model_b", help="Display name for model B")
    parser.add_argument("--games", type=int, default=20, help="Number of games")
    parser.add_argument("--topk-a", type=int, default=10, help="Model A top-k candidates")
    parser.add_argument("--topk-b", type=int, default=10, help="Model B top-k candidates")
    parser.add_argument("--winner-side-a", default="W", choices=["W", "B", "D", "?"], help="Conditioning token for model A")
    parser.add_argument("--winner-side-b", default="W", choices=["W", "B", "D", "?"], help="Conditioning token for model B")
    parser.add_argument(
        "--sequence-decode-policy-a",
        choices=["sequence_path", "step1_legal"],
        default="sequence_path",
        help="Dual-sequence first-move decode policy for model A",
    )
    parser.add_argument(
        "--sequence-decode-policy-b",
        choices=["sequence_path", "step1_legal"],
        default="sequence_path",
        help="Dual-sequence first-move decode policy for model B",
    )
    parser.add_argument(
        "--fallback-topk-multipliers-a",
        default="1,2,5",
        help="Comma-separated top-k multipliers for model A progressive decode retries",
    )
    parser.add_argument(
        "--fallback-topk-multipliers-b",
        default="1,2,5",
        help="Comma-separated top-k multipliers for model B progressive decode retries",
    )
    parser.add_argument("--device-a", default="auto", help="Torch device for model A (cpu/cuda/auto)")
    parser.add_argument("--device-b", default="auto", help="Torch device for model B (cpu/cuda/auto)")
    parser.add_argument("--max-plies", type=int, default=300, help="Hard cap on plies per game")
    parser.add_argument("--alternate-colors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-out", default="", help="Optional JSON summary output path")
    parser.add_argument("--pgn-out", default="", help="Optional PGN output path for all games")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_a_path = _resolve_optional_model_path(args.model_a)
    model_b_path = _resolve_optional_model_path(args.model_b)
    model_a_white_path = _resolve_optional_model_path(args.model_a_white)
    model_a_black_path = _resolve_optional_model_path(args.model_a_black)
    model_b_white_path = _resolve_optional_model_path(args.model_b_white)
    model_b_black_path = _resolve_optional_model_path(args.model_b_black)

    device_a = _resolve_device(args.device_a)
    device_b = _resolve_device(args.device_b)
    fallback_topk_multipliers_a = _parse_topk_multipliers(args.fallback_topk_multipliers_a)
    fallback_topk_multipliers_b = _parse_topk_multipliers(args.fallback_topk_multipliers_b)

    if args.verbose:
        print(
            {
                "match_start": {
                    "model_a": str(model_a_path or ""),
                    "model_b": str(model_b_path or ""),
                    "model_a_white": str(model_a_white_path or ""),
                    "model_a_black": str(model_a_black_path or ""),
                    "model_b_white": str(model_b_white_path or ""),
                    "model_b_black": str(model_b_black_path or ""),
                    "alias_a": args.alias_a,
                    "alias_b": args.alias_b,
                    "games": args.games,
                    "topk_a": args.topk_a,
                    "topk_b": args.topk_b,
                    "winner_side_a": args.winner_side_a,
                    "winner_side_b": args.winner_side_b,
                    "sequence_decode_policy_a": args.sequence_decode_policy_a,
                    "sequence_decode_policy_b": args.sequence_decode_policy_b,
                    "fallback_topk_multipliers_a": fallback_topk_multipliers_a,
                    "fallback_topk_multipliers_b": fallback_topk_multipliers_b,
                    "device_a": device_a,
                    "device_b": device_b,
                    "max_plies": args.max_plies,
                    "alternate_colors": bool(args.alternate_colors),
                }
            }
        )

    runtime_a = LoadedMoveModelRuntime(
        alias=args.alias_a,
        device_str=device_a,
        model_path=model_a_path,
        model_white_path=model_a_white_path,
        model_black_path=model_a_black_path,
        sequence_decode_policy=args.sequence_decode_policy_a,
        fallback_topk_multipliers=fallback_topk_multipliers_a,
    )
    runtime_b = LoadedMoveModelRuntime(
        alias=args.alias_b,
        device_str=device_b,
        model_path=model_b_path,
        model_white_path=model_b_white_path,
        model_black_path=model_b_black_path,
        sequence_decode_policy=args.sequence_decode_policy_b,
        fallback_topk_multipliers=fallback_topk_multipliers_b,
    )
    if args.verbose:
        print(
            {
                "policy_selection": {
                    "model_a": {
                        "training_objective": runtime_a.training_objective,
                        "policy_mode_default": runtime_a.policy_mode_default,
                        "artifact_rollout_horizon": runtime_a.artifact_rollout_horizon,
                        "mode": runtime_a.mode,
                    },
                    "model_b": {
                        "training_objective": runtime_b.training_objective,
                        "policy_mode_default": runtime_b.policy_mode_default,
                        "artifact_rollout_horizon": runtime_b.artifact_rollout_horizon,
                        "mode": runtime_b.mode,
                    },
                }
            }
        )

    a_wins = draws = a_losses = 0
    fallback_total_a = 0
    fallback_total_b = 0
    total_plies = 0
    per_game = []
    pgn_games: List[chess.pgn.Game] = []

    for game_idx in range(args.games):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "Model vs Model"
        game.headers["Site"] = "Local"
        game.headers["Round"] = str(game_idx + 1)

        a_color = chess.WHITE if (not args.alternate_colors or game_idx % 2 == 0) else chess.BLACK
        b_color = chess.BLACK if a_color == chess.WHITE else chess.WHITE
        game.headers["White"] = args.alias_a if a_color == chess.WHITE else args.alias_b
        game.headers["Black"] = args.alias_b if a_color == chess.WHITE else args.alias_a
        game.headers["ModelA"] = str(model_a_path or "")
        game.headers["ModelB"] = str(model_b_path or "")
        game.headers["ModelAWhite"] = str(model_a_white_path or "")
        game.headers["ModelABlack"] = str(model_a_black_path or "")
        game.headers["ModelBWhite"] = str(model_b_white_path or "")
        game.headers["ModelBBlack"] = str(model_b_black_path or "")
        node = game

        context: List[str] = []
        a_fallbacks = 0
        b_fallbacks = 0
        while not board.is_game_over(claim_draw=True) and len(context) < args.max_plies:
            if board.turn == a_color:
                mout = _model_move(runtime_a, context, args.winner_side_a, args.topk_a, board)
                move_uci = mout.get("move_uci", "")
                if not move_uci:
                    break
                if mout.get("fallback"):
                    a_fallbacks += 1
            else:
                mout = _model_move(runtime_b, context, args.winner_side_b, args.topk_b, board)
                move_uci = mout.get("move_uci", "")
                if not move_uci:
                    break
                if mout.get("fallback"):
                    b_fallbacks += 1

            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                raise RuntimeError(f"Illegal move produced during match: {move.uci()}")
            board.push(move)
            context.append(move.uci())
            node = node.add_variation(move)

        if len(context) >= args.max_plies and not board.is_game_over(claim_draw=True):
            game.headers["Termination"] = "max_plies"
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        game.headers["Result"] = result
        pgn_games.append(game)

        outcome_for_a = _result_for_player(board, a_color)
        if outcome_for_a == "win":
            a_wins += 1
        elif outcome_for_a == "draw":
            draws += 1
        elif outcome_for_a == "loss":
            a_losses += 1

        fallback_total_a += a_fallbacks
        fallback_total_b += b_fallbacks
        total_plies += len(context)
        rec = {
            "game_index": game_idx + 1,
            "a_color": "white" if a_color == chess.WHITE else "black",
            "b_color": "white" if b_color == chess.WHITE else "black",
            "result": result,
            "outcome_for_a": outcome_for_a,
            "plies": len(context),
            "fallback_moves_a": a_fallbacks,
            "fallback_moves_b": b_fallbacks,
        }
        per_game.append(rec)
        if args.progress:
            _render_progress(game_idx + 1, args.games, a_wins, draws, a_losses)
        if args.verbose:
            print({"game_done": rec})

    summary = {
        "model_a_path": str(model_a_path or ""),
        "model_b_path": str(model_b_path or ""),
        "model_a_white_path": str(model_a_white_path or ""),
        "model_a_black_path": str(model_a_black_path or ""),
        "model_b_white_path": str(model_b_white_path or ""),
        "model_b_black_path": str(model_b_black_path or ""),
        "alias_a": args.alias_a,
        "alias_b": args.alias_b,
        "games": args.games,
        "a_wins": a_wins,
        "draws": draws,
        "a_losses": a_losses,
        "a_score": a_wins + 0.5 * draws,
        "a_score_pct": ((a_wins + 0.5 * draws) / args.games) if args.games else 0.0,
        "avg_plies": (total_plies / args.games) if args.games else 0.0,
        "fallback_moves_a_total": fallback_total_a,
        "fallback_moves_b_total": fallback_total_b,
        "fallback_moves_a_avg_per_game": (fallback_total_a / args.games) if args.games else 0.0,
        "fallback_moves_b_avg_per_game": (fallback_total_b / args.games) if args.games else 0.0,
        "settings": {
            "topk_a": args.topk_a,
            "topk_b": args.topk_b,
            "winner_side_a": args.winner_side_a,
            "winner_side_b": args.winner_side_b,
            "sequence_decode_policy_a": args.sequence_decode_policy_a,
            "sequence_decode_policy_b": args.sequence_decode_policy_b,
            "fallback_topk_multipliers_a": fallback_topk_multipliers_a,
            "fallback_topk_multipliers_b": fallback_topk_multipliers_b,
            "device_a": device_a,
            "device_b": device_b,
            "max_plies": args.max_plies,
            "alternate_colors": bool(args.alternate_colors),
            "policy_mode_a": runtime_a.policy_mode_default,
            "policy_mode_b": runtime_b.policy_mode_default,
            "artifact_rollout_horizon_a": runtime_a.artifact_rollout_horizon,
            "artifact_rollout_horizon_b": runtime_b.artifact_rollout_horizon,
            "model_a_mode": runtime_a.mode,
            "model_b_mode": runtime_b.mode,
        },
        "per_game": per_game,
    }

    if args.summary_out:
        ensure_parent(args.summary_out)
        write_json(args.summary_out, summary)
    if args.pgn_out:
        ensure_parent(args.pgn_out)
        with open(args.pgn_out, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            for game in pgn_games:
                game.accept(exporter)

    print(summary)


if __name__ == "__main__":
    main()
