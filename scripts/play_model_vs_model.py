#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import chess
import chess.pgn
import torch

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.inference import best_legal_from_topk
from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.model import NextMoveLSTM, encode_tokens, side_to_move_id_from_context_len, winner_to_id
from src.chessbot.phase import PHASE_UNKNOWN, classify_context_phase, phase_to_id


def _find_latest_model(artifacts_dir: Path) -> Path:
    candidates = [p for p in artifacts_dir.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise SystemExit(f"No model artifacts found under {artifacts_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _resolve_model_path(text: str) -> Path:
    if text == "latest":
        return _find_latest_model(REPO_ROOT / "artifacts")
    return Path(text).resolve()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


class LoadedMoveModelRuntime:
    def __init__(self, model_path: Path, *, device_str: str, alias: str):
        self.model_path = model_path
        self.alias = alias
        self.device = torch.device(device_str)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

        artifact = torch.load(str(model_path), map_location="cpu")
        self.artifact = artifact
        runtime_meta = artifact.get("runtime") or {}
        self.training_objective = str(runtime_meta.get("training_objective") or artifact.get("training_objective") or "single_step_next_move")
        try:
            self.artifact_rollout_horizon = max(0, int(runtime_meta.get("rollout_horizon") or 0))
        except Exception:
            self.artifact_rollout_horizon = 0
        self.policy_mode_default = "rollout" if self.training_objective.startswith("multistep_") else "next"
        self.vocab = artifact["vocab"]
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        cfg = artifact["config"]
        self.model = NextMoveLSTM(vocab_size=len(self.vocab), **cfg).to(self.device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def infer(self, context: List[str], winner_side: str, topk: int) -> Dict:
        original_context_len = len(context)
        context_ids = encode_tokens(context, self.vocab)
        if not context_ids:
            context_ids = [self.vocab.get("<UNK>", 1)]
        tokens = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(context_ids)], dtype=torch.long, device=self.device)
        winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long, device=self.device)
        phase_name = str(classify_context_phase(context).get("phase", PHASE_UNKNOWN))
        phases = torch.tensor([phase_to_id(phase_name)], dtype=torch.long, device=self.device)
        side_to_moves = torch.tensor([side_to_move_id_from_context_len(original_context_len)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(tokens, lengths, winners, phases, side_to_moves)
            k = max(1, min(int(topk), int(logits.shape[-1])))
            pred_ids = logits.topk(k, dim=1).indices[0].detach().cpu().tolist()
        topk_tokens = [self.inv_vocab.get(i, "") for i in pred_ids]
        return {
            "topk": topk_tokens,
            "best_legal": best_legal_from_topk(topk_tokens, context),
            "device": str(self.device),
        }

    def infer_rollout(self, context: List[str], winner_side: str, topk: int, rollout_plies: int, fallback_legal: bool = True) -> Dict:
        board = chess.Board()
        for uci in context:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                raise ValueError(f"Illegal context move: {uci}")
            board.push(mv)
        local_context = list(context)
        rollout: List[str] = []
        step_debug: List[Dict] = []
        for _ in range(max(0, int(rollout_plies))):
            out = self.infer(local_context, winner_side=winner_side, topk=topk)
            topk_tokens = out.get("topk", [])
            legal = out.get("best_legal", "")
            chosen = legal
            fallback_used = False
            if not chosen and fallback_legal and not board.is_game_over(claim_draw=True):
                fallback_mv = next(iter(board.legal_moves), None)
                if fallback_mv is not None:
                    chosen = fallback_mv.uci()
                    fallback_used = True
            step_debug.append({"topk": topk_tokens, "best_legal": legal, "chosen": chosen, "fallback": fallback_used})
            if not chosen:
                break
            mv = chess.Move.from_uci(chosen)
            if mv not in board.legal_moves:
                break
            board.push(mv)
            local_context.append(chosen)
            rollout.append(chosen)
            if board.is_game_over(claim_draw=True):
                break
        return {
            "rollout": rollout,
            "first_move": rollout[0] if rollout else "",
            "steps_generated": len(rollout),
            "fallback_moves": sum(1 for x in step_debug if x.get("fallback")),
            "step_debug": step_debug,
        }


def _model_move(
    runtime: LoadedMoveModelRuntime,
    context: List[str],
    winner_side: str,
    topk: int,
    board: chess.Board,
) -> Dict:
    if runtime.policy_mode_default == "rollout":
        rollout_len = max(1, runtime.artifact_rollout_horizon or 1)
        rout = runtime.infer_rollout(context=context, winner_side=winner_side, topk=topk, rollout_plies=rollout_len, fallback_legal=True)
        first_move = str(rout.get("first_move") or "")
        step_debug = rout.get("step_debug") or []
        first_debug = step_debug[0] if step_debug else {}
        topk_tokens = list(first_debug.get("topk") or [])
        predicted_uci = topk_tokens[0] if topk_tokens else ""
        uci = first_move
        fallback_flag = bool(first_debug.get("fallback"))
    else:
        out = runtime.infer(context=context, winner_side=winner_side, topk=topk)
        topk_tokens = out.get("topk", [])
        predicted_uci = topk_tokens[0] if topk_tokens else ""
        uci = out.get("best_legal", "")
        fallback_flag = False
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
    parser.add_argument("--model-a", required=True, help="Model A artifact path or 'latest'")
    parser.add_argument("--model-b", required=True, help="Model B artifact path or 'latest'")
    parser.add_argument("--alias-a", default="model_a", help="Display name for model A")
    parser.add_argument("--alias-b", default="model_b", help="Display name for model B")
    parser.add_argument("--games", type=int, default=20, help="Number of games")
    parser.add_argument("--topk-a", type=int, default=10, help="Model A top-k candidates")
    parser.add_argument("--topk-b", type=int, default=10, help="Model B top-k candidates")
    parser.add_argument("--winner-side-a", default="W", choices=["W", "B", "D", "?"], help="Conditioning token for model A")
    parser.add_argument("--winner-side-b", default="W", choices=["W", "B", "D", "?"], help="Conditioning token for model B")
    parser.add_argument("--device-a", default="auto", help="Torch device for model A (cpu/cuda/auto)")
    parser.add_argument("--device-b", default="auto", help="Torch device for model B (cpu/cuda/auto)")
    parser.add_argument("--max-plies", type=int, default=300, help="Hard cap on plies per game")
    parser.add_argument("--alternate-colors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-out", default="", help="Optional JSON summary output path")
    parser.add_argument("--pgn-out", default="", help="Optional PGN output path for all games")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_a_path = _resolve_model_path(args.model_a)
    model_b_path = _resolve_model_path(args.model_b)
    device_a = _resolve_device(args.device_a)
    device_b = _resolve_device(args.device_b)

    if args.verbose:
        print(
            {
                "match_start": {
                    "model_a": str(model_a_path),
                    "model_b": str(model_b_path),
                    "alias_a": args.alias_a,
                    "alias_b": args.alias_b,
                    "games": args.games,
                    "topk_a": args.topk_a,
                    "topk_b": args.topk_b,
                    "winner_side_a": args.winner_side_a,
                    "winner_side_b": args.winner_side_b,
                    "device_a": device_a,
                    "device_b": device_b,
                    "max_plies": args.max_plies,
                    "alternate_colors": bool(args.alternate_colors),
                }
            }
        )

    runtime_a = LoadedMoveModelRuntime(model_a_path, device_str=device_a, alias=args.alias_a)
    runtime_b = LoadedMoveModelRuntime(model_b_path, device_str=device_b, alias=args.alias_b)
    if args.verbose:
        print(
            {
                "policy_selection": {
                    "model_a": {
                        "training_objective": runtime_a.training_objective,
                        "policy_mode_default": runtime_a.policy_mode_default,
                        "artifact_rollout_horizon": runtime_a.artifact_rollout_horizon,
                    },
                    "model_b": {
                        "training_objective": runtime_b.training_objective,
                        "policy_mode_default": runtime_b.policy_mode_default,
                        "artifact_rollout_horizon": runtime_b.artifact_rollout_horizon,
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
        game.headers["ModelA"] = str(model_a_path)
        game.headers["ModelB"] = str(model_b_path)
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
        "model_a_path": str(model_a_path),
        "model_b_path": str(model_b_path),
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
            "device_a": device_a,
            "device_b": device_b,
            "max_plies": args.max_plies,
            "alternate_colors": bool(args.alternate_colors),
            "policy_mode_a": runtime_a.policy_mode_default,
            "policy_mode_b": runtime_b.policy_mode_default,
            "artifact_rollout_horizon_a": runtime_a.artifact_rollout_horizon,
            "artifact_rollout_horizon_b": runtime_b.artifact_rollout_horizon,
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
