#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import chess
import chess.engine
import chess.pgn
import torch

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.inference import infer_from_artifact_on_device
from src.chessbot.io_utils import ensure_parent, write_json


def _find_latest_model(artifacts_dir: Path) -> Path:
    candidates = [p for p in artifacts_dir.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise SystemExit(f"No model artifacts found under {artifacts_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _model_move(
    artifact: Dict,
    context: List[str],
    winner_side: str,
    topk: int,
    device_str: str,
    board: chess.Board,
) -> Dict:
    out = infer_from_artifact_on_device(
        artifact=artifact,
        context=context,
        winner_side=winner_side,
        topk=topk,
        device_str=device_str,
    )
    uci = out.get("best_legal", "")
    if uci:
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            mv = None
        if mv is not None and mv in board.legal_moves:
            out["move_uci"] = uci
            out["fallback"] = False
            return out

    fallback = next(iter(board.legal_moves), None)
    if fallback is None:
        out["move_uci"] = ""
        out["fallback"] = True
        return out
    out["move_uci"] = fallback.uci()
    out["fallback"] = True
    return out


def _engine_move(engine: chess.engine.SimpleEngine, board: chess.Board, movetime_ms: int, depth: int) -> chess.Move:
    limit = chess.engine.Limit(time=movetime_ms / 1000.0 if movetime_ms > 0 else None, depth=depth if depth > 0 else None)
    result = engine.play(board, limit)
    if result.move is None:
        raise RuntimeError("Engine returned no move")
    return result.move


def _game_result_for_model(board: chess.Board, model_color: chess.Color) -> str:
    result = board.result(claim_draw=True)
    if result == "1-0":
        return "win" if model_color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if model_color == chess.BLACK else "loss"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def _render_progress(i: int, total: int, wins: int, draws: int, losses: int) -> None:
    width = 28
    frac = min(1.0, max(0.0, i / max(total, 1)))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[match] [{bar}] {i}/{total} "
        f"W/D/L={wins}/{draws}/{losses}"
    )
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play model vs UCI engine and summarize results")
    parser.add_argument("--model", default="latest", help="Model artifact path or 'latest'")
    parser.add_argument("--engine", required=True, help="Path to UCI engine binary (e.g. stockfish)")
    parser.add_argument("--games", type=int, default=20, help="Number of games")
    parser.add_argument("--movetime-ms", type=int, default=100, help="Engine move time per move in ms")
    parser.add_argument("--engine-depth", type=int, default=0, help="Engine fixed depth (0 disables)")
    parser.add_argument("--topk", type=int, default=10, help="Model top-k candidates")
    parser.add_argument("--winner-side", default="W", choices=["W", "B", "D", "?"], help="Model conditioning token")
    parser.add_argument("--device", default="cpu", help="Torch device for model inference (cpu/cuda/auto)")
    parser.add_argument("--max-plies", type=int, default=300, help="Hard cap on plies to avoid runaway games")
    parser.add_argument("--summary-out", default="", help="Optional JSON summary output path")
    parser.add_argument("--pgn-out", default="", help="Optional PGN output path for all games")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_path = _find_latest_model(REPO_ROOT / "artifacts") if args.model == "latest" else Path(args.model).resolve()
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    if args.verbose:
        print(
            {
                "match_start": {
                    "model_path": str(model_path),
                    "engine_path": str(Path(args.engine).resolve()),
                    "games": args.games,
                    "movetime_ms": args.movetime_ms,
                    "engine_depth": args.engine_depth,
                    "model_topk": args.topk,
                    "winner_side": args.winner_side,
                    "device": device_str,
                    "max_plies": args.max_plies,
                }
            }
        )

    artifact = torch.load(str(model_path), map_location="cpu")

    wins = draws = losses = 0
    fallback_moves_total = 0
    total_plies = 0
    per_game = []
    pgn_games: List[chess.pgn.Game] = []

    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        for game_idx in range(args.games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Model vs Engine"
            game.headers["Site"] = "Local"
            game.headers["Round"] = str(game_idx + 1)
            model_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
            game.headers["White"] = "model" if model_color == chess.WHITE else "engine"
            game.headers["Black"] = "engine" if model_color == chess.WHITE else "model"
            node = game

            context: List[str] = []
            model_fallbacks = 0
            while not board.is_game_over(claim_draw=True) and len(context) < args.max_plies:
                if board.turn == model_color:
                    mout = _model_move(
                        artifact=artifact,
                        context=context,
                        winner_side=args.winner_side,
                        topk=args.topk,
                        device_str=device_str,
                        board=board,
                    )
                    move_uci = mout.get("move_uci", "")
                    if not move_uci:
                        break
                    move = chess.Move.from_uci(move_uci)
                    if mout.get("fallback"):
                        model_fallbacks += 1
                else:
                    move = _engine_move(engine, board, movetime_ms=args.movetime_ms, depth=args.engine_depth)

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

            outcome = _game_result_for_model(board, model_color)
            if outcome == "win":
                wins += 1
            elif outcome == "draw":
                draws += 1
            elif outcome == "loss":
                losses += 1

            fallback_moves_total += model_fallbacks
            total_plies += len(context)
            per_game.append(
                {
                    "game_index": game_idx + 1,
                    "model_color": "white" if model_color == chess.WHITE else "black",
                    "result": result,
                    "outcome_for_model": outcome,
                    "plies": len(context),
                    "model_fallback_moves": model_fallbacks,
                }
            )
            if args.progress:
                _render_progress(game_idx + 1, args.games, wins, draws, losses)
            if args.verbose:
                print(
                    {
                        "game_done": {
                            "game": game_idx + 1,
                            "model_color": per_game[-1]["model_color"],
                            "result": result,
                            "outcome_for_model": outcome,
                            "plies": len(context),
                            "model_fallback_moves": model_fallbacks,
                        }
                    }
                )

    summary = {
        "model_path": str(model_path),
        "engine_path": str(Path(args.engine).resolve()),
        "games": args.games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": wins + 0.5 * draws,
        "score_pct": ((wins + 0.5 * draws) / args.games) if args.games else 0.0,
        "avg_plies": (total_plies / args.games) if args.games else 0.0,
        "model_fallback_moves_total": fallback_moves_total,
        "model_fallback_moves_avg_per_game": (fallback_moves_total / args.games) if args.games else 0.0,
        "settings": {
            "movetime_ms": args.movetime_ms,
            "engine_depth": args.engine_depth,
            "topk": args.topk,
            "winner_side": args.winner_side,
            "device": device_str,
            "max_plies": args.max_plies,
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
                f.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
