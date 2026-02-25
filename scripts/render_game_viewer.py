#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from src.chessbot.io_utils import ensure_parent
from src.chessbot.viewer import render_game_viewer_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a navigable HTML chess game viewer from PGN")
    parser.add_argument("--pgn", required=True, help="Path to PGN file")
    parser.add_argument("--game-index", type=int, default=0, help="0-based game index in PGN")
    parser.add_argument("--out-html", default="artifacts/viewer/game_viewer.html")
    parser.add_argument(
        "--piece-base",
        default=None,
        help="Path from output HTML to piece image directory (default auto-relative to repo assets/pieces/cburnett)",
    )
    args = parser.parse_args()

    ensure_parent(args.out_html)
    piece_base = args.piece_base
    if piece_base is None:
        out_dir = Path(args.out_html).resolve().parent
        repo_root = Path(__file__).resolve().parents[1]
        assets_dir = (repo_root / "assets" / "pieces" / "cburnett").resolve()
        piece_base = os.path.relpath(str(assets_dir), start=str(out_dir))

    result = render_game_viewer_html(
        pgn_path=args.pgn,
        out_html=args.out_html,
        game_index=args.game_index,
        piece_base=piece_base,
    )
    print(result)


if __name__ == "__main__":
    main()
