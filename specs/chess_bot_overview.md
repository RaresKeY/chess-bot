# Chess Bot Overview

## Scope
Reproducible supervised chess move-prediction pipeline:
1. Validate PGN games
2. Build splice-based dataset
3. Train baseline next-move model
4. Evaluate top-k and legality metrics
5. Infer best legal move from context
6. Render a navigable HTML viewer for PGN games
7. Provide an interactive play-vs-model web app

## Architecture Pattern
- `scripts/*.py`: thin orchestration CLIs
- `src/chessbot/*`: reusable logic modules
- `scripts/__init__.py`: allows importing CLI modules in tests/tooling (`import scripts.<name>`)
- `data/*`: generated datasets and reports
- `artifacts/*`: model and metrics artifacts
- `assets/pieces/cburnett/*`: local SVG chess piece images used by viewer
- `scripts/play_vs_model_server.py` + `src/chessbot/play_vs_model.py`: interactive gameplay UI/API server

## Core Decisions
- Move encoding: UCI tokens.
- Primary target: next move (`target[0]`) from splice window.
- Winner-aware training supported via winner embedding and weighted loss.
- Split strategy: by `game_id` only (no sample-level splitting).

## Artifact Contract
- Validation:
  - `data/validated/valid_games.jsonl`
  - `data/validated/invalid_games.csv`
  - `data/validated/summary.json`
- Dataset:
  - `data/dataset/train.jsonl`
  - `data/dataset/val.jsonl`
  - `data/dataset/test.jsonl`
  - `data/dataset/stats.json`
- Training/Eval:
  - `artifacts/model.pt`
  - `artifacts/train_metrics.json`
  - `artifacts/eval_metrics.json`
- Viewer:
  - `artifacts/viewer/game_viewer.html` (or other generated HTML output)
- Play-vs-model:
  - dynamic page served at `/play-vs-model` (no static artifact required)

## Current Training-Scale Corpus Snapshot
- A full validated elite monthly corpus has been generated:
  - `data/validated/elite_2025-11/valid_games.jsonl` (`279,613` valid games)
- This is sufficient scale for baseline supervised training experiments beyond toy samples.
- A streaming splice dataset build has also been generated from that corpus:
  - `data/dataset/elite_2025-11_cap4/*` (~`941,451` total samples across train/val/test, capped at 4 samples/game)
