# Chess Bot MVP

Python chess-move prediction project that builds datasets from PGNs, trains an LSTM next-move model, evaluates it, and provides browser UIs to inspect games and play against the model locally.

Implementation scaffold for the plan in `implement.md`.

## Quick Start (Fresh Clone / Outside Container)
Fastest way to test the PGN viewer after cloning.

1. Setup + generate viewer HTML:

```bash
python3 -m venv --clear .venv && . .venv/bin/activate
python -m ensurepip --upgrade || true
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/render_game_viewer.py --pgn data/raw/sample_games.pgn --game-index 0 --out-html artifacts/viewer/game_viewer.html
```

2. Serve repo root and open the viewer:

```bash
PYTHONPATH=. python scripts/serve_viewer.py --dir . --port 8000 --example-path artifacts/viewer/game_viewer.html
```

Open `http://127.0.0.1:8000/artifacts/viewer/game_viewer.html`.

## Setup Component
Owns environment/bootstrap concerns only.

- Entry script: none
- Component files:
  - `requirements.txt`
  - `specs/chess_bot_mvp_plan.md`

```bash
python3 -m venv --clear /work/.venv
/work/.venv/bin/python -m ensurepip --upgrade || true
/work/.venv/bin/python -m pip install -r requirements.txt
```

## Component Map
Each stage is intentionally separable and can be run independently.

1. `Validation`:
   - Script: `scripts/validate_games.py`
   - Module: `src/chessbot/validation.py`
2. `Splicing + Splits`:
   - Script: `scripts/build_splice_dataset.py`
   - Module: `src/chessbot/splicing.py`
3. `Training`:
   - Script: `scripts/train_baseline.py`
   - Modules: `src/chessbot/training.py`, `src/chessbot/model.py`
4. `Evaluation`:
   - Script: `scripts/eval_model.py`
   - Module: `src/chessbot/evaluation.py`
5. `Inference`:
   - Script: `scripts/infer_move.py`
   - Module: `src/chessbot/inference.py`
6. `Game Viewer`:
   - Script: `scripts/render_game_viewer.py`
   - Module: `src/chessbot/viewer.py`
   - Assets: `assets/pieces/cburnett/*.svg`
7. `Viewer Local Server`:
   - Script: `scripts/serve_viewer.py`
   - Runtime: Python stdlib `http.server`
8. `Play vs Model`:
   - Script: `scripts/play_vs_model_server.py`
   - Module: `src/chessbot/play_vs_model.py`
   - Reuses model artifact + piece assets
9. `IO Utilities`:
   - Shared module: `src/chessbot/io_utils.py`

## 1) Validation Component
Converts raw PGN files into clean accepted/rejected artifacts.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/validate_games.py \
  --input data/raw/games.pgn \
  --valid-out data/validated/valid_games.jsonl \
  --invalid-out data/validated/invalid_games.csv \
  --summary-out data/validated/summary.json
```

## 2) Splicing Component
Builds supervised training rows and leak-safe game-level splits.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/build_splice_dataset.py \
  --input data/validated/valid_games.jsonl \
  --output-dir data/dataset \
  --k 4 \
  --min-context 8 \
  --min-target 1
```

## 3) Training Component
Trains winner-aware next-move predictor and writes model artifact.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/train_baseline.py \
  --train data/dataset/train.jsonl \
  --val data/dataset/val.jsonl \
  --device auto \
  --num-workers 4 \
  --amp \
  --output artifacts/model.pt \
  --metrics-out artifacts/train_metrics.json
```

Outside the container (host GPU), use explicit CUDA device selection:

```bash
PYTHONPATH=. python scripts/train_baseline.py \
  --train data/dataset/elite_2025-11_cap4/train.jsonl \
  --val data/dataset/elite_2025-11_cap4/val.jsonl \
  --device cuda:0 \
  --num-workers 8 \
  --amp \
  --output artifacts/model.pt \
  --metrics-out artifacts/train_metrics.json
```

## 4) Evaluation Component
Computes top-k and legal-rate metrics from test split.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/eval_model.py \
  --model artifacts/model.pt \
  --data data/dataset/test.jsonl \
  --device cpu
```

## 5) Inference Component
Produces top-k candidate moves and best legal move for a context.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/infer_move.py \
  --model artifacts/model.pt \
  --context "e2e4 e7e5 g1f3 b8c6" \
  --device cpu
```

## 6) Game Viewer Component
Renders a local HTML chessboard viewer for a PGN game with left/right arrow navigation.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/render_game_viewer.py \
  --pgn data/raw/sample_games.pgn \
  --game-index 0 \
  --out-html artifacts/viewer/game_viewer.html
```

Open `artifacts/viewer/game_viewer.html` in a browser, then use on-screen arrows or keyboard Left/Right.

## 7) Viewer Local Server Utility
Serves any chosen directory over local HTTP (document root = `--dir`), which is useful for opening generated viewer HTML and local piece assets in a browser.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/serve_viewer.py \
  --dir /work \
  --port 8000 \
  --example-path artifacts/viewer/game_viewer.html
```

Then open `http://127.0.0.1:8000/artifacts/viewer/game_viewer.html`.

Portable usage (outside this container/repo layout) works the same. Example:

```bash
python scripts/serve_viewer.py --dir /path/to/site --port 8000 --example-path index.html
```

## 8) Play-vs-Model Component
Runs an interactive local web app (board + move list + server-side legality checks) to play against the model.

```bash
PYTHONPATH=/work /work/.venv/bin/python scripts/play_vs_model_server.py \
  --dir /work \
  --model artifacts/model.pt \
  --port 8020
```

Open `http://127.0.0.1:8020/play-vs-model`.

Portable (fresh clone) usage:

```bash
PYTHONPATH=. python scripts/play_vs_model_server.py --dir . --model artifacts/model.pt --port 8020
```

## Artifacts By Component
- Validation: `data/validated/valid_games.jsonl`, `data/validated/invalid_games.csv`, `data/validated/summary.json`
- Splicing: `data/dataset/train.jsonl`, `data/dataset/val.jsonl`, `data/dataset/test.jsonl`, `data/dataset/stats.json`
- Training: `artifacts/model.pt`, `artifacts/train_metrics.json`
- Evaluation: `artifacts/eval_metrics.json`
- Viewer: `artifacts/viewer/game_viewer.html`
- Play-vs-model: dynamic page + JSON API (no static artifact required)

## Notes
- Dataset split is game-level to avoid leakage.
- Validation requires legal replay of every move.
- Default training is winner-aware using winner embedding and weighted winner-side examples.
- Viewer uses the open-source `cburnett` SVG piece set from the Lichess `lila` repository (`assets/pieces/cburnett`).
- Play-vs-model uses legal fallback moves when the model predicts no legal move, so interactive games can continue.
