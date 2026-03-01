# Chess Bot

Python chess-bot platform for dataset generation, move-prediction training, evaluation/inference, local UI play, and RunPod/Vast cloud orchestration.

_Last updated: 2026-03-01_

## TL;DR

- Build validated datasets from PGN.
- Train baseline and dual-sequence models.
- Evaluate with top-k + legality metrics.
- Run local game viewer and play-vs-model UI.
- Automate cloud lifecycle flows (RunPod and Vast).

## Quick Navigation

- Specs index (read first): `specs/_readme.md`
- Runtime control mapping: `specs/chess_bot_param_config_mappings.md`
- Core implementation: `src/chessbot/`
- CLI/script entrypoints: `scripts/`
- Tests: `tests/`
- Deploy modules: `deploy/runpod_cloud_training/`, `deploy/vast_cloud_training/`
- Project overview doc: `local_docs/PROJECT_SUMMARY_OVERVIEW.md`

## Component Matrix

Each component includes script entrypoint(s), core module(s), and primary artifacts.

| Component | CLI Entrypoint(s) | Core Module(s) | Primary Artifacts |
|---|---|---|---|
| Validation | `scripts/validate_games.py` | `src/chessbot/validation.py` | `data/validated/*/valid_games.jsonl`, `invalid_games.csv`, `summary.json` |
| Dataset Build (game-level) | `scripts/build_game_dataset.py`, `scripts/batch_build_compact_caches.py` | `src/chessbot/splicing.py`, `src/chessbot/phase.py` | `data/dataset/*_game/{train,val,test}.jsonl`, `stats.json` |
| Runtime Splice Cache | `scripts/build_runtime_splice_cache.py`, `scripts/validate_runtime_splice_cache.py`, `scripts/build_runtime_splice_vocab_meta.py` | `src/chessbot/splicing.py` | `data/dataset/*_game/runtime_splice_cache/*` |
| Legacy Splice Dataset | `scripts/build_splice_dataset.py` | `src/chessbot/splicing.py` | `data/dataset/{train,val,test}.jsonl`, `stats.json` |
| Baseline Training | `scripts/train_baseline.py` | `src/chessbot/training.py`, `src/chessbot/model.py`, `src/chessbot/phase.py` | model `.pt`, training metrics JSON |
| Dual-Sequence Training | `scripts/train_dual_sequence.py` | `src/chessbot/training_dual_sequence.py`, `src/chessbot/model.py` | dual-model artifacts + metrics |
| Evaluation | `scripts/eval_model.py` | `src/chessbot/evaluation.py`, `src/chessbot/phase.py` | eval metrics JSON |
| Inference | `scripts/infer_move.py` | `src/chessbot/inference.py` | stdout JSON/selected legal move |
| Game Viewer | `scripts/render_game_viewer.py` | `src/chessbot/viewer.py` | `artifacts/viewer/game_viewer.html` |
| Viewer Server | `scripts/serve_viewer.py` | stdlib server utility | local HTTP serving of generated assets |
| Play vs Model | `scripts/play_vs_model_server.py`, `main.py` | `src/chessbot/play_vs_model.py` | interactive `/play-vs-model` page + JSON API |
| Lichess Bot + Preview | `scripts/lichess_bot.py`, `scripts/serve_lichess_preview.py`, `scripts/run_lichess_bot_with_preview.py` | `src/chessbot/lichess_bot.py` | live/archived game preview artifacts |
| RunPod Direct API Lane | `scripts/runpod_provision.py`, `scripts/runpod_cycle_*.sh`, `scripts/runpod_full_train_easy*.sh` | `src/chessbot/runpod_cycle_verify.py`, `src/chessbot/secrets.py` | `artifacts/runpod_cycles/<run-id>/*`, `config/runpod_tracked_pods.jsonl` |
| RunPod SDK Lane | `scripts/runpod_sdk_component.py`, `scripts/runpod_sdk_cycle_*.sh` | `src/chessbot/runpod_sdk_component.py` | SDK cycle logs/reports under `artifacts/runpod_cycles/*` |
| Vast Lane | `scripts/vast_provision.py`, `scripts/vast_cycle_*.sh` | provider helpers + shared scripts | Vast lifecycle logs/reports, tracked metadata |

## Fast Start (Local)

1. Create environment and install deps:

```bash
python3 -m venv --clear .venv
. .venv/bin/activate
python -m ensurepip --upgrade || true
python -m pip install -r requirements.txt
```

2. Quick viewer smoke:

```bash
PYTHONPATH=. .venv/bin/python scripts/render_game_viewer.py \
  --pgn data/raw/sample_games.pgn \
  --game-index 0 \
  --out-html artifacts/viewer/game_viewer.html

PYTHONPATH=. .venv/bin/python scripts/serve_viewer.py \
  --dir . \
  --port 8000 \
  --example-path artifacts/viewer/game_viewer.html
```

Open `http://127.0.0.1:8000/artifacts/viewer/game_viewer.html`.

## Canonical Local Pipeline

```bash
# 1) Validate games
PYTHONPATH=. .venv/bin/python scripts/validate_games.py \
  --input data/raw/sample_games.pgn \
  --valid-out data/validated/sample/valid_games.jsonl \
  --invalid-out data/validated/sample/invalid_games.csv \
  --summary-out data/validated/sample/summary.json

# 2) Build game dataset
PYTHONPATH=. .venv/bin/python scripts/build_game_dataset.py \
  --input data/validated/sample/valid_games.jsonl \
  --output-dir data/dataset/sample_game

# 3) Build runtime splice cache
PYTHONPATH=. .venv/bin/python scripts/build_runtime_splice_cache.py \
  --dataset-dir data/dataset/sample_game

# 4) Train baseline
PYTHONPATH=. .venv/bin/python scripts/train_baseline.py \
  --train data/dataset/sample_game/train.jsonl \
  --val data/dataset/sample_game/val.jsonl \
  --output artifacts/model.pt \
  --metrics-out artifacts/train_metrics.json

# 5) Evaluate
PYTHONPATH=. .venv/bin/python scripts/eval_model.py \
  --model artifacts/model.pt \
  --data data/dataset/sample_game/test.jsonl \
  --device cpu

# 6) Infer
PYTHONPATH=. .venv/bin/python scripts/infer_move.py \
  --model artifacts/model.pt \
  --context "e2e4 e7e5 g1f3 b8c6" \
  --device cpu
```

## RunPod Quick Ops (Up to Date)

Doctor/auth check:

```bash
bash scripts/runpod_cli_doctor.sh
```

Easy full-train wrapper (defaults to `RUNPOD_GPU_TYPE_ID='NVIDIA GeForce RTX 5090'`):

```bash
RUNPOD_CYCLE_RUN_ID="fulltrain-$(date -u +%Y%m%dT%H%M%SZ)" \
RUNPOD_CLOUD_TYPE=COMMUNITY \
bash scripts/runpod_full_train_easy.sh
```

A5000 non-spot start example for SSH cycle workflows:

```bash
RUNPOD_GPU_TYPE_ID='NVIDIA RTX A5000' \
RUNPOD_GPU_COUNT=2 \
RUNPOD_CLOUD_TYPE=COMMUNITY \
RUNPOD_TEMPLATE_NAME='chess-bot-training' \
bash scripts/runpod_cycle_start.sh
```

Full smoke cycle:

```bash
bash scripts/runpod_cycle_full_smoke.sh
```

Terminate all tracked pods when done:

```bash
bash scripts/runpod_cycle_terminate_all_tracked.sh --yes
```

For full controls and defaults, use:
- `specs/chess_bot_runpod_cli_controls.md`
- `specs/chess_bot_runpod_preferred_training_flow.md`
- `specs/chess_bot_runpod_sdk_component.md`

## Testing

Run full test suite:

```bash
bash scripts/test.sh
```

Compatibility entrypoint:

```bash
bash scripts/run_all_tests.sh
```

Targeted regression lanes for cloud workflows live under `tests/test_runpod_*` and `tests/test_vast_*`.

## Repo Layout

| Path | Purpose |
|---|---|
| `src/chessbot/` | Core code: validation, splicing, training, evaluation, inference, play, secrets |
| `scripts/` | End-user/operator CLI scripts |
| `tests/` | Pytest regression/unit tests |
| `specs/` | Source-of-truth behavior and control docs |
| `config/` | Catalogs/tracked state/config manifests |
| `data/` | Raw/validated/dataset/live-play data |
| `artifacts/` | Models, reports, run outputs, logs |
| `deploy/` | Cloud deployment modules |
| `local_docs/` | Internal summaries/experiment notes |

## Documentation Rules

- Read `specs/_readme.md` before making changes.
- When behavior changes, update relevant spec(s) and tests.
- When non-constant runtime controls change (env/CLI/config), update `specs/chess_bot_param_config_mappings.md`.

