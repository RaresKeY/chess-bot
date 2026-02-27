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

## RunPod Smoke Test (Host CLI)
Host-side lifecycle smoke test for the RunPod training template/image (provision -> dataset push -> remote train -> collect -> local validate -> stop).

Prereqs:
- RunPod pod template exists (for example `chess-bot-training`)
- host has `ssh`, `rsync`, `jq`, `curl`
- no manual host SSH key setup required for easy/full flows; scripts auto-generate a managed no-passphrase temp key at `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}`
- RunPod API key available via CLI arg/env/keyring, with dotenv fallback (`.env.runpod`/`.env`)

Preferred auth (keyring):

```bash
bash scripts/runpod_cli_doctor.sh
```

Optional explicit `.env` fallback (instead of keyring):

1. Create a local file from the example and fill in your key:

```bash
cp .env.runpod.example .env.runpod
```

2. Load it explicitly for the current shell/session:

```bash
set -a
. ./.env.runpod
set +a
```

3. Re-run the doctor:

```bash
bash scripts/runpod_cli_doctor.sh
```

Optional: populate repo-local `.env` directly from keyring (writes `RUNPOD_API_KEY`, `HF_TOKEN`, `LICHESS_BOT_TOKEN`):

```bash
PYTHONPATH=. .venv/bin/python scripts/populate_env_from_keyring.py
```

If you want placeholders for missing keyring entries instead of a failure:

```bash
PYTHONPATH=. .venv/bin/python scripts/populate_env_from_keyring.py --allow-missing
```

Build and push the RunPod image (recommended to use a pinned tag, not only `latest`):

```bash
# default repo for this project:
# IMAGE_REPO=ghcr.io/rareskey/chess-bot-runpod
IMAGE_REPO=ghcr.io/<your-user>/chess-bot-runpod \
IMAGE_TAG=<your-tag> \
PUSH_IMAGE=1 \
bash scripts/build_runpod_image.sh
```

Then update the RunPod template image to the new pinned tag in the RunPod UI.

Run the full smoke cycle:

```bash
bash scripts/runpod_cycle_full_smoke.sh
```

Notes for current smoke-cycle behavior:
- The cycle launcher injects a unique `REPO_DIR` per run under `/workspace` to avoid stale/root-owned repo directories on reused volumes.
- The cycle launcher disables optional pod services by default during smoke tests (`START_JUPYTER=0`, `START_INFERENCE_API=0`, `START_HF_WATCH=0`, `START_IDLE_WATCHDOG=0`) so `sshd` remains stable.
- The full smoke cycle ends with RunPod `stop` (compute halted), not pod deletion.

Cleanup (delete tracked pods, not just stop):

```bash
bash scripts/runpod_cycle_terminate_all_tracked.sh --yes
```

## Hugging Face Dataset Publish/Fetch Flow (Reusable Validated Datasets)
Use this for validated datasets you want to reuse across multiple RunPod training runs. It avoids repeated host->pod `rsync` uploads and gives you versioned dataset paths.

Preferred auth:
- Hugging Face write token in keyring:
  - `service=huggingface`
  - `username=codex_hf_write_token`

Optional explicit env fallback:

```bash
cp .env.hf_dataset.example .env.hf_dataset
set -a
. ./.env.hf_dataset
set +a
```

Token resolution order for publish/fetch scripts: `--token` -> `HF_TOKEN` -> keyring -> dotenv (`.env.hf_dataset`/`.env`).

Publish a validated dataset directory (default upload is a compressed `tar.gz` bundle + `manifest.json` + `checksums.sha256`):

```bash
PYTHONPATH=. .venv/bin/python scripts/hf_dataset_publish.py \
  --repo-id <hf-user-or-org>/chess-bot-datasets \
  --dataset-dir data/dataset/elite_2025-11_cap4 \
  --dataset-name elite_2025_11_cap4 \
  --version 2026-02-26-valid-v1
```

Dry-run first (no network upload):

```bash
PYTHONPATH=. .venv/bin/python scripts/hf_dataset_publish.py \
  --repo-id <hf-user-or-org>/chess-bot-datasets \
  --dataset-dir data/dataset/_smoke_runpod \
  --dataset-name smoke_runpod \
  --version test-$(date -u +%Y%m%dT%H%M%SZ) \
  --dry-run
```

Fetch later (host or pod), then point training to the extracted path:

```bash
PYTHONPATH=. .venv/bin/python scripts/hf_dataset_fetch.py \
  --repo-id <hf-user-or-org>/chess-bot-datasets \
  --dataset-name elite_2025_11_cap4 \
  --version 2026-02-26-valid-v1 \
  --dest-dir /workspace/datasets
```

After fetch, use the extracted dataset path with training:
- example extracted path: `/workspace/datasets/elite_2025_11_cap4/2026-02-26-valid-v1/dataset`
- then set `TRAIN_DATASET_DIR=/workspace/datasets/elite_2025_11_cap4/2026-02-26-valid-v1/dataset`

Notes:
- Install host deps if needed: `.venv/bin/pip install huggingface_hub hf_transfer`
- The publish/fetch helpers default to `HF_HUB_ENABLE_HF_TRANSFER=1`
- For tiny smoke datasets, direct `rsync` remains simpler; use HF for reusable validated datasets

### RunPod: Train On All Latest HF Datasets
Feasible and now supported.

The RunPod training preset can fetch the latest published version of every dataset under your HF dataset repo prefix and train on all of them together (future uploads are picked up automatically as long as version names sort lexicographically, e.g. `validated-YYYYMMDDTHHMMSSZ`).

Host-side modular flow (skip dataset push):

```bash
bash scripts/runpod_cycle_start.sh
export RUNPOD_TRAIN_FROM_HF_LATEST_ALL=1
export RUNPOD_HF_DATASET_REPO_ID='LogicLark-QuantumQuill/chess-bot-datasets'
export RUNPOD_HF_DATASET_PATH_PREFIX='validated_datasets'   # optional (default)
bash scripts/runpod_cycle_train.sh
bash scripts/runpod_cycle_collect.sh
bash scripts/runpod_cycle_local_validate.sh
bash scripts/runpod_cycle_stop.sh
```

Inside the pod (manual usage), equivalent envs for the preset:

```bash
export HF_FETCH_LATEST_ALL_DATASETS=1
export HF_DATASET_REPO_ID='LogicLark-QuantumQuill/chess-bot-datasets'
export HF_DATASET_PATH_PREFIX='validated_datasets'
bash /opt/runpod_cloud_training/train_baseline_preset.sh
```
