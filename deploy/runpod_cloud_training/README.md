# RunPod Cloud Training Deployment Module

RunPod-oriented container module for this repo with:
- SSH access (CLI workflows, `rsync` / `scp`)
- JupyterLab (explore files, run notebooks, download artifacts)
- HTTP inference API (loads latest model or explicit model path)
- Hugging Face sync scripts (manual and automatic watcher)
- Idle autostop watchdog (RunPod API or PID1 exit fallback)

This module is intentionally self-contained under `deploy/runpod_cloud_training/`.

## Files
- `PLAN.md`: saved implementation plan and design choices
- `Dockerfile`: GPU-ready base image with venv, Jupyter, FastAPI, `rclone`, SSH
- `entrypoint.sh`: startup orchestration (clone/pull repo, requirement sync, start services)
- `inference_api.py`: reusable HTTP inference service
- `hf_sync.py`: one-shot Hugging Face sync
- `hf_auto_sync_watch.py`: periodic auto-sync watcher
- `idle_watchdog.py`: idle detection + autostop hook
- `train_baseline_preset.sh`: one-command training launcher using current repo baseline architecture defaults
- `env.example`: environment variable template
- `requirements-extra.txt`: deployment-only Python dependencies

## Build
From repo root:

```bash
docker build -f deploy/runpod_cloud_training/Dockerfile -t chess-bot-runpod:latest .
```

Or use the repo helper (tags image with git SHA + `latest`, optional push):

```bash
IMAGE_REPO=ghcr.io/<user>/chess-bot-runpod \
bash scripts/build_runpod_image.sh
```

Push to registry:

```bash
IMAGE_REPO=ghcr.io/<user>/chess-bot-runpod \
PUSH_IMAGE=1 \
bash scripts/build_runpod_image.sh
```

## Recommended RunPod Setup
- Mount a persistent volume to `/workspace`
- Expose ports:
  - `22` (SSH)
  - `8888` (Jupyter)
  - `8000` (inference API)
- Set environment variables from `deploy/runpod_cloud_training/env.example`

## Startup Behavior (default)
1. Configure SSH for user `runner` using `AUTHORIZED_KEYS`
2. Clone/pull `https://github.com/RaresKeY/chess-bot.git` into `/workspace/chess-bot`
3. Compare repo `requirements.txt` hash and `pip install -r` if changed
4. Start `sshd`, JupyterLab, and inference API
5. Optionally start HF auto-sync and idle watchdog

## Core Environment Variables
Repo/bootstrap:
- `REPO_URL`, `REPO_REF`, `REPO_DIR`
- `CLONE_REPO_ON_START=1`
- `GIT_AUTO_PULL=1`

Python/runtime:
- `VENV_DIR=/opt/venvs/chessbot`
- `SYNC_REQUIREMENTS_ON_START=1`
- `FORCE_PIP_SYNC=0`
- `RUNPOD_PHASE_TIMING_ENABLED=1`
- `RUNPOD_PHASE_TIMING_LOG=/workspace/chess-bot/artifacts/timings/runpod_phase_times.jsonl`
- `RUNPOD_PHASE_TIMING_RUN_ID` (optional correlation id for entrypoint + training preset timings)

SSH:
- `AUTHORIZED_KEYS` (newline-separated public keys)

Jupyter:
- `START_JUPYTER=1`
- `JUPYTER_PORT=8888`
- `JUPYTER_TOKEN` (generated if omitted)

Inference API:
- `START_INFERENCE_API=1`
- `INFERENCE_API_MODEL_PATH=latest`
- `INFERENCE_API_DEVICE=auto`
- `INFERENCE_API_PORT=8000`
- `INFERENCE_API_VERBOSE=1`

HF sync:
- `START_HF_WATCH=1`
- `HF_REPO_ID=<user-or-org>/<repo>`
- `HF_REPO_TYPE=model` (or `dataset`)
- `HF_TOKEN=<token>`
- `HF_SYNC_SOURCE_DIR=/workspace/chess-bot/artifacts`
- `HF_SYNC_PATTERNS=*.pt,*.json`
- `HF_SYNC_INTERVAL_SECONDS=120`

Idle autostop:
- `START_IDLE_WATCHDOG=1`
- `IDLE_TIMEOUT_SECONDS=3600`
- `AUTOSTOP_ACTION=runpod_api` (or `exit`)
- `RUNPOD_API_KEY`, `RUNPOD_POD_ID`

## Inference API
Endpoints:
- `GET /healthz`
- `POST /infer`

Example:

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "context": ["e2e4", "e7e5", "g1f3"],
    "winner_side": "W",
    "topk": 10
  }'
```

The API now computes phase + side-to-move features explicitly so inference matches current trained artifacts that use the expanded model head.

## Fastest Training Start (Current Architecture Preset)
On a pod shell (SSH or Jupyter terminal), run:

```bash
bash /opt/runpod_cloud_training/train_baseline_preset.sh
```

This launcher:
- auto-detects the latest dataset dir under `data/dataset/` containing `train.jsonl` + `val.jsonl` (unless overridden)
- uses the current bumped baseline defaults (`embed=256`, `hidden=512`, `num_layers=2`, `dropout=0.15`, `epochs=40`, `lr=2e-4`)
- enables phase + side-to-move features
- enables `ReduceLROnPlateau` and early stopping
- logs phase timing JSONL records (`resolve_dataset_paths`, `train_baseline`) into `RUNPOD_PHASE_TIMING_LOG`

Entry-point startup also logs phase timings (`clone_or_update_repo`, `sync_repo_requirements`, service starts, etc.) to the same JSONL file so you can compare local smoke vs RunPod timings later.

Common overrides:

```bash
TRAIN_DATASET_DIR=/workspace/chess-bot/data/dataset/elite_2025-11_cap4 \
TRAIN_BATCH_SIZE=2048 \
TRAIN_NUM_WORKERS=6 \
OUTPUT_PATH=/workspace/chess-bot/artifacts/model_runpod.pt \
METRICS_OUT=/workspace/chess-bot/artifacts/model_runpod_metrics.json \
bash /opt/runpod_cloud_training/train_baseline_preset.sh
```

Extra CLI args can be appended via `TRAIN_EXTRA_ARGS`, for example:

```bash
TRAIN_EXTRA_ARGS="--phase-weight-endgame 2.5 --lr-plateau-patience 4" \
bash /opt/runpod_cloud_training/train_baseline_preset.sh
```

## Local RunPod-Style Smoke Test (Docker)
Use the repo helper to run an ephemeral `--rm` container with the real entrypoint and training preset:

```bash
bash scripts/runpod_local_smoke_test.sh
```

This writes timing records to:
- `artifacts/timings/runpod_phase_times.jsonl`

Records include a `source` field (for example `local_runpod_smoke`, `runpod_entrypoint`, `runpod_train_preset`) and a shared `run_id` for correlation.

## Fast File Transfer Options
- `rsync` over SSH (fast and resumable)
- JupyterLab file browser download/upload
- `rclone` (S3/GCS/Drive/etc.)
- Hugging Face auto-sync (`hf_auto_sync_watch.py`) for model artifacts

## Notes / Limits
- Public repo clone flow only (no private repo token bootstrap included)
- RunPod GraphQL autostop mutation may evolve; `idle_watchdog.py` supports endpoint/mutation overrides via env vars
- If repo dependencies drift significantly from image extras, startup `pip install -r requirements.txt` handles sync
