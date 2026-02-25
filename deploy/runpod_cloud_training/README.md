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
- `env.example`: environment variable template
- `requirements-extra.txt`: deployment-only Python dependencies

## Build
From repo root:

```bash
docker build -f deploy/runpod_cloud_training/Dockerfile -t chess-bot-runpod:latest .
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

## Fast File Transfer Options
- `rsync` over SSH (fast and resumable)
- JupyterLab file browser download/upload
- `rclone` (S3/GCS/Drive/etc.)
- Hugging Face auto-sync (`hf_auto_sync_watch.py`) for model artifacts

## Notes / Limits
- Public repo clone flow only (no private repo token bootstrap included)
- RunPod GraphQL autostop mutation may evolve; `idle_watchdog.py` supports endpoint/mutation overrides via env vars
- If repo dependencies drift significantly from image extras, startup `pip install -r requirements.txt` handles sync
