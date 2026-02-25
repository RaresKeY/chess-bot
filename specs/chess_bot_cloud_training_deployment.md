# Chess Bot Cloud Training Deployment (RunPod) Component

## Responsibility
Provide a modular containerized deployment package for running this repo on GPU cloud pods (RunPod-oriented) with SSH, Jupyter, HTTP inference, artifact sync, and idle autostop support.

## Code Ownership
- Module folder: `deploy/runpod_cloud_training/`
- Container image build: `deploy/runpod_cloud_training/Dockerfile`
- Startup orchestration: `deploy/runpod_cloud_training/entrypoint.sh`
- Inference HTTP service: `deploy/runpod_cloud_training/inference_api.py`
- Hugging Face sync tools:
  - `deploy/runpod_cloud_training/hf_sync.py`
  - `deploy/runpod_cloud_training/hf_auto_sync_watch.py`
- Idle autostop watchdog: `deploy/runpod_cloud_training/idle_watchdog.py`
- Module docs/config:
  - `deploy/runpod_cloud_training/README.md`
  - `deploy/runpod_cloud_training/PLAN.md`
  - `deploy/runpod_cloud_training/env.example`

## Runtime Services (default-capable)
- `sshd` for SSH CLI access (key-based auth)
- `jupyter lab` for interactive exploration and file upload/download
- FastAPI/uvicorn inference endpoint (`/healthz`, `/infer`)
- Optional Hugging Face artifact auto-sync watcher
- Optional idle watchdog for autostop behavior

## Repo Bootstrap Behavior
- Repo clone/pull at startup is supported and enabled by environment defaults (`CLONE_REPO_ON_START=1`, `GIT_AUTO_PULL=1`)
- Public GitHub repo flow only (no private clone token bootstrap logic)
- Startup requirement sync compares repo `requirements.txt` hash against a venv stamp and runs `pip install -r` when changed (or forced)

## Environment / Deployment Decisions (current)
- Prebuilt Python venv in image (`/opt/venvs/chessbot`)
- `rclone` installed for generic high-throughput transfers
- `huggingface_hub` + `hf_transfer` installed for HF artifact uploads
- Inference access supported via both SSH CLI (repo scripts) and HTTP API service
- Idle autostop watchdog script included for RunPod-style workflows

## Inference API Behavior (module service)
- Resolves model path from explicit arg/env or latest `*.pt` under repo `artifacts/`
- Loads model once at startup on selected device (`auto` -> CUDA when available)
- Logs startup runtime metadata and can print a startup loading progress bar
- Per-request response includes `topk`, `predicted_uci`, `best_legal`, `device`, and latency
- Touches a heartbeat file so idle watchdog can treat API traffic as activity

## Hugging Face Sync Behavior
- `hf_sync.py`: one-shot upload of matching files from a source dir to a HF repo
- `hf_auto_sync_watch.py`: polls for changed files and triggers sync on change
- Pattern-based file selection defaults to model/metrics artifacts (`*.pt`, `*.json`)

## Idle Watchdog Behavior
- Polls GPU utilization/memory (`nvidia-smi`) and connection/process activity
- Treats SSH, Jupyter, inference API traffic, matching process patterns, or heartbeat updates as activity
- When idle timeout is exceeded:
  - can call RunPod GraphQL API (`AUTOSTOP_ACTION=runpod_api`)
  - or terminate PID1 (`AUTOSTOP_ACTION=exit`) as a fallback action
- RunPod GraphQL payload is implemented with retry/fallback mutation shapes and may require env override if API schema changes

## Current Limitations
- No built-in reverse proxy/TLS or API auth layer (assumes trusted networking or RunPod access controls)
- Private repo clone/bootstrap is intentionally out of scope in this module version
- RunPod API stop mutation schema may drift; watchdog includes best-effort defaults but may need adjustment
