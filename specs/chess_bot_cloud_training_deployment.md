# Chess Bot Cloud Training Deployment (RunPod) Component

## Responsibility
Provide a modular containerized deployment package for running this repo on GPU cloud pods (RunPod-oriented) with SSH, Jupyter, HTTP inference, artifact sync, and idle autostop support.

## Code Ownership
- Module folder: `deploy/runpod_cloud_training/`
- Optional RunPod provisioning helper: `scripts/runpod_provision.py`
- Optional image build/push helper: `scripts/build_runpod_image.sh`
- Optional local smoke helper (Docker, RunPod-style entrypoint/training): `scripts/runpod_local_smoke_test.sh`
- Container image build: `deploy/runpod_cloud_training/Dockerfile`
- Startup orchestration: `deploy/runpod_cloud_training/entrypoint.sh`
- Inference HTTP service: `deploy/runpod_cloud_training/inference_api.py`
- RunPod training launcher preset: `deploy/runpod_cloud_training/train_baseline_preset.sh`
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
- RunPod API helpers support API key lookup via env `RUNPOD_API_KEY` or keyring (`service=runpod`, `username=RUNPOD_API_KEY`)
- RunPod account API keys are treated as high-sensitivity control credentials:
  - send only to RunPod API endpoints
  - pass via `Authorization: Bearer ...` headers (not URL query params)
  - never print/log the key in script output
- Docker image for the RunPod module can be built directly with `docker build` or via `scripts/build_runpod_image.sh` (tags with git SHA + `latest`, optional registry push)
- Phase timings are logged in JSONL under a conventional artifacts path by default: `artifacts/timings/runpod_phase_times.jsonl` (configurable via `RUNPOD_PHASE_TIMING_LOG`)

## Inference API Behavior (module service)
- Resolves model path from explicit arg/env or latest `*.pt` under repo `artifacts/`
- Loads model once at startup on selected device (`auto` -> CUDA when available)
- Logs startup runtime metadata and can print a startup loading progress bar
- Per-request response includes `topk`, `predicted_uci`, `best_legal`, `device`, and latency
- API request inference computes phase and side-to-move feature inputs so predictions align with artifacts trained using the expanded model head
- Touches a heartbeat file so idle watchdog can treat API traffic as activity

## RunPod Training Preset Launcher
- `train_baseline_preset.sh` provides a one-command training path inside the container (`bash /opt/runpod_cloud_training/train_baseline_preset.sh`)
- Auto-detects latest dataset dir under `data/dataset/` containing `train.jsonl` + `val.jsonl` when `TRAIN_DATASET_DIR` / `TRAIN_PATH` / `VAL_PATH` are not set
- Uses current repo baseline architecture/training defaults:
  - `embed_dim=256`, `hidden_dim=512`, `num_layers=2`, `dropout=0.15`
  - `epochs=40`, `lr=2e-4`
  - phase + side-to-move features enabled
  - `ReduceLROnPlateau` + early stopping enabled (preset patience/min-delta values)
- Supports env overrides for dataset paths, output paths, batch size/worker count, endgame phase weight, and arbitrary extra train flags (`TRAIN_EXTRA_ARGS`)
- Emits timing records to the shared phase timing JSONL (`resolve_dataset_paths`, `train_baseline`) with `source=runpod_train_preset`

## Phase Timing Logging Convention
- Default log path: `${REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl` (or `RUNPOD_PHASE_TIMING_LOG`)
- Format: JSON Lines (`.jsonl`), one record per phase completion
- Shared fields:
  - `ts_epoch_ms`
  - `source` (`runpod_entrypoint`, `runpod_train_preset`, `local_runpod_smoke`)
  - `run_id` (correlates multiple scripts in one run)
  - `phase`
  - `status` (`ok` / `error`)
  - `elapsed_ms`
  - optional `exit_code` on failures
- `entrypoint.sh` logs startup/bootstrap/service-start phases
- `train_baseline_preset.sh` logs dataset resolution + training command runtime
- `scripts/runpod_local_smoke_test.sh` logs host-side phases (`prepare_smoke_dataset`, `docker_run_smoke_training`) and passes the same `run_id` into the container
- `scripts/runpod_local_smoke_test.sh` now pre-creates/chmods the shared timing log file before `docker run` to improve append compatibility for local rootless Docker bind mounts (and logs `prepare_timing_log_file`)

## Local Timing Sample (2026-02-25)
- Timing log path: `artifacts/timings/runpod_phase_times.jsonl`
- Run ID: `local-smoke-20260225T144213Z-357798`
- Smoke parameters (local): `SMOKE_TRAIN_ROWS=64`, `SMOKE_VAL_ROWS=16`, `SMOKE_EPOCHS=1`, `SMOKE_BATCH_SIZE=32`
- Observed local smoke phase durations (host-side wrapper):
  - `prepare_smoke_dataset`: `6 ms`
  - `prepare_timing_log_file`: `7 ms`
  - `docker_run_smoke_training`: `148349 ms` (~148.3s, includes container startup + repo requirement sync + 1-epoch smoke train)
- Local rootless Docker note:
  - Container-side timing rows (`source=runpod_entrypoint`, `source=runpod_train_preset`) are instrumented and expected on RunPod, but were still not observed in the host bind-mounted timing log during this local smoke sample; host-side wrapper timings are recorded and usable for baseline local-vs-RunPod comparison.

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
- Watchdog strips any accidental `api_key` query parameter from the configured GraphQL endpoint and authenticates with `Authorization: Bearer ...`

## RunPod API Provisioning Helper
- `scripts/runpod_provision.py` supports:
  - `gpu-search` (GraphQL GPU type search/ranking by cloud, memory, and price filters)
  - `template-list` (REST template listing/filtering)
  - `provision` (choose GPU + template and create a Pod via REST, with optional wait loop)
- `provision` supports template selection by exact `--template-id` or name/substring `--template-name`
- Pod create flow supports common chess-bot ports (`22/tcp`, `8888/http`, `8000/http`) and optional env injection
- Key lookup order is explicit arg -> env `RUNPOD_API_KEY` -> keyring (`runpod` / `RUNPOD_API_KEY`); scripts should avoid echoing CLI/env values containing the token
- Provisioning helper strips any accidental `api_key` query parameter from GraphQL endpoints and uses bearer headers for GraphQL/REST authentication

## Current Limitations
- No built-in reverse proxy/TLS or API auth layer (assumes trusted networking or RunPod access controls)
- Private repo clone/bootstrap is intentionally out of scope in this module version
- RunPod API stop mutation schema may drift; watchdog includes best-effort defaults but may need adjustment
- First local Docker smoke run can be slow when `SYNC_REQUIREMENTS_ON_START=1` because the image venv intentionally installs repo `requirements.txt` (including `torch`) at startup when the repo requirements hash is not yet stamped

## Regression Tests (current)
- `tests/test_runpod_api_helpers.py` covers:
  - GraphQL endpoint sanitization (`api_key` query param stripped)
  - bearer header auth for provisioning helper + idle watchdog stop API call
  - GPU ranking helper filtering/sorting
  - template selection by partial name
