# Chess Bot Cloud Training Deployment (RunPod) Component

## Responsibility
Provide a modular containerized deployment package for running this repo on GPU cloud pods (RunPod-oriented) with SSH, Jupyter, HTTP inference, artifact sync, and idle autostop support.

## Code Ownership
- Module folder: `deploy/runpod_cloud_training/`
- Optional RunPod provisioning helper: `scripts/runpod_provision.py`
- Optional image build/push helper: `scripts/build_runpod_image.sh`
- Optional local smoke helper (Docker, RunPod-style entrypoint/training): `scripts/runpod_local_smoke_test.sh`
- Optional host-side CLI diagnostics helper: `scripts/runpod_cli_doctor.sh`
- Optional host-side RunPod regression checks wrapper: `scripts/runpod_regression_checks.sh`
- Optional host-side quick launch wrapper: `scripts/runpod_quick_launch.sh`
- Optional host-side modular RunPod lifecycle scripts: `scripts/runpod_cycle_*.sh`
- Optional host-side HF dataset publish/fetch helpers: `scripts/hf_dataset_publish.py`, `scripts/hf_dataset_fetch.py`
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
- Host-side cycle scripts now default to safer SSH client behavior (`StrictHostKeyChecking=accept-new` with persistent `config/runpod_known_hosts`) instead of the older insecure `/tmp` + host-key-checking-disabled pattern; overrides remain available via `RUNPOD_SSH_HOST_KEY_CHECKING` and `RUNPOD_SSH_KNOWN_HOSTS_FILE`
- Entrypoint now explicitly unlocks the `runner` account (while keeping password auth disabled) so Ubuntu/Debian `sshd` accepts public-key auth for direct mapped SSH
- Direct mapped SSH is intended for the `runner` user; `entrypoint.sh` configures `PermitRootLogin no` and `AllowUsers runner`, so `root@<public-ip>:<mapped-port>` public-key login is expected to fail
- Entrypoint uses `wait -n` to supervise services; if any enabled child process exits, cleanup can stop the remaining services (including `sshd`)

## Repo Bootstrap Behavior
- Repo clone/pull at startup is supported and enabled by environment defaults (`CLONE_REPO_ON_START=1`, `GIT_AUTO_PULL=1`)
- Public GitHub repo flow only (no private clone token bootstrap logic)
- Startup requirement sync compares repo `requirements.txt` hash against a venv stamp and runs `pip install -r` when changed (or forced)
- If `REPO_DIR` exists but is non-git:
  - existing empty dir: clone into it
  - existing non-empty dir with `requirements.txt`: treat as mounted checkout and skip clone/pull
  - existing non-empty dir without `.git`/`requirements.txt`: log and skip clone/pull (operator must choose a valid `REPO_DIR` or mount a repo)

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
- GHCR image repository names must be lowercase (for example `ghcr.io/rareskey/chess-bot-runpod`); `scripts/build_runpod_image.sh` now normalizes `IMAGE_REPO` to lowercase before tagging/building.
- GitHub CLI (`gh auth login`) auth and Docker registry auth are separate:
  - `gh auth` authenticates GitHub API/CLI usage on the host
  - `docker login ghcr.io` authenticates local Docker/Buildx pushes to GHCR
- On Linux Mint host setups, Docker can be configured to store GHCR credentials in the system keyring via the `secretservice` credential helper (`"credsStore": "secretservice"` in `~/.docker/config.json`) instead of plaintext auth entries.
- Phase timings are logged in JSONL under a conventional artifacts path by default: `artifacts/timings/runpod_phase_times.jsonl` (configurable via `RUNPOD_PHASE_TIMING_LOG`)

## RunPod Lifecycle Semantics (Stop vs Terminate)
- `stop` in RunPod context means halting pod compute (for example via GraphQL `podStop`); the pod resource and storage can remain.
- `terminate` means deleting the pod resource entirely (typically via RunPod REST pod delete).
- These are intentionally different operator actions and billing outcomes; specs/scripts should not use them interchangeably.

## Inference API Behavior (module service)
- Resolves model path from explicit arg/env or latest `*.pt` under repo `artifacts/`
- Loads model once at startup on selected device (`auto` -> CUDA when available)
- Logs startup runtime metadata and can print a startup loading progress bar
- Per-request response includes `topk`, `predicted_uci`, `best_legal`, `device`, and latency
- API request inference computes phase and side-to-move feature inputs so predictions align with artifacts trained using the expanded model head
- Touches a heartbeat file so idle watchdog can treat API traffic as activity
- Entry-point startup now skips launching the inference API (with a log message) when `torch` is not installed in the image/runtime venv, avoiding a full container restart loop during smoke/provisioning scenarios

## RunPod Training Preset Launcher
- `train_baseline_preset.sh` provides a one-command training path inside the container (image-baked path: `bash /opt/runpod_cloud_training/train_baseline_preset.sh`)
- In active RunPod pod workflows, prefer the repo copy when available (`bash "$REPO_DIR/deploy/runpod_cloud_training/train_baseline_preset.sh"`) because the repo clone can be newer than the image-baked `/opt/...` script (avoids stale feature behavior)
- Auto-detects latest dataset dir under `data/dataset/` containing `train.jsonl` + `val.jsonl` when `TRAIN_DATASET_DIR` / `TRAIN_PATH` / `VAL_PATH` are not set
- Supports optional HF dataset bootstrap mode for reusable datasets:
  - `HF_FETCH_LATEST_ALL_DATASETS=1` fetches the latest published version of every dataset under `HF_DATASET_REPO_ID` + `HF_DATASET_PATH_PREFIX`
  - fetched datasets are extracted under `HF_DATASET_CACHE_DIR`
  - `scripts/hf_dataset_fetch.py` aggregate manifests now include `aggregate_by_format` (for example `game_jsonl_runtime_splice_v1`, `splice_rows_legacy`) when dataset manifests expose `dataset_format`
  - the preset can filter aggregate training inputs by schema/format via `HF_DATASET_SCHEMA_FILTER` (`auto` default prefers compact game datasets when present, otherwise legacy splice datasets)
  - the preset aggregates discovered `train.jsonl` and `val.jsonl` files from the selected schema bucket and passes them to `scripts/train_baseline.py` via repeated `--train` / `--val` flags
  - fetch summary manifest is written to `HF_DATASET_FETCH_MANIFEST`
  - `HF_USE_EXISTING_FETCH_MANIFEST=1` lets the preset reuse a previously-created aggregate fetch manifest (skip a second HF fetch and train from the cached dataset set)
  - "latest" selection is lexicographic on the version path segment, so sortable version labels (for example timestamp-prefixed `validated-YYYYMMDDTHHMMSSZ`) are recommended
  - if the selected dataset rows are compact game-level rows (`moves`/`moves_uci` schema), the preset detects this and passes runtime splice controls through to `scripts/train_baseline.py` (`TRAIN_RUNTIME_MIN_CONTEXT`, `TRAIN_RUNTIME_MIN_TARGET`, `TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME`)
- Uses current repo baseline architecture/training defaults:
  - `embed_dim=256`, `hidden_dim=512`, `num_layers=2`, `dropout=0.15`
  - `epochs=40`, `lr=2e-4`
  - phase + side-to-move features enabled
  - `ReduceLROnPlateau` + early stopping enabled (preset patience/min-delta values)
- Supports env overrides for dataset paths, output paths, batch size/worker count, endgame phase weight, optional progress event stream path (`TRAIN_PROGRESS_JSONL_OUT`), and arbitrary extra train flags (`TRAIN_EXTRA_ARGS`)
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
- `scripts/runpod_local_smoke_test.sh` supports optional `SMOKE_PROGRESS_JSONL_OUT` passthrough to set `TRAIN_PROGRESS_JSONL_OUT` in the container for progress-event producer smoke checks

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

## Validated Dataset HF Publish/Fetch Flow (host-side)
- `scripts/hf_dataset_publish.py` publishes reusable validated datasets to a HF **dataset** repo with versioned paths:
  - `validated_datasets/<dataset_name>/<version>/`
- Publish script behavior:
  - validates required `train.jsonl`/`val.jsonl` by default
  - generates `manifest.json` and `checksums.sha256`
  - publish manifest includes `dataset_format` (detected from `stats.json` or row schema sniffing) and embeds `stats_json` when present for downstream schema-aware fetch/training flows
  - uploads a compressed `tar.gz` bundle by default (faster transfer for JSONL-heavy datasets)
  - supports `--archive-format none` to upload raw files instead
  - token lookup defaults to keyring (`service=huggingface`, `username=codex_hf_write_token`), with `--token` / `HF_TOKEN` overrides
  - sets `HF_HUB_ENABLE_HF_TRANSFER=1` by default for faster HF transfers when supported by the active env
  - supports `--dry-run` to inspect repo path/versioning without network upload
- `scripts/hf_dataset_fetch.py` fetches a published dataset version from the HF dataset repo and extracts the archive into a local destination by default
- `scripts/hf_dataset_fetch.py --all-latest` can fetch the latest version for every dataset under a repo prefix and emit an aggregate manifest containing all extracted `train.jsonl` / `val.jsonl` paths plus `aggregate_by_format` buckets keyed by dataset manifest `dataset_format` (used by the RunPod train preset HF mode)
- Recommended workflow for RunPod:
  - publish validated dataset once from host
  - fetch into pod persistent volume/cache on demand
  - point training to the fetched dataset path (`TRAIN_DATASET_DIR` / `TRAIN_PATH` / `VAL_PATH`) or enable `HF_FETCH_LATEST_ALL_DATASETS=1` to train over all latest published datasets automatically

## Idle Watchdog Behavior
- Polls GPU utilization/memory (`nvidia-smi`) and connection/process activity
- Treats SSH, Jupyter, inference API traffic, matching process patterns, or heartbeat updates as activity
- When idle timeout is exceeded:
  - can call RunPod GraphQL API (`AUTOSTOP_ACTION=runpod_api`) to request pod stop (compute halt, not pod deletion)
  - or terminate PID1 (`AUTOSTOP_ACTION=exit`) as a fallback action (container process exit, not RunPod pod termination/deletion)
- RunPod GraphQL payload is implemented with retry/fallback mutation shapes and may require env override if API schema changes
- Watchdog strips any accidental `api_key` query parameter from the configured GraphQL endpoint, authenticates with `Authorization: Bearer ...`, and sends explicit `Accept` / `User-Agent` headers for better compatibility with upstream filtering/WAF behavior

## RunPod API Provisioning Helper
- `scripts/runpod_provision.py` supports:
  - `gpu-search` (GraphQL GPU type search/ranking by cloud, memory, and price filters)
  - `template-list` (REST template listing/filtering)
  - `provision` (choose GPU + template and create a Pod via REST, with optional wait loop)
- `provision` supports template selection by exact `--template-id` or name/substring `--template-name`
- Pod create flow supports common chess-bot ports (`22/tcp`, `8888/http`, `8000/http`) and optional env injection
- Key lookup order is explicit arg -> env `RUNPOD_API_KEY` -> keyring (`runpod` / `RUNPOD_API_KEY`); scripts should avoid echoing CLI/env values containing the token
- Provisioning helper strips any accidental `api_key` query parameter from GraphQL endpoints and uses bearer headers for GraphQL/REST authentication
- GraphQL GPU listing (`gpuTypes`) can fail with `HTTP 403` even when REST template listing works if the API key/account lacks GraphQL access; helper now raises an actionable error instead of a raw traceback
- `provision --gpu-type-id <id>` now supports a fallback path when GraphQL GPU discovery is denied: it proceeds with the explicit GPU type ID (without GraphQL validation) and emits a warning

## Current Limitations
- No built-in reverse proxy/TLS or API auth layer (assumes trusted networking or RunPod access controls)
- Private repo clone/bootstrap is intentionally out of scope in this module version
- RunPod API stop mutation schema may drift; watchdog includes best-effort defaults but may need adjustment
- Watchdog stop mutation currently requests object subfields (`id`, `desiredStatus`) on `podStop` to match the observed GraphQL schema
- This module/docs distinguish RunPod `stop` from pod `terminate`; operator cleanup may still require a separate pod delete step after stop/autostop
- For host-side smoke/lifecycle validation, disabling optional services (Jupyter/inference/HF-watchdog/idle-watchdog) can improve SSH stability because a single child-process exit otherwise tears down `sshd`
- First local Docker smoke run can be slow when `SYNC_REQUIREMENTS_ON_START=1` because the image venv intentionally installs repo `requirements.txt` (including `torch`) at startup when the repo requirements hash is not yet stamped
- If `START_INFERENCE_API=1` and `torch` is missing from the deployment venv, the inference API itself remains unavailable until repo requirement sync installs `torch`; entrypoint now degrades by skipping API start instead of crashing the container
- Image-vs-repo script version skew is possible for `train_baseline_preset.sh` if the image was built before newer HF/progress features landed in the repo; operator flows should prefer the repo copy for the latest behavior

## Observed RunPod Failure + Fix (2026-02-26)
- Observed on a community GPU smoke pod (`NVIDIA GeForce RTX 4090`) provisioned from template `chess-bot-training`
- Failure mode:
  - `REPO_DIR` (`/workspace/chess-bot`) existed and was non-empty but not a valid git checkout, so clone failed (`destination path already exists and is not an empty directory`)
  - requirement sync skipped (`requirements.txt` missing in that directory)
  - inference API startup failed (`ModuleNotFoundError: No module named 'torch'`)
  - entrypoint exited after child process failure, causing a pod restart loop
- Fixes implemented:
  - `entrypoint.sh` now handles non-git `REPO_DIR` states more explicitly (empty dir clone / mounted checkout detect / non-git warning)
  - `entrypoint.sh` now preflights `torch` and skips inference API start when missing
  - `scripts/runpod_provision.py` now keeps helper env injection opt-in by default (prevents overriding template service flags unexpectedly)
- Detailed operator-facing problem/solution report saved at `artifacts/reports/runpod_cycle_observations_2026-02-26.md`
- Follow-on operator note from the same date:
  - direct mapped SSH auth may fail if template `AUTHORIZED_KEYS` is malformed
  - RunPod SSH gateway access can still succeed and provide a recovery shell for fixing `/home/runner/.ssh/authorized_keys`
  - direct mapped SSH auth can also fail even with a correct `authorized_keys` when the `runner` account is created in a locked state (`useradd` default on Ubuntu); `entrypoint.sh` now unlocks `runner` at startup and `Dockerfile` also clears the password during image build (`passwd -d runner`) while `PasswordAuthentication no` remains enforced

## Regression Tests (current)
- `tests/test_runpod_api_helpers.py` covers:
  - GraphQL endpoint sanitization (`api_key` query param stripped)
  - bearer header auth for provisioning helper + idle watchdog stop API call
  - GPU ranking helper filtering/sorting
  - template selection by partial name
