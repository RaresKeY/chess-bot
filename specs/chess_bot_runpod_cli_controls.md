# Chess Bot RunPod CLI Controls

## Responsibility
Document host-side CLI workflows for building/pushing the RunPod image, diagnosing RunPod API auth, provisioning pods from templates, and controlling a running pod over SSH/Jupyter/HTTP.

## Code Ownership
- `scripts/build_runpod_image.sh`
- `scripts/runpod_provision.py`
- `scripts/runpod_cli_doctor.sh`
- `scripts/runpod_quick_launch.sh`
- `scripts/runpod_regression_checks.sh`
- `scripts/cloud_connectivity_health_checks.sh`
- `scripts/cloud_checks/providers/runpod.sh`
- `scripts/runpod_connectivity_telemetry_checks.sh`
- `scripts/runpod_cycle_common.sh`
- `scripts/runpod_cycle_start.sh`
- `scripts/runpod_cycle_push_dataset.sh`
- `scripts/runpod_cycle_train.sh`
- `scripts/runpod_cycle_collect.sh`
- `scripts/runpod_cycle_local_validate.sh`
- `scripts/runpod_cycle_stop.sh`
- `scripts/runpod_cycle_terminate_all_tracked.sh`
- `scripts/runpod_cycle_full_smoke.sh`
- `scripts/runpod_cycle_watch_progress.sh`
- `scripts/runpod_cycle_watchdog.sh`
- `scripts/runpod_cycle_full_train_hf.sh`
- `scripts/runpod_cycle_report_style.py`
- `scripts/runpod_cycle_summarize_gpu_observations.py`
- `scripts/runpod_full_train_easy.sh`
- `scripts/runpod_cycle_benchmark_matrix.sh`
- `scripts/runpod_cycle_benchmark_10k_sixpack.sh`
- `scripts/runpod_file_transfer.sh`
- `scripts/telemetry/telemetry_common.sh`
- `scripts/telemetry_control.sh`
- `scripts/telemetry_emit_event.sh`
- `scripts/telemetry_checkpoint.sh`
- `scripts/telemetry_healthchecks_ping.sh`
- `scripts/telemetry_status.sh`
- `scripts/telemetry_watchdog.sh`
- `scripts/hf_dataset_publish.py`
- `scripts/hf_dataset_fetch.py`
- Runtime pod interfaces used by operators:
  - `ssh`
  - `scp` / `rsync`
  - `curl` (inference API)

## Host Runtime Assumptions
- Commands are run from the host workspace (not inside the RunPod pod)
- Prefer project venv Python when available (`.venv/bin/python`)
- RunPod API key lookup in `scripts/runpod_provision.py`:
  1. `--api-key`
  2. env `RUNPOD_API_KEY`
  3. keyring (`service=runpod`, `username=RUNPOD_API_KEY`)
  4. dotenv fallback (`RUNPOD_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.runpod`, then `.env`)

## RunPod Pod Lifecycle Terms (Do Not Conflate)
- `provision` / `start`: create a new RunPod pod resource and start compute.
- `stop`: RunPod `podStop` action on an existing pod. This halts compute, but the pod resource and attached storage can remain.
- `terminate`: delete the pod resource (REST `DELETE /pods/<id>`). This is not the same as `stop`.
- `AUTOSTOP_ACTION=exit` (inside the container) stops PID1/processes in the runtime; it is not a host-side RunPod pod deletion request.

## Image Build / Push Control
- `scripts/build_runpod_image.sh` builds the RunPod image from `deploy/runpod_cloud_training/Dockerfile`
- `IMAGE_REPO` is required; script now normalizes it to lowercase before tagging (GHCR requires lowercase repository names)
- project default image repo value: `ghcr.io/rareskey/chess-bot-runpod`
- Default tags:
  - `<IMAGE_REPO>:<git-short-sha>`
  - `<IMAGE_REPO>:latest`
- `PUSH_IMAGE=1` switches to `docker buildx build --push`

## RunPod Provisioning CLI (`runpod_provision.py`)
- Subcommands:
  - `template-list`: REST-based template discovery/filtering
  - `gpu-search`: GraphQL `gpuTypes` query + local ranking/filtering by cloud/memory/price
  - `provision`: GraphQL GPU selection + REST pod creation
- Template list can succeed while GPU search fails if the API key lacks GraphQL access/scopes (REST and GraphQL permissions may differ)

## GraphQL GPU Search Failure Behavior (current)
- `gpu-search` now converts raw GraphQL `HTTP 403` traces into an actionable error message
- Error guidance explicitly suggests:
  - use a key with GraphQL access
  - launch from RunPod UI template
  - or use `provision --gpu-type-id <gpuTypeId>` to bypass GPU discovery
- Important diagnostic caveat (observed 2026-02-26 host run):
  - Python `urllib` requests to `https://api.runpod.io/graphql` returned Cloudflare `error code: 1010` (`HTTP 403`) for `myself` and `gpuTypes`
  - equivalent `curl` requests with the same bearer token returned `HTTP 200` for both queries
  - implication: a reported GraphQL `403` from the Python helper may be a client-signature/WAF issue rather than an API key permission problem
  - helper scripts now send an explicit `User-Agent` (+ `Accept`) header on RunPod REST/GraphQL requests to avoid this false-negative path on the observed host

## Provision Fallback with Explicit GPU Type ID
- `provision --gpu-type-id <id>` now bypasses GraphQL GPU discovery/validation and uses the explicit GPU type directly in pod creation
- This preserves a CLI provisioning path for keys that can use REST pod creation/template APIs but cannot access GraphQL `gpuTypes`
- Practical effect:
  - wrapper scripts that pass explicit `--gpu-type-id` no longer trigger GraphQL `gpu-search` access errors during provisioning
  - invalid GPU type IDs are now rejected by RunPod pod creation (REST) instead of local GraphQL pre-validation

## RunPod CLI Helper Scripts
- `scripts/runpod_cli_doctor.sh`
  - checks local key source (env vs keyring)
  - tolerates missing local keyring backend (reports keyring unavailable/error instead of crashing)
  - checks REST auth via `template-list`
  - checks GraphQL auth via `gpu-search`
  - reports GraphQL `403` as an expected access-scope limitation (with remediation hint) when REST still works
- `scripts/runpod_regression_checks.sh`
  - runs RunPod-focused unit tests (`tests/test_runpod_api_helpers.py`, `tests/test_runpod_local_smoke_script.py`)
  - runs `scripts/runpod_cli_doctor.sh`
  - runs timeout-guarded connectivity+telemetry category via `scripts/cloud_connectivity_health_checks.sh --provider runpod` (default enabled, can disable with `RUN_CONNECTIVITY_TELEMETRY_CHECKS=0`)
  - provider can be overridden with `RUN_CONNECTIVITY_PROVIDER` for cross-cloud architecture parity
  - optionally runs direct `template-list`/`gpu-search` probes when `RUNPOD_API_KEY` is set
  - optionally runs Docker local smoke via `RUN_LOCAL_SMOKE=1`
- `scripts/cloud_connectivity_health_checks.sh`
  - provider-based connectivity/health/telemetry framework with timeout-guarded checks
  - RunPod provider implementation lives in `scripts/cloud_checks/providers/runpod.sh`
- `scripts/runpod_connectivity_telemetry_checks.sh`
  - backward-compatible wrapper to `cloud_connectivity_health_checks.sh --provider runpod`
  - maps legacy env names (`RUNPOD_CONNECTIVITY_TIMEOUT_SECONDS`, `RUNPOD_ENABLE_LIVE_CONNECTIVITY_CHECKS`) to framework envs
- `scripts/runpod_quick_launch.sh`
  - thin wrapper around `runpod_provision.py provision`
  - driven by env vars (`RUNPOD_TEMPLATE_NAME`, `RUNPOD_POD_NAME`, `RUNPOD_GPU_TYPE_ID`, etc.)
  - supports quick launch from the host terminal with optional explicit GPU type ID
  - inherits current `runpod_provision.py` defaults (notably `--use-runpod-training-preset-env` is now opt-in)
- `scripts/runpod_cycle_common.sh`
  - shared helpers for modular lifecycle scripts (run id paths, keyring token lookup, pod JSON parsing, SSH/connection fields)
  - managed temp key only: scripts always use `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}` and do not support personal/local key override variables
  - shared SSH args disable host agent/keyring import prompts for managed keys (`AddKeysToAgent=no`, `IdentityAgent=none`)
  - defines tracked pod registry path helper (`config/runpod_tracked_pods.jsonl` by default)
- `scripts/runpod_cycle_start.sh`
  - provisions a pod from template using keyring-backed RunPod auth
  - now injects a managed no-passphrase temp SSH key by default (`AUTHORIZED_KEYS`, `PUBLIC_KEY`), generated at `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}`
  - managed key injection can be controlled with `RUNPOD_INJECT_MANAGED_SSH_KEY_ENV`
  - now injects a unique per-run `REPO_DIR` by default (`/workspace/chess-bot-<run_id>`) to avoid stale/root-owned repo directories on reused persistent volumes
  - unique `REPO_DIR` injection can be controlled with `RUNPOD_SET_UNIQUE_REPO_DIR` and `RUNPOD_DEFAULT_REMOTE_REPO_DIR`
  - now injects smoke-safe service env defaults by default (`START_SSHD=1`, `START_JUPYTER=0`, `START_INFERENCE_API=0`, `START_HF_WATCH=0`, `START_IDLE_WATCHDOG=0`) to keep `sshd` stable during lifecycle smoke tests
  - smoke-safe service defaults also set `START_OTEL_COLLECTOR=0` by default to avoid collector startup failures cascading into `sshd` teardown under entrypoint `wait -n` supervision
  - smoke-service default injection can be controlled with `RUNPOD_SET_SMOKE_SERVICE_ENVS`
  - now verifies direct SSH login readiness before returning success (`RUNPOD_REQUIRE_SSH_READY=1` default)
  - SSH readiness wait controls:
    - `RUNPOD_SSH_READY_TIMEOUT_SECONDS` (default `240`)
    - `RUNPOD_SSH_READY_POLL_SECONDS` (default `8`)
  - on SSH readiness timeout, start script can auto-terminate the just-created pod to avoid stalled-cost leaks (`RUNPOD_TERMINATE_ON_SSH_NOT_READY=1` default)
  - saves provisioning JSON to `artifacts/runpod_cycles/<run_id>/provision.json`
  - initializes a per-run markdown report under `artifacts/runpod_cycles/<run_id>/reports/observations.md`
  - appends a `RUNNING` record to the tracked pod registry (`config/runpod_tracked_pods.jsonl`)
- `scripts/runpod_cycle_push_dataset.sh`
  - pushes a valid local dataset directory (`train.jsonl`, `val.jsonl`) to the pod via `rsync`/SSH
  - defaults local dataset source to `data/dataset/_smoke_runpod`
  - waits for remote `REPO_DIR` to exist and become writable before `mkdir`/`rsync` (avoids startup race with repo clone/chown)
  - readiness wait is controlled with `RUNPOD_REMOTE_READY_TIMEOUT_SECONDS` / `RUNPOD_REMOTE_READY_POLL_SECONDS`
- `scripts/hf_dataset_publish.py`
  - publishes a validated dataset directory to a Hugging Face **dataset** repo using path versioning under `validated_datasets/<dataset_name>/<version>/`
  - canonical HF keyring identities:
    - read/fetch: `service=huggingface`, `username=codex_hf_read_token`
    - write/publish: `service=huggingface`, `username=codex_hf_write_token` (do not use in cloud training pods)
  - equivalent explicit CLI flags:
    - publish: `--keyring-service huggingface --keyring-username codex_hf_write_token`
    - fetch: `--keyring-service huggingface --keyring-username codex_hf_read_token`
  - default token lookup order:
    1. `--token`
    2. env (`HF_WRITE_TOKEN` for publish, `HF_READ_TOKEN` for fetch, legacy `HF_TOKEN` fallback)
    3. keyring (`codex_hf_write_token` for publish, `codex_hf_read_token` for fetch)
    4. dotenv fallback (`HF_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.hf_dataset`, then `.env`)
  - writes `manifest.json` + `checksums.sha256` and uploads either:
    - a compressed `*.tar.gz` archive (default, faster for JSONL uploads), or
    - raw copied files (`--archive-format none`)
  - HF web UI showing only `manifest.json`, `checksums.sha256`, and the `*.tar.gz` payload inside a version folder is expected for the default archive mode (the actual `train.jsonl` / `val.jsonl` / `stats.json` are inside the archive)
  - supports `--dry-run` to validate repo path/versioning without network upload
- `scripts/hf_dataset_fetch.py`
  - fetches a versioned dataset package from a Hugging Face dataset repo and extracts the uploaded archive by default
  - supports `--all-latest` to fetch the latest version for every dataset under the repo prefix and emit an aggregate manifest of extracted `train.jsonl` / `val.jsonl` paths plus `aggregate_by_format` buckets (when manifests expose `dataset_format`)
  - supports `--dry-run` to print the planned repo path and snapshot patterns without downloading
- `scripts/runpod_cycle_train.sh`
  - runs `train_baseline_preset.sh` on the pod with explicit output/metrics paths under `artifacts/runpod_cycles/<run_id>/`
  - prefers repo copy of `deploy/runpod_cloud_training/train_baseline_preset.sh` over image-baked `/opt/...` copy (fallback only) so long-lived pods/old images still use latest pulled training script
  - runs a short remote inference smoke command (`scripts/infer_move.py`) against the produced model
  - waits for remote repo scripts/readiness before launching training (same timeout/poll env controls as dataset push)
  - exports `REPO_DIR`, `RUNPOD_PHASE_TIMING_LOG`, and `PYTHONPATH` inside the remote SSH command so per-run `REPO_DIR` overrides work end-to-end
  - supports HF-backed aggregate training mode via `RUNPOD_TRAIN_FROM_HF_LATEST_ALL=1` plus `RUNPOD_HF_DATASET_REPO_ID`:
    - pod fetches the latest version of every dataset from the HF dataset repo prefix (default `validated_datasets`)
    - `runpod_cycle_push_dataset.sh` can be skipped for this mode
    - remote fetch summary manifest is written under `${REPO_DIR}/artifacts/hf_dataset_fetch_manifest.json`
    - exports `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1` so training fails on runtime-cache miss/mismatch instead of runtime index fallback
- `scripts/runpod_cycle_collect.sh`
  - pulls remote run artifacts and timing logs into local `artifacts/runpod_cycles/<run_id>/collected/`
  - supports fast-collect mode via `RUNPOD_COLLECT_INCLUDE_EPOCH_CHECKPOINTS=0` (default include is `1`) to skip heavy `epoch_checkpoints/**` trees during rsync
  - automatically writes a best-effort runtime log bundle under `.../collected/logs_auto/`:
    - `remote_state_snapshot.txt` (GPU/process/service + train/progress log tails)
    - `train_log_indexing_summary.json` (keyword-based indexing/cache-miss detection across collected logs)
    - `collection_manifest.json` (deterministic file inventory for the collected directory)
- `scripts/runpod_cycle_local_validate.sh`
  - runs local CPU inference (`scripts/infer_move.py`) on the collected `.pt` artifact and saves output
  - exports local `PYTHONPATH=${REPO_ROOT}` before invoking `scripts/infer_move.py` so direct script execution resolves `src.*` imports reliably on the host
- `scripts/runpod_cycle_stop.sh`
  - cleanly requests pod stop via RunPod GraphQL `podStop` using RunPod token from env `RUNPOD_API_KEY` first, then keyring fallback (`runpod` / `RUNPOD_API_KEY`), then dotenv fallback (`RUNPOD_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, `.env.runpod`, `.env`), plus saved `pod_id`
  - stop mutation payload now requests object subfields (`id`, `desiredStatus`) to match the current GraphQL schema and avoid validation failures
  - appends a `STOPPED` record to the tracked pod registry for operator bookkeeping
  - note: `stop` halts compute but does not delete the pod; storage charges can still apply until termination
- `scripts/runpod_cycle_terminate.sh`
  - terminates the current cycle pod via RunPod REST `DELETE /pods/<id>` using the saved `pod_id` from `provision.json` (or env override)
  - writes `artifacts/runpod_cycles/<run_id>/terminate_response.json` with HTTP status + response body
  - appends a `TERMINATED` record to the tracked pod registry on success (and treats `404 pod not found` as already-terminated reconciliation)
- `scripts/runpod_cycle_terminate_all_tracked.sh`
  - safe cleanup utility to terminate (delete) all pods tracked by our scripts whose latest tracked state is not `TERMINATED`
  - requires explicit confirmation (`--yes` or `RUNPOD_CONFIRM_TERMINATE_ALL=YES`)
  - uses RunPod REST `DELETE /pods/<id>` with RunPod token from env `RUNPOD_API_KEY` first, then keyring fallback (`runpod` / `RUNPOD_API_KEY`), then dotenv fallback (`RUNPOD_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, `.env.runpod`, `.env`), and appends `TERMINATED` registry records on success
  - treats RunPod REST `404` responses that clearly indicate `pod not found to terminate` as an "already gone" reconciliation case (records local `TERMINATED` instead of failing the whole cleanup)
- `scripts/runpod_cycle_full_smoke.sh`
  - orchestration wrapper that composes the modular scripts in order (start -> push -> train -> collect -> local-validate -> stop)
- `scripts/runpod_cycle_watch_progress.sh`
  - keeps a single SSH session open to stream remote JSONL progress snapshots and renders a local epoch progress bar using machine-readable events from `scripts/train_baseline.py --progress-jsonl-out`
  - watches a remote `train_exit_code.txt` sentinel and exits with the same code
  - on `epoch_end`, attempts to pull remote best/epoch checkpoints into local `artifacts/runpod_cycles/<run_id>/live_checkpoints/`
  - appends per-epoch ETA/data snapshots to `artifacts/runpod_cycles/<run_id>/reports/epoch_eta_report_<run_id>.jsonl`
  - optional override: set `RUNPOD_LOCAL_SYNC_DIR=<preconfigured_local_dir>` to sync checkpoints/reports under `<preconfigured_local_dir>/<run_id>/...`
  - scans snapshot tails for the latest valid JSON progress event (avoids getting stuck when the newest tailed line is partial/non-JSON)
  - supports manual PTY allocation via `RUNPOD_SSH_FORCE_TTY=1` when a host environment requires it
  - handles `Ctrl-C`/`SIGTERM` by restoring TTY state and stopping local child watcher processes to reduce terminal corruption/noisy leftover streams
- `scripts/runpod_cycle_watchdog.sh`
  - host-side stall watchdog for active runs (`RUNPOD_CYCLE_RUN_ID`)
  - this script is now a backward-compatible alias to `scripts/telemetry_watchdog.sh`
  - polls `scripts/runpod_cycle_status.sh --no-write` and treats stage/progress/log growth as activity
  - configurable stall action (`--on-stall`) with options:
    - `none`, `collect`, `stop`, `terminate`, `collect-stop`, `collect-terminate`
  - optional auto-collect on normal completion (`RUNPOD_WATCHDOG_AUTO_COLLECT_ON_FINISH=1`)
- `scripts/telemetry_control.sh`
  - central CLI entrypoint for telemetry commands (`status`, `event`, `checkpoint`, `watchdog`, `health`)
- `scripts/telemetry_status.sh`
  - single-run telemetry snapshot that merges latest event/checkpoint rows with `runpod_cycle_status.sh --no-write`
- `scripts/runpod_cycle_full_train_hf.sh`
  - sequential host-side orchestration for a full HF-backed training run: GPU selection (`gpu-search` with fallback), start pod, remote HF fetch, remote context probe/spec suggestions, async remote training, local progress watch, collect artifacts, stop pod
  - runs a remote GPU sampler during training and writes per-run pre-train and post-run GPU/dataset/parameter observation artifacts for tuning future runs
  - writes a quick local play-test command (`.venv/bin/python main.py --model <collected model>`) into the run directory after collection
  - forwards `RUNPOD_HF_DATASET_SCHEMA_FILTER` to the remote preset and direct fallback path so compact game datasets (`game_jsonl_runtime_splice_v1`) can be selected explicitly from mixed HF repos
  - supports single-dataset remote HF fetch for smoke/targeted runs via `RUNPOD_HF_DATASET_NAME` and optional `RUNPOD_HF_DATASET_VERSION` (otherwise defaults to `--all-latest` under `RUNPOD_HF_DATASET_PATH_PREFIX`)
  - forwards runtime-splice smoke throttles (`RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT`, `RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET`, `RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME`) into the remote preset/direct fallback for compact `*_game` dataset smoke runs
  - enforces runtime splice cache usage in remote training (`TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1` / `--require-runtime-splice-cache`); cache miss/mismatch now fails the run instead of falling back to runtime index build
  - remote training launcher now prefers the repo copy of `deploy/runpod_cloud_training/train_baseline_preset.sh` over the image-baked `/opt/...` copy to avoid stale image-script behavior
  - if the selected preset lacks HF aggregate support, wrapper falls back to a direct `scripts/train_baseline.py` invocation using paths from the already-fetched HF manifest
  - local quick-play command/model retrieval now uses robust collected-artifact lookup (prefers `model_<run_id>.pt`, falls back to latest `.pt`) to tolerate naming variations while preserving run-id preference
  - default remote `num_workers` now uses a safer capped auto policy unless `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` is set:
    - policy now computes per-rank workers (the value passed to `--num-workers` under torchrun):
    - `cpu_worker_budget_total = max(nproc - TRAIN_NPROC_PER_NODE - RUNPOD_FULL_TRAIN_CPU_RESERVE_THREADS, 0)` (reserve default `0`)
    - `cpu_based_per_rank = floor(cpu_worker_budget_total / TRAIN_NPROC_PER_NODE)`
    - `ddp_suggested_per_rank = vram_suggested_num_workers`
    - `hard_cap_per_rank = floor(RUNPOD_FULL_TRAIN_NUM_WORKERS_HARD_CAP / TRAIN_NPROC_PER_NODE)` (default total cap `32`)
    - `auto_num_workers_per_rank = min(cpu_based_per_rank, ddp_suggested_per_rank, hard_cap_per_rank)`
    - effective total workers = `auto_num_workers_per_rank * TRAIN_NPROC_PER_NODE`
  - supports subset caps for fast-stage runs:
    - `RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS`
    - `RUNPOD_FULL_TRAIN_MAX_TRAIN_ROWS`
    - `RUNPOD_FULL_TRAIN_MAX_VAL_ROWS`
  - forwards best/epoch checkpoint paths so trainer writes intermediate model artifacts during long runs
- `scripts/runpod_cycle_status.sh`
  - single-command status snapshot for an active run id
  - reports local watcher/full-flow process presence plus remote stage (`repo_not_ready`, `run_dir_not_created`, `bootstrap_or_fetching`, `hf_fetch_done`, `pretrain_ready_or_launching`, `training_running`, `training_finished`, `manual_training_or_artifacts_present`, `manual_training_finished`, `ssh_unreachable`)
  - includes remote sentinel/file counts (`train_pid`, `progress_lines`, `train_log_lines`, `train_exit_code`, checkpoint presence) and one-line GPU sample when available
  - includes manual sub-run discovery metadata under `remote.manual_runs` and `remote.manual_latest` for `manual_*` training directories
  - supports `--watch` polling mode for continuous JSON snapshots
  - supports `--auto-collect` to trigger `scripts/runpod_cycle_collect.sh` once when a detected `manual_*` run reaches completed exit code (uses local marker files to avoid repeated collects)
  - traps `Ctrl-C`/`SIGTERM`, stops local child processes, restores terminal state, and exits `130` before running best-effort pod-stop cleanup
  - operational caveat: if the local watcher step fails after remote training has started/completed, the wrapper's error trap can stop the pod before `collect`; this does not mutate/delete the source HF dataset repo, and remote run artifacts typically remain on the pod volume until the pod is terminated (restart the same pod and run `scripts/runpod_cycle_collect.sh` for the same `RUNPOD_CYCLE_RUN_ID`)
- `scripts/runpod_cycle_report_style.py`
  - generates a concise operator-facing markdown/json progress report for one run id
  - report includes epoch progress, ETA, latest/peak GPU usage samples, DDP/cache fields, and local artifact presence
  - default markdown output path: `artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md`
- `scripts/runpod_cycle_summarize_gpu_observations.py`
  - aggregates `gpu_full_training_observation_*.json` artifacts across runs, groups by GPU SKU, and emits heuristic next-run override suggestions (batch size / workers)
  - supports JSON and Markdown summary outputs for operator notes/spec updates
- `scripts/runpod_full_train_easy.sh`
  - one-command opinionated wrapper around `runpod_cycle_full_train_hf.sh`
  - uses the shared RunPod managed-key path from `runpod_cycle_common.sh`: temporary no-passphrase SSH key under `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}` with private key mode `0600`
  - injects that managed key for provisioning and SSH commands so host passphrase prompts are avoided by default
  - defaults to multi-GPU provisioning and launch:
    - `RUNPOD_GPU_COUNT=2`
    - `RUNPOD_FULL_TRAIN_NPROC_PER_NODE=${RUNPOD_GPU_COUNT}` (effective remote train process count)
  - single-GPU compatibility remains via explicit env override (`RUNPOD_GPU_COUNT=1`, `RUNPOD_FULL_TRAIN_NPROC_PER_NODE=1`)
  - logs effective HF prefix/schema filter and runtime-splice smoke throttle overrides when provided
  - prints the canonical easy-style report destination (`artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md`)
- `scripts/runpod_full_train_easy_smoke_test.sh`
  - end-to-end smoke wrapper around the easy HF flow using cheap/fast defaults (single epoch, compact month prefix, runtime-splice sample cap)
  - verifies local collected artifacts with `scripts/runpod_cycle_verify_full_hf_run.py`
  - terminates the pod after verification and re-verifies termination markers
- `scripts/runpod_cycle_benchmark_matrix.sh`
  - one-pod matrix runner for repeated training trials across precision modes on the same provisioned pod
  - defaults to `RUNPOD_GPU_TYPE_ID=NVIDIA A40`, `RUNPOD_GPU_COUNT=2`, and trial list `fp32,tf32,fp16,bf16,sparsity`
  - syncs remote repo to host `HEAD` before dataset fetch/trials and fails fast on SHA mismatch (prevents stale remote code during long-lived template/image reuse)
  - performs a one-time remote HF fetch before trials and reuses the generated manifest across all trials
  - optional single-dataset mode for faster benchmark loops:
    - `RUNPOD_BENCH_HF_DATASET_NAME=<dataset_name>`
    - `RUNPOD_BENCH_HF_DATASET_VERSION=<version|latest>`
    - `RUNPOD_HF_DATASET_PATH_PREFIX` remains the repo prefix (default `validated_datasets`)
  - supports sparse variants (`fp32_sparse`, `fp16_sparse`, `bf16_sparse`) by passing trainer sparsity flags (`--sparsity-mode l1 --sparsity-l1-lambda ...`)
  - supports sparse lambda sweeps in one run via `RUNPOD_BENCH_SPARSITY_L1_LAMBDAS` (comma-separated); sparse base trials are expanded into trial ids like `<base>@<lambda>` and each variant is executed independently
  - benchmark batch sizing now supports auto mode (`RUNPOD_BENCH_BATCH_SIZE=auto` default path):
    - resolves an aggressive per-rank batch target from remote GPU VRAM (`8192/4096/2048/1024/512` tiers)
    - builds a descending retry plan and reruns the same trial at lower batches when OOM signatures are detected in train logs
    - explicit numeric `RUNPOD_BENCH_BATCH_SIZE` remains a hard override (no auto downgrade plan)
  - supports distributed backend override for trial debugging/compatibility: `RUNPOD_BENCH_DISTRIBUTED_BACKEND=<nccl|gloo>`
  - benchmark trial artifact pull defaults to a fast filtered copy (metrics/progress/logs/models/telemetry), excluding `epoch_checkpoints/**` unless explicitly enabled with `RUNPOD_BENCH_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS=1`
  - transfer controls:
    - `RUNPOD_BENCH_TRANSFER_TOOL` (`rclone` default)
    - `RUNPOD_BENCH_TRANSFER_RETRIES` (default `3`)
    - `RUNPOD_BENCH_TRANSFER_TIMEOUT_SECONDS` (default `1800`)
    - `RUNPOD_BENCH_RCLONE_TRANSFERS`/`RUNPOD_BENCH_RCLONE_CHECKERS`/`RUNPOD_BENCH_RCLONE_MULTI_THREAD_STREAMS`
  - `rclone` SFTP is preferred with automatic fallback to filtered `rsync`; set `RUNPOD_BENCH_TRANSFER_STRICT=1` to fail fast when `rclone` is requested but unavailable
  - final full collect can be tuned independently:
    - `RUNPOD_BENCH_COLLECT_INCLUDE_EPOCH_CHECKPOINTS` (default `0`)
    - `RUNPOD_BENCH_SKIP_FINAL_COLLECT=1` to skip end-of-matrix full collect entirely
  - benchmark/fetch path enforces cloud token safety by clearing write-capable HF env vars on pod-side fetch/train shells (`HF_WRITE_TOKEN`, legacy `HF_TOKEN`)
  - records `benchmark_image_used` telemetry event from provision metadata and writes image summary into trial markdown
  - runs a remote dependency freshness check against repo `requirements.txt` using pod venv, writes `artifacts/runpod_cycles/<run_id>/reports/dependency_check.json`, and emits `benchmark_dependencies` telemetry status
  - supports one-time remote dataset manifest preparation per run before trials, so trial loops reuse the same fetched dataset selection
  - uses remote `train_baseline_preset.sh` with per-trial overrides (`--no-amp/--amp`, `--tf32`, `--amp-dtype`) and stores outputs under `artifacts/runpod_cycles/<run_id>/manual_bench/<trial>/`
  - propagates subset cap via both env (`TRAIN_MAX_TOTAL_ROWS`) and CLI (`--max-total-rows`) for deterministic logging and behavior checks
  - pulls each trial directory locally to `artifacts/runpod_cycles/<run_id>/benchmarks/<trial>/` and writes:
    - `artifacts/runpod_cycles/<run_id>/benchmarks/trial_summary.jsonl`
    - `artifacts/runpod_cycles/<run_id>/benchmarks/trial_summary.md`
  - trial summary now includes extracted speed/loss/top-k fields from local metrics/progress artifacts for quick comparison
  - supports graceful cleanup mode toggle:
    - `RUNPOD_BENCH_STOP_POD=1` to stop compute
    - `RUNPOD_BENCH_TERMINATE_POD=1` to terminate/delete pod after collection
  - emits benchmark/trial telemetry checkpoints and events under `artifacts/runpod_cycles/<run_id>/telemetry/`
- `scripts/runpod_cycle_benchmark_10k_sixpack.sh`
  - opinionated wrapper for quick throughput comparisons on one pod
  - defaults to 6 base trials (`fp32,fp16,bf16,fp32_sparse,fp16_sparse,bf16_sparse`) with sparse lambda sweep `RUNPOD_BENCH_SPARSITY_L1_LAMBDAS=1e-5,5e-5,1e-4`
  - defaults benchmark batch mode to auto (`RUNPOD_BENCH_BATCH_SIZE=auto`) so the matrix can push higher VRAM-efficient batches per GPU SKU with OOM fallback
  - defaults to `RUNPOD_BENCH_MAX_TOTAL_ROWS=100000` and `RUNPOD_BENCH_EPOCHS=10` for stronger precision/sparsity differentiation
  - defaults to single-dataset fetch mode for speed (`RUNPOD_BENCH_HF_DATASET_NAME=elite_2025-11_game`, `RUNPOD_HF_DATASET_PATH_PREFIX=validated_datasets`)
  - defaults to pod termination (`RUNPOD_BENCH_TERMINATE_POD=1`) after artifact collection and writes final telemetry snapshot JSON
- `scripts/build_runtime_splice_vocab_meta.py`
  - local utility to build `runtime_splice_cache/vocab_rows_meta.json` for compact game datasets using multiprocessing
  - metadata includes train vocab and split game/sample row counts keyed by runtime splice config, enabling cache-backed training startup without full JSONL recount/vocab scans when config matches
- `scripts/runpod_file_transfer.sh`
  - host-side file transfer utility for RunPod pods using managed SSH metadata from `provision.json`
  - modes:
    - `pull <remote_path> <local_path>`
    - `push <local_path> <remote_path>`
    - `sync <local_dir> <remote_dir>` (rsync with `--delete`)
  - transfer hardening:
    - retry loop (`RUNPOD_TRANSFER_RETRIES`, `RUNPOD_TRANSFER_RETRY_SLEEP_SECONDS`)
    - resumable mode (`--partial --append-verify`) enabled by default
    - optional checksum mode (`RUNPOD_TRANSFER_CHECKSUM=1`) and bandwidth cap (`RUNPOD_TRANSFER_BWLIMIT_KBPS`)
  - writes transfer logs under `artifacts/runpod_cycles/<run_id>/logs/`
  - emits transfer start/complete telemetry events under `artifacts/runpod_cycles/<run_id>/telemetry/events.jsonl`
- `scripts/runpod_cycle_verify_full_hf_run.py` / `src/chessbot/runpod_cycle_verify.py`
  - verifies full-HF cycle local artifacts (`provision.json`, `stop_response.json`, collected model/metrics/logs/progress/GPU samples/HF fetch manifest/context probe)
  - optional `--require-terminated` also checks `terminate_response.json`

## Tracked Pod Registry
- Default file: `config/runpod_tracked_pods.jsonl`
- Purpose: keep a local script-owned ledger of pods launched/stopped/terminated via the repo RunPod lifecycle scripts
- Format: JSON Lines (one event per line, append-only)
- Override path with `RUNPOD_TRACKED_PODS_FILE`
- Typical states written by scripts:
  - `RUNNING` (after `runpod_cycle_start.sh`)
  - `STOPPED` (after `runpod_cycle_stop.sh`; pod still exists unless later terminated)
  - `TERMINATED` (after `runpod_cycle_terminate_all_tracked.sh`; pod resource deleted via REST)

## Running Pod Control (CLI)
- SSH into the pod using RunPod-provided connection info (port `22/tcp`)
- For direct mapped SSH on this image, use `runner@<public-ip>:<mapped-port>` (not `root@...`): `sshd` is configured with `PermitRootLogin no` and `AllowUsers runner`
- Operator support ergonomics for ad-hoc/manual recovery commands: when reconstructing commands from a prior run log (`public_ip`, mapped SSH port, temp key path), present the minimal working SSH command first (`ssh -i <key> -p <port> runner@<ip>`); include stricter host-key/known-hosts options only as an explicitly labeled optional variant
- SSH client security defaults for lifecycle scripts:
  - host key checking now defaults to `StrictHostKeyChecking=accept-new` (override with `RUNPOD_SSH_HOST_KEY_CHECKING=yes|no|accept-new`)
  - known hosts are persisted in `config/runpod_known_hosts` by default (override with `RUNPOD_SSH_KNOWN_HOSTS_FILE`)
  - this replaces the older `/tmp/runpod_known_hosts` + `StrictHostKeyChecking=no` behavior
- Jupyter is exposed on HTTP port `8888`
- Inference API is exposed on HTTP port `8000`
- Common operator commands after connect:
  - `bash "$REPO_DIR/deploy/runpod_cloud_training/train_baseline_preset.sh"` (preferred; repo copy can be newer than image-baked `/opt/...` script)
  - `curl http://127.0.0.1:8000/healthz`
  - `tail -f /workspace/chess-bot/artifacts/timings/runpod_phase_times.jsonl`
- Data/artifact transfer options:
  - `scp`
  - `rsync`
  - Jupyter file browser
  - `rclone` (installed in image)
  - Hugging Face dataset repo publish/fetch helpers (`scripts/hf_dataset_publish.py`, `scripts/hf_dataset_fetch.py`) for reusable validated datasets

## Modular Lifecycle Flow (host-side)
- Full cycle (defaults to a short smoke run and local `_smoke_runpod` dataset):
  - `bash scripts/runpod_cycle_full_smoke.sh`
  - ends with `stop` only (compute halt), not termination/deletion
  - current smoke defaults intentionally disable optional pod services (Jupyter/inference/HF-watchdog/idle-watchdog) unless explicitly overridden, to prevent `entrypoint.sh` `wait -n` child-exit cleanup from killing `sshd` during the CLI smoke
- Full sequential HF training cycle (100-epoch-oriented wrapper with local progress bar):
  - `export RUNPOD_HF_DATASET_REPO_ID='LogicLark-QuantumQuill/chess-bot-datasets'`
  - optional GPU/cost overrides: `RUNPOD_GPU_TYPE_ID`, `RUNPOD_GPU_MIN_MEMORY_GB`, `RUNPOD_GPU_MAX_HOURLY_PRICE`
  - `bash scripts/runpod_cycle_full_train_hf.sh`
  - wrapper currently fetches HF datasets first, writes remote context/spec-suggestion artifacts (dataset rows/size + GPU/VRAM snapshot + suggested params), runs async training with progress JSONL + GPU sampling, collects artifacts, and stops the pod
  - if the local progress watcher crashes near the end of training, check the remote `train_exit_code.txt` / train log under `${REPO_DIR}/artifacts/runpod_cycles/<run_id>/`, then restart the same pod and run `RUNPOD_CYCLE_RUN_ID=<run_id> bash scripts/runpod_cycle_collect.sh`
- One-command "just run it" wrapper (same full flow, opinionated defaults):
  - `bash scripts/runpod_full_train_easy.sh`
  - defaults:
    - HF repo `LogicLark-QuantumQuill/chess-bot-datasets`
    - `RUNPOD_FULL_TRAIN_EPOCHS=100`
    - community GPU with explicit default `NVIDIA GeForce RTX 5090`
    - `RUNPOD_GPU_COUNT=2` and `RUNPOD_FULL_TRAIN_NPROC_PER_NODE=2` (default multi-process train launch)
    - temporary no-passphrase SSH key under `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}`
    - progress output enabled by default (plus JSONL progress stream for watcher)
    - remote training `num_workers` defaults to a per-rank value from the policy above (effective total remains capped by CPU budget and `RUNPOD_FULL_TRAIN_NUM_WORKERS_HARD_CAP`, default `32`) when no override is set
  - operator can still override by env, but no flags/params are required for the default path
  - compact-dataset / runtime-splice-friendly env overrides (useful for smoke runs):
    - `RUNPOD_HF_DATASET_PATH_PREFIX` (for example `validated_datasets/elite_2025-11_game`)
    - `RUNPOD_HF_DATASET_NAME` (for example `elite_2025-11_game`, single-dataset fetch mode)
    - `RUNPOD_HF_DATASET_VERSION` (optional explicit version in single-dataset mode)
    - `RUNPOD_HF_DATASET_SCHEMA_FILTER` (for example `game_jsonl_runtime_splice_v1`)
    - `RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT`
    - `RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET`
    - `RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME`
  - helper smoke wrapper:
    - `bash scripts/runpod_full_train_easy_smoke_test.sh`
    - uses the same easy/full-HF path, then verifies local artifacts + stop response + termination response
    - defaults to a single compact month fetch (`RUNPOD_HF_DATASET_NAME=elite_2025-11_game`) instead of aggregate `--all-latest` to reduce smoke runtime/cost
  - supervised-run references:
    - monitoring snapshot loop: `bash scripts/runpod_cycle_status.sh --watch`
    - concise progress summary: `python scripts/runpod_cycle_report_style.py --run-id <run_id>`
    - canonical supervised checklist/acceptance criteria: `specs/chess_bot_runpod_preferred_training_flow.md` section `Supervised Cloud Training Reference Notes`
- For larger/reused validated datasets, prefer publishing once to a HF dataset repo and fetching into the pod/volume cache, instead of repeated host->pod `rsync` uploads
- Stepwise modular flow:
  1. `bash scripts/runpod_cycle_start.sh`
  2. `bash scripts/runpod_cycle_push_dataset.sh`
  3. `bash scripts/runpod_cycle_train.sh`
  4. `bash scripts/runpod_cycle_collect.sh`
  5. `bash scripts/runpod_cycle_local_validate.sh`
  6. `bash scripts/runpod_cycle_stop.sh`
  - HF aggregate training variant (skip dataset push):
    - `export RUNPOD_TRAIN_FROM_HF_LATEST_ALL=1`
    - `export RUNPOD_HF_DATASET_REPO_ID='LogicLark-QuantumQuill/chess-bot-datasets'`
    - optional `export RUNPOD_HF_DATASET_PATH_PREFIX='validated_datasets'`
    - run steps `1 -> 3 -> 4 -> 5 -> 6`
- Shared run identifier:
  - set `RUNPOD_CYCLE_RUN_ID=<custom-id>` to keep all files under one run directory
- Core outputs per run:
  - `artifacts/runpod_cycles/<run_id>/provision.json`
  - `artifacts/runpod_cycles/<run_id>/reports/observations.md`
  - `artifacts/runpod_cycles/<run_id>/logs/*` (persisted command/transfer logs for cycle steps)
  - `artifacts/runpod_cycles/<run_id>/collected/run_artifacts/*`
  - `artifacts/runpod_cycles/<run_id>/local_validation/infer_move_output.txt`
  - `artifacts/runpod_cycles/<run_id>/quick_play_command.txt` (written by `runpod_cycle_full_train_hf.sh`)
  - `artifacts/runpod_cycles/<run_id>/terminate_response.json` (when `runpod_cycle_terminate.sh` is used)
  - `artifacts/runpod_cycles/<run_id>/spec_suggestions/*.json|*.md` (post-run local summaries for GPU/dataset/VRAM/param tuning)
  - examples under `logs/`:
    - `push_dataset_ready_check.log`, `push_dataset_rsync.log`
    - `train_ready_check.log`, `train_remote_ssh.log`
    - `collect_rsync_run_artifacts.log`, `collect_rsync_timing.log`
    - `gpu_search.log`, `hf_fetch_remote.log`, `context_probe_remote.log`, `train_launch_remote.log`, `train_progress_watch.log` (sequential HF full-training flow)

## Sequential HF Full-Training Artifact Contract (new wrapper)
- Remote artifacts under `${REPO_DIR}/artifacts/runpod_cycles/<run_id>/` (collected into `artifacts/runpod_cycles/<run_id>/collected/run_artifacts/`):
  - `hf_dataset_fetch_manifest.json` (aggregate HF fetch manifest reused for training)
  - `context_probe_<run_id>.json` (pre-train dataset size/row counts + GPU/VRAM + torch runtime context)
  - `spec_suggestions_gpu_training_<run_id>.json|.md` (pre-train GPU-specific parameter suggestions for future spec tuning)
  - `train_progress_<run_id>.jsonl` (epoch-level machine-readable progress events)
  - `gpu_usage_samples_<run_id>.csv` (timestamped GPU util + memory samples collected during training)
  - `train_stdout_<run_id>.log`, `train_exit_code.txt`, `train_pid.txt`
  - `model_<run_id>.pt`, `metrics_<run_id>.json`
- Local post-run observation artifacts (generated after collect):
  - `artifacts/runpod_cycles/<run_id>/spec_suggestions/gpu_full_training_observation_<run_id>.json|.md`
  - summarize dataset size/rows, peak GPU memory/util observed, training epochs/metrics, and next-run tuning notes
- Cross-run follow-up summarization:
  - `python scripts/runpod_cycle_summarize_gpu_observations.py --output-json artifacts/reports/runpod_gpu_observation_summary.json --output-md artifacts/reports/runpod_gpu_observation_summary.md`
  - use the grouped GPU recommendations to set future `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE` / `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE`
- Intended operator workflow:
  - use pre-train `spec_suggestions_gpu_training_*` to choose initial batch size/workers for the run
  - use post-run `gpu_full_training_observation_*` to refine future per-GPU defaults (especially batch size vs peak VRAM)

## Observed Real Training Run Tuning (2026-02-26, RTX 6000 Ada)
- Pods successfully provisioned on `NVIDIA RTX 6000 Ada Generation` using the current template/image, but direct mapped SSH (`runner@<public-ip>:<mapped-port>`) repeatedly reset during key exchange across multiple fresh pods.
- During a real HF aggregate training run (`elite_2025-10_cap4` + `elite_2025-11_cap4`; ~1,511,216 train rows / 188,811 val rows):
  - `batch_size=2048`, `num_workers=6` showed low VRAM usage (~14 GB / 48 GB) with high-but-variable GPU utilization
  - rerun with `batch_size=8192`, `num_workers=12` increased VRAM usage to ~34 GB / 48 GB and sustained ~99% GPU utilization (better hardware saturation)
- Practical tuning implication (current observed-good starting point for this GPU + dataset scale):
  - prefer `TRAIN_BATCH_SIZE=8192`, `TRAIN_NUM_WORKERS=12` (or `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE=8192`, `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE=12`) and then refine from collected run telemetry

## Current Working Operator Path (2026-02-26)
- Most reliable manual flow on the observed host:
  - connect via direct mapped SSH (`runner@<public-ip>:<mapped-port>`)
  - start training inside the pod using the repo preset (`$REPO_DIR/deploy/runpod_cloud_training/train_baseline_preset.sh`)
  - run host-side watcher/collect/stop scripts with the same direct SSH mapping

## Provisioning Helper Defaults (2026-02-26 update)
- `provision --use-runpod-training-preset-env` is now **opt-in** (default disabled)
- Reason: implicit helper env injection could override template-configured service flags (for example re-enabling `START_IDLE_WATCHDOG=1`)
- Operators should enable it explicitly only when they want helper-provided service defaults layered on top of template config

## Observed Pod Restart Loop Debug (2026-02-26)
- Symptom from host CLI:
  - SSH and HTTP ports mapped but connections reset repeatedly
- Root cause confirmed via RunPod UI logs:
  - inference API process crashed on `ModuleNotFoundError: No module named 'torch'`
  - entrypoint then shut down all services, causing the pod to restart repeatedly
  - a bad/non-git `REPO_DIR` also prevented repo requirement sync (so `torch` never got installed)
- Operator workaround prior to image patch:
  - launch with a unique `REPO_DIR`
  - `--env START_INFERENCE_API=0`
  - `--env START_IDLE_WATCHDOG=0`
  - `--no-use-runpod-training-preset-env`
- Code fixes now implemented in `deploy/runpod_cloud_training/entrypoint.sh` and `scripts/runpod_provision.py`
- Detailed dated incident report (problems + solutions) saved at `artifacts/reports/runpod_cycle_observations_2026-02-26.md`

## Observed SSH Access Recovery (2026-02-26)
- Direct SSH to mapped pod port reached auth but failed with `Permission denied (publickey)` (template-side `AUTHORIZED_KEYS` issue suspected)
- Additional root cause found during follow-up debugging:
  - even with correct `/home/runner/.ssh/authorized_keys`, direct mapped `runner@<ip>:<port>` can still be denied when the `runner` account is locked by default (`useradd` on Ubuntu)
  - deployment flow now fixes this automatically by unlocking `runner` at image build/startup while keeping `PasswordAuthentication no`

## Observed CLI Smoke Blocker (2026-02-26 host run)
- Full `scripts/runpod_cycle_full_smoke.sh` retries were able to provision community pods (for example `RTX 4090`, `RTX 3090`) using explicit `RUNPOD_GPU_TYPE_ID` after transient REST capacity failures.
- At that point `gpu-search` appeared unavailable due GraphQL `403`, so explicit GPU IDs were required.
- Direct mapped SSH still failed with `Permission denied (publickey)` on newly provisioned pods even when the provision payload showed the expected public key in `AUTHORIZED_KEYS`/`PUBLIC_KEY`.
- Practical implication:
  - full CLI smoke could not progress past `runpod_cycle_push_dataset.sh` without a working SSH path
  - operator should use a template/image build with working direct SSH (`runner`) to continue modular cycle steps
  - after failed runs, issue `stop` first to halt compute, then use the tracked terminate script when you want deletion cleanup
- Additional API diagnostic from the same date:
  - `scripts/runpod_cli_doctor.sh` reported GraphQL denial, but direct `curl` GraphQL probes (`myself`, `gpuTypes`) succeeded with the same key
  - likely cause was Cloudflare/browser-signature filtering against the helper's Python `urllib` requests; helper code now adds an explicit `User-Agent` header and the doctor GraphQL probe returned `ok` after the fix

## Current GPU Availability Query (2026-02-26 host run, keyring-backed)
- `gpu-search` is working again with the current RunPod API key/keyring flow and should be used before ad-hoc relaunch retries.
- Exact command used:
  - ``.venv/bin/python scripts/runpod_provision.py --keyring-service runpod --keyring-username RUNPOD_API_KEY gpu-search --cloud-type COMMUNITY --limit 20``
- Practical use:
  - pick a cheap/community GPU from the live list (for example `RTX A5000`, `RTX A4000`, `RTX 3070`, `RTX 3080`) instead of retrying a stale hardcoded SKU
  - then set `RUNPOD_GPU_TYPE_ID` explicitly for `scripts/runpod_cycle_start.sh` or `scripts/runpod_full_train_easy_smoke_test.sh`
- Reminder:
  - the list is an API capability/pricing snapshot and does not guarantee capacity on the next provision call; transient `500`/`no instances` can still happen

## Wait-Ready Timing Caveat (2026-02-26 host run correction)
- `desiredStatus="RUNNING"` with empty `publicIp` / null `portMappings` is **not** sufficient evidence that provisioning is dead.
- We observed pods remain in this state for an extended period after rent while the wrapper was polling `--wait-ready`.
- Operational rule:
  - do **not** terminate early based only on a short sequence of `wait_status` logs with empty `publicIp`
  - use a longer readiness timeout/window and re-check the live pod JSON before concluding failure
- Practical implication for manual recovery/testing:
  - if a pod is rented and `RUNNING`, prefer waiting longer (or resuming later) before deleting it, unless there is a confirmed terminal error/state from the API

## SECURE vs COMMUNITY for SSH-Based Cycle Flows (2026-02-26)
- `scripts/runpod_provision.py provision --wait-ready` currently waits for `publicIp` to become non-empty.
- The helper also auto-sets `supportPublicIp=true` only for `COMMUNITY` (`--support-public-ip-auto` behavior).
- Consequence:
  - `SECURE` cloud pods launched via current helper defaults can remain `RUNNING` with empty `publicIp` indefinitely from the wrapper's perspective (not a pod failure, but incompatible with this readiness check + direct SSH workflow).
- For `scripts/runpod_cycle_*` and `scripts/runpod_full_train_easy*.sh` (direct SSH/rsync flows):
  - prefer `RUNPOD_CLOUD_TYPE=COMMUNITY`
  - or patch the provisioning/readiness logic before attempting `SECURE`

## Compact Dataset Smoke Bottleneck (2026-02-26)
- Full-easy smoke against `validated_datasets/elite_2025-11_game` (compact game dataset, full month) successfully reached:
  - pod ready
  - remote HF fetch (`elite_2025-11_game@latest`)
  - schema detection/filter (`game_jsonl_runtime_splice_v1`)
  - remote train launch
- Observed bottleneck:
  - remote `train_baseline.py` remained CPU-bound (~99%) for >8 minutes before any progress JSONL event beyond `script_start`
  - GPU remained idle during this period (`~0% util`, only a few MiB VRAM used)
- Interpretation:
  - runtime-splice startup/index/phase-cache build on the full monthly compact dataset is too heavy for a "fast smoke" run
- Practical smoke guidance (current):
  - do not use a full-month compact `*_game` dataset as the default smoke target
  - use a smaller compact dataset variant (recommended dedicated smoke publish) or add a remote subsetting step before training
- Evidence saved for run:
  - `artifacts/runpod_cycles/easy-smoke-retry-20260226T161214Z/reports/bottleneck_note.md`

## Observed Full Cycle Success (2026-02-26 host run, after fixes)
- Successful end-to-end RunPod smoke run completed on `runpod-cycle-20260226T065518Z` after applying the following fixes:
  - helper `User-Agent`/`Accept` headers for RunPod GraphQL requests (`urllib` Cloudflare `1010` false-negative fix)
  - template image updated to `ghcr.io/rareskey/chess-bot-runpod:ssh-unlock-fix-20260226T0634Z` (runner unlock fix)
  - cycle start default unique `REPO_DIR` injection (avoids root-owned stale repo path on reused volumes)
  - cycle start smoke-service env defaults (keeps only `sshd` enabled for lifecycle smoke stability)
  - dataset/train remote readiness waits
  - train/local-validate `PYTHONPATH` + `REPO_DIR` path fixes
- Pod/GPU details for the successful run:
  - pod id: `qtxtw8ui07v3zg`
  - gpu: `NVIDIA RTX A4000` community
  - image: `ghcr.io/rareskey/chess-bot-runpod:ssh-unlock-fix-20260226T0634Z`
- Successful phases:
  - `start` (provision + wait-ready)
  - `push_dataset`
  - `train` (remote train preset + remote inference smoke)
  - `collect`
  - `local_validate`
  - `stop`
- Local artifacts collected under:
  - `artifacts/runpod_cycles/runpod-cycle-20260226T065518Z/`

## Template Field Conventions (observed workflow)
- Pod template type: `Pod`
- Compute type: `Nvidia GPU`
- Container image: `ghcr.io/<lowercase-user>/chess-bot-runpod:<tag>`
- Ports:
  - `8888/http` (`jupyter`)
  - `8000/http` (`inference-api`)
  - `22/tcp` (`ssh`)
- Environment variables commonly set in template:
  - repo bootstrap: `REPO_URL`, `REPO_REF`, `REPO_DIR`, `CLONE_REPO_ON_START`, `GIT_AUTO_PULL`
  - runtime: `VENV_DIR`, `SYNC_REQUIREMENTS_ON_START`, `FORCE_PIP_SYNC`
  - services: `START_SSHD`, `START_JUPYTER`, `START_INFERENCE_API`, `START_HF_WATCH`, `START_IDLE_WATCHDOG`
  - SSH access: `AUTHORIZED_KEYS` (public key only)
  - timing logs: `RUNPOD_PHASE_TIMING_ENABLED`, `RUNPOD_PHASE_TIMING_LOG`

## RunPod GPU Types Snapshot (2026-02-26)
- Collected on host via direct RunPod GraphQL query using keyring-backed token (`service=runpod`, `username=RUNPOD_API_KEY`)
- Raw snapshot saved at `artifacts/reports/runpod_gpu_types_snapshot_2026-02-26.json`
- Dated config-style catalog saved at `config/runpod_gpu_types_catalog_2026-02-26.json` for reuse by RunPod CLI helper workflows/scripts
- Snapshot count: `44` GPU types
- Listed below as `id | displayName | memoryInGb` (pricing/availability fields are in the saved JSON and can change over time)

- `AMD Instinct MI300X OAM | MI300X | 192`
- `NVIDIA A100 80GB PCIe | A100 PCIe | 80`
- `NVIDIA A100-SXM4-80GB | A100 SXM | 80`
- `NVIDIA A30 | A30 | 24`
- `NVIDIA A40 | A40 | 48`
- `NVIDIA B200 | B200 | 180`
- `NVIDIA B300 SXM6 AC | B300 | 288`
- `NVIDIA GeForce RTX 3070 | RTX 3070 | 8`
- `NVIDIA GeForce RTX 3080 | RTX 3080 | 10`
- `NVIDIA GeForce RTX 3080 Ti | RTX 3080 Ti | 12`
- `NVIDIA GeForce RTX 3090 | RTX 3090 | 24`
- `NVIDIA GeForce RTX 3090 Ti | RTX 3090 Ti | 24`
- `NVIDIA GeForce RTX 4070 Ti | RTX 4070 Ti | 12`
- `NVIDIA GeForce RTX 4080 | RTX 4080 | 16`
- `NVIDIA GeForce RTX 4080 SUPER | RTX 4080 SUPER | 16`
- `NVIDIA GeForce RTX 4090 | RTX 4090 | 24`
- `NVIDIA GeForce RTX 5080 | RTX 5080 | 16`
- `NVIDIA GeForce RTX 5090 | RTX 5090 | 32`
- `NVIDIA H100 80GB HBM3 | H100 SXM | 80`
- `NVIDIA H100 NVL | H100 NVL | 94`
- `NVIDIA H100 PCIe | H100 PCIe | 80`
- `NVIDIA H200 | H200 SXM | 141`
- `NVIDIA H200 NVL | NVIDIA H200 NVL | 143`
- `NVIDIA L4 | L4 | 24`
- `NVIDIA L40 | L40 | 48`
- `NVIDIA L40S | L40S | 48`
- `NVIDIA RTX 2000 Ada Generation | RTX 2000 Ada | 16`
- `NVIDIA RTX 4000 Ada Generation | RTX 4000 Ada | 20`
- `NVIDIA RTX 4000 SFF Ada Generation | RTX 4000 Ada SFF | 20`
- `NVIDIA RTX 5000 Ada Generation | RTX 5000 Ada | 32`
- `NVIDIA RTX 6000 Ada Generation | RTX 6000 Ada | 48`
- `NVIDIA RTX A2000 | RTX A2000 | 6`
- `NVIDIA RTX A4000 | RTX A4000 | 16`
- `NVIDIA RTX A4500 | RTX A4500 | 20`
- `NVIDIA RTX A5000 | RTX A5000 | 24`
- `NVIDIA RTX A6000 | RTX A6000 | 48`
- `NVIDIA RTX PRO 4500 Blackwell | RTX PRO 4500 | 32`
- `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | RTX PRO 6000 MaxQ | 96`
- `NVIDIA RTX PRO 6000 Blackwell Server Edition | RTX PRO 6000 | 96`
- `NVIDIA RTX PRO 6000 Blackwell Workstation Edition | RTX PRO 6000 WK | 96`
- `Tesla V100-PCIE-16GB | Tesla V100 | 16`
- `Tesla V100-SXM2-16GB | V100 SXM2 | 16`
- `Tesla V100-SXM2-32GB | V100 SXM2 32GB | 32`
- `unknown | unknown | 0`

## RunPod GPU Availability Snapshot (2026-02-27, COMMUNITY)
- Command:
  - ``.venv/bin/python scripts/runpod_provision.py --keyring-service runpod --keyring-username RUNPOD_API_KEY gpu-search --cloud-type COMMUNITY --min-memory-gb 12 --limit 120``
- Raw/dated outputs saved:
  - `artifacts/reports/runpod_gpu_types_snapshot_2026-02-27.json`
  - `config/runpod_gpu_types_catalog_2026-02-27.json`
- Snapshot count: `40` GPU entries.
- Cheapest entries supporting at least 2 GPUs at snapshot time:
  - `NVIDIA RTX A5000` (`price_per_hr=0.16`, `max_gpu_count=10`, `memory_gb=24`)
  - `NVIDIA RTX A4000` (`price_per_hr=0.17`, `max_gpu_count=8`, `memory_gb=16`)
  - `NVIDIA GeForce RTX 3080 Ti` (`price_per_hr=0.18`, `max_gpu_count=6`, `memory_gb=12`)
- `NVIDIA GeForce RTX 5090` was present but higher cost (`price_per_hr=0.69`, `max_gpu_count=8`, `memory_gb=32`).

## Manual 2-GPU Custom-10k DDP Flow (Saved Runbook, 2026-02-27)
- Purpose: run a quick single-node multi-GPU validation with a manually pushed local custom dataset, without HF fetch/publish side effects.
- Successful reference run:
  - run id: `runpod-cycle-20260227T101419Z`
  - pod id: `3mwx5mfh2208mf`
  - gpu: `2 x NVIDIA RTX A4000` (community)
  - ssh endpoint at run time: `runner@87.197.146.56:40320`

1. Start a 2-GPU pod from the host (`COMMUNITY`, cheap SKU first):
   - `RUNPOD_GPU_COUNT=2 RUNPOD_GPU_TYPE_ID='NVIDIA RTX A5000' RUNPOD_CLOUD_TYPE=COMMUNITY RUNPOD_TEMPLATE_NAME='chess-bot-training' bash scripts/runpod_cycle_start.sh`
   - If capacity returns REST `500`/`no instances`, retry with next-cheapest 2-GPU type (for example `NVIDIA RTX A4000`).
2. Push manual local dataset (no HF train mode):
   - `RUNPOD_CYCLE_RUN_ID=<run_id> RUNPOD_LOCAL_DATASET_DIR=<repo>/data/dataset/_custom_test_10k_all_months_game RUNPOD_REMOTE_DATASET_NAME=_custom_test_10k_all_months_game bash scripts/runpod_cycle_push_dataset.sh`
3. Run explicit DDP training over 2 GPUs via `torchrun` on the pod:
   - From host, SSH to the pod and run:
   - `CUDA_VISIBLE_DEVICES=0,1 /opt/venvs/chessbot/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/train_baseline.py --train <remote_dataset>/train.jsonl --val <remote_dataset>/val.jsonl --output <remote_run_dir>/model_ddp_10k.pt --metrics-out <remote_run_dir>/metrics_ddp_10k.json --progress-jsonl-out <remote_run_dir>/progress_ddp_10k.jsonl --telemetry-dir <remote_run_dir>/telemetry --epochs 1 --batch-size 256 --num-workers 4 --distributed on --runtime-min-context 8 --runtime-min-target 1 --runtime-max-samples-per-game 0 --no-progress --max-total-rows 10000`
4. Pull artifacts locally:
   - `artifacts/runpod_cycles/<run_id>/collected/manual_ddp_10k/`
   - expected files:
     - `model_ddp_10k.pt`
     - `metrics_ddp_10k.json`
     - `progress_ddp_10k.jsonl`
     - `torchrun_train.log`
     - `train_exit_code.txt`
5. Validate DDP evidence:
   - `train_exit_code.txt` should be `0`
   - `metrics_ddp_10k.json` should include `distributed.enabled=true` and `distributed.world_size=2`
   - `torchrun_train.log` should show `train_setup.distributed.rank=0` and `rank=1` with device assignment split (`cuda:0`, `cuda:1`)
   - this run showed `train_rows=8889`, `val_rows=1111` and `subset_sampling.max_total_rows=10000`
