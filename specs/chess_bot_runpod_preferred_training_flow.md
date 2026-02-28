# Chess Bot RunPod Preferred Training Flow

## Responsibility
Define the current preferred end-to-end RunPod training flow, including exact operator steps and run parameters we want to keep stable across iterations.

## Scope
- Host-side orchestration script: `scripts/runpod_full_train_easy.sh`
- Underlying lifecycle: `scripts/runpod_cycle_full_train_hf.sh` and `scripts/runpod_cycle_*.sh`
- Dataset source: Hugging Face dataset repo fetch inside pod (no host->pod dataset push)
- Progress/watchdog: `scripts/runpod_cycle_watch_progress.sh` and remote training sentinels
  - watcher now also syncs remote best/epoch checkpoints locally at epoch boundaries and appends epoch ETA report JSONL
  - full-flow now writes an easy-style summary report at run end via `scripts/runpod_cycle_report_style.py`:
    - `artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md`
    - `artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.json`

## Current Preferred Defaults (2026-02-27)
- Include all published monthly datasets:
  - `RUNPOD_HF_DATASET_NAME` unset
  - `RUNPOD_HF_DATASET_VERSION` unset
  - use aggregate fetch (`--all-latest`) under prefix `validated_datasets`
- Dataset schema:
  - `RUNPOD_HF_DATASET_SCHEMA_FILTER=game_jsonl_runtime_splice_v1`
- Runtime splice cache policy:
  - `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1` enforced by flow (fail on cache miss/mismatch; no runtime index build fallback)
- Workers/batch sizing:
  - no override envs set (`RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` unset, `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE` unset)
  - remote auto logic computes per-rank workers, then passes that per-rank value to training:
    - `cpu_worker_budget_total = max(nproc - TRAIN_NPROC_PER_NODE - RUNPOD_FULL_TRAIN_CPU_RESERVE_THREADS, 0)` (reserve default `0`)
    - `cpu_based_per_rank = floor(cpu_worker_budget_total / TRAIN_NPROC_PER_NODE)`
    - `ddp_suggested_per_rank = vram_suggested_num_workers`
    - `hard_cap_per_rank = floor(RUNPOD_FULL_TRAIN_NUM_WORKERS_HARD_CAP / TRAIN_NPROC_PER_NODE)` (default hard cap `32`, interpreted as total cap)
    - `num_workers_per_rank = min(cpu_based_per_rank, ddp_suggested_per_rank, hard_cap_per_rank)`
  - batch size chosen by remote VRAM tier heuristic
  - on RTX 5090 without explicit batch override, flow now uses an auto retry ladder (`4096 -> 3072 -> 2048`) and falls back on OOM
- Fast-stage subset option:
  - `RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS` can cap effective train+val rows after indexing/cache load (auto split by original train/val ratio)
  - optional explicit split caps: `RUNPOD_FULL_TRAIN_MAX_TRAIN_ROWS`, `RUNPOD_FULL_TRAIN_MAX_VAL_ROWS`
- GPU default:
  - `RUNPOD_GPU_TYPE_ID` default in easy flow is `NVIDIA GeForce RTX 5090`
  - `RUNPOD_GPU_COUNT` default in easy flow is `2`
  - `RUNPOD_FULL_TRAIN_NPROC_PER_NODE` defaults to `${RUNPOD_GPU_COUNT}` in easy flow (multi-process train launch)
  - set `RUNPOD_GPU_COUNT=1` and/or `RUNPOD_FULL_TRAIN_NPROC_PER_NODE=1` to keep single-GPU single-process behavior
- SSH handling:
  - managed temp key auto-generated on host at `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}`
  - only the managed temp keypair path is used by runpod cycle scripts

## Exact Training Parameter Profile (Current)
The flow runs `train_baseline.py` via preset (or direct fallback when needed) with this profile:
- `epochs=20`
- `lr=2e-4`
- `embed_dim=256`
- `hidden_dim=512`
- `num_layers=2`
- `dropout=0.15`
- `amp` enabled
- `phase_feature` enabled
- `side_to_move_feature` enabled
- `runtime_min_context=8`
- `runtime_min_target=1`
- `runtime_max_samples_per_game=0`
- LR scheduler: `plateau` on `val_loss` with `factor=0.5`, `patience=3`, `threshold=1e-4`
- `early_stopping_patience=0`
- progress output enabled (machine-readable JSONL progress stream enabled for watcher)

## Start-To-Finish Operator Steps
1. Validate host tooling and RunPod auth:
```bash
bash scripts/runpod_cli_doctor.sh
```

2. Optional cleanup of stale locally tracked pods:
```bash
RUNPOD_CONFIRM_TERMINATE_ALL=YES \
bash scripts/runpod_cycle_terminate_all_tracked.sh --yes
```

3. Launch preferred full training flow:
```bash
export RUNPOD_HF_DATASET_REPO_ID="LogicLark-QuantumQuill/chess-bot-datasets"
export RUNPOD_HF_DATASET_PATH_PREFIX="validated_datasets"
export RUNPOD_HF_DATASET_SCHEMA_FILTER="game_jsonl_runtime_splice_v1"
export RUNPOD_FULL_TRAIN_EPOCHS="20"
# Optional: preconfigured local sync root for epoch checkpoints + ETA reports
# export RUNPOD_LOCAL_SYNC_DIR="/absolute/path/for/runpod_sync"

# Leave these unset for auto behavior:
unset RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE
unset RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE
# Optional single-GPU override:
# export RUNPOD_GPU_COUNT="1"
# export RUNPOD_FULL_TRAIN_NPROC_PER_NODE="1"

bash scripts/runpod_full_train_easy.sh
```

4. Follow progress:
- Full flow already launches watcher and tracks remote `train_exit_code.txt`.
- For the concise operator summary format used in status updates, use:
```bash
python scripts/runpod_cycle_report_style.py --run-id <run_id>
```
- For supervised/manual remote runs created under `manual_*`, monitor and auto-collect artifacts when complete:
```bash
bash scripts/runpod_cycle_status.sh --watch --auto-collect
bash scripts/telemetry_control.sh status --json
```
- Per-run logs/artifacts are stored under:
  - `artifacts/runpod_cycles/<run_id>/`

5. Validate result artifacts locally:
- Collected artifacts path:
  - `artifacts/runpod_cycles/<run_id>/collected/run_artifacts/`
- Quick play command file:
  - `artifacts/runpod_cycles/<run_id>/quick_play_command.txt`
- Easy-style progress report:
  - `artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md`
  - `artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.json`

## Supervised Cloud Training Reference Notes
Use this section as the default reference when manually supervising active cloud training runs.

1. Start run with explicit supervised intent:
```bash
export RUNPOD_HF_DATASET_REPO_ID="LogicLark-QuantumQuill/chess-bot-datasets"
export RUNPOD_HF_DATASET_PATH_PREFIX="validated_datasets"
export RUNPOD_HF_DATASET_SCHEMA_FILTER="game_jsonl_runtime_splice_v1"
export RUNPOD_GPU_COUNT="2"
export RUNPOD_FULL_TRAIN_NPROC_PER_NODE="2"
bash scripts/runpod_full_train_easy.sh
```
2. During training, supervised monitoring commands:
```bash
bash scripts/runpod_cycle_status.sh --watch
python scripts/runpod_cycle_report_style.py --run-id <run_id>
```
3. Supervision acceptance checks:
- DDP active: `world_size > 1` and `distributed=on` in progress metadata/report.
- Cache behavior: `cache_train=hit` and `cache_val=hit` in easy report whenever cache is expected.
- Progress health: epoch advancement continues and ETA remains finite (not stalled).
- GPU health: average utilization remains high for compute-bound runs and no repeated OOM fallback loops.
4. Completion checks:
- `train_exit_code.txt` equals `0`.
- local model present under `artifacts/runpod_cycles/<run_id>/collected/run_artifacts/`.
- easy report exists at `artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md`.
5. Cost control:
- flow stops pod automatically (`runpod_cycle_stop.sh`).
- terminate explicitly when done with volume/pod lifecycle:
```bash
RUNPOD_CYCLE_RUN_ID="<run_id>" bash scripts/runpod_cycle_terminate.sh
```

## Flow Guarantees and Checks
- Uses HF dataset fetch in pod for training data selection; does not depend on host SSH dataset upload for preferred full flow.
- Uses managed temporary SSH key by default for provision + SSH lifecycle commands.
- Uses cache-first runtime splice behavior and fails fast if runtime splice cache cannot be used.
- Trainer can persist `best` and `epoch-end` checkpoints to disk during training when flow-provided paths are set; watcher attempts to copy those artifacts locally each epoch end.
- Stops pod at end of flow (`runpod_cycle_stop.sh`); explicit termination remains a separate operator action.

## Iteration Policy
Update this spec whenever training-flow behavior or preferred parameter defaults change (GPU tier, epoch count, schema filter, cache policy, watcher behavior, or artifact checks).
