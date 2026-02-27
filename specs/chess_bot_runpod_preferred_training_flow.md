# Chess Bot RunPod Preferred Training Flow

## Responsibility
Define the current preferred end-to-end RunPod training flow, including exact operator steps and run parameters we want to keep stable across iterations.

## Scope
- Host-side orchestration script: `scripts/runpod_full_train_easy.sh`
- Underlying lifecycle: `scripts/runpod_cycle_full_train_hf.sh` and `scripts/runpod_cycle_*.sh`
- Dataset source: Hugging Face dataset repo fetch inside pod (no host->pod dataset push)
- Progress/watchdog: `scripts/runpod_cycle_watch_progress.sh` and remote training sentinels

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
  - remote auto logic chooses `num_workers=max(nproc-1,1)`
  - batch size chosen by remote VRAM tier heuristic
- SSH handling:
  - managed temp key auto-generated on host at `${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}`
  - no personal-key passphrase prompt path in preferred flow

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
- `no_progress` enabled (machine-readable JSONL progress stream still enabled for watcher)

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

# Leave these unset for auto behavior:
unset RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE
unset RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE

bash scripts/runpod_full_train_easy.sh
```

4. Follow progress:
- Full flow already launches watcher and tracks remote `train_exit_code.txt`.
- Per-run logs/artifacts are stored under:
  - `artifacts/runpod_cycles/<run_id>/`

5. Validate result artifacts locally:
- Collected artifacts path:
  - `artifacts/runpod_cycles/<run_id>/collected/run_artifacts/`
- Quick play command file:
  - `artifacts/runpod_cycles/<run_id>/quick_play_command.txt`

## Flow Guarantees and Checks
- Uses HF dataset fetch in pod for training data selection; does not depend on host SSH dataset upload for preferred full flow.
- Uses managed temporary SSH key by default for provision + SSH lifecycle commands.
- Uses cache-first runtime splice behavior and fails fast if runtime splice cache cannot be used.
- Stops pod at end of flow (`runpod_cycle_stop.sh`); explicit termination remains a separate operator action.

## Iteration Policy
Update this spec whenever training-flow behavior or preferred parameter defaults change (GPU tier, epoch count, schema filter, cache policy, watcher behavior, or artifact checks).
