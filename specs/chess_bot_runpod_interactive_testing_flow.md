# Chess Bot RunPod Interactive Testing Flow

## Responsibility
Define a separate, reusable RunPod test-session workflow for iterative/manual testing that keeps one pod reusable across restarts and supports controlled training interruption/restart without modifying the full end-to-end training flow.

## Scope
- `scripts/runpod_cycle_interactive_test.sh`
- `scripts/runpod_cycle_start.sh`
- `scripts/runpod_cycle_watch_progress.sh`
- `scripts/runpod_cycle_status.sh`
- `scripts/runpod_cycle_stop.sh`
- `scripts/runpod_cycle_terminate.sh`

## Separation Contract
- Interactive/manual testing uses `scripts/runpod_cycle_interactive_test.sh`.
- Full end-to-end flow remains `scripts/runpod_full_train_easy.sh` + `scripts/runpod_cycle_full_train_hf.sh`.
- Interactive runs are written under the same run root, but in dedicated remote subdirs named `manual_*`:
  - `<REMOTE_REPO_DIR>/artifacts/runpod_cycles/<RUN_ID>/manual_*`
- This keeps manual test-control operations separate from canonical full-flow orchestration behavior.

## Interactive Session Defaults
- Session id:
  - `RUNPOD_TEST_SESSION_ID=default`
- Run id:
  - `RUNPOD_CYCLE_RUN_ID=runpod-interactive-<session_id>` (derived when unset)
- Start/resume defaults for interactive session:
  - `RUNPOD_INTERRUPTIBLE=1`
  - `RUNPOD_CLOUD_TYPE=SECURE`
  - `RUNPOD_RESUME_STOPPED_POD=1`
  - `RUNPOD_TERMINATE_ON_SSH_NOT_READY=0`

## Commands (`scripts/runpod_cycle_interactive_test.sh`)
- `up`:
  - Provision-or-resume one reusable interactive pod/session.
- `ssh`:
  - Open interactive shell to the session pod.
- `train-start`:
  - Optionally interrupts existing train processes first (default enabled), then launches a new background training control run in a `manual_*` directory.
- `train-stop`:
  - Stops the last tracked interactive training run and writes exit sentinel `130`.
- `train-status`:
  - Prints PID/exit/GPU/progress/log tail status for the tracked interactive run.
- `watch`:
  - Attaches `runpod_cycle_watch_progress.sh` to the tracked interactive run’s progress/exit files.
- `status`:
  - One-shot `runpod_cycle_status.sh` snapshot.
- `down`:
  - Stops pod compute (`podStop`, no terminate).
- `terminate`:
  - Deletes pod resource (`DELETE /pods/<id>`).

## Interactive Train Launch Contract
- Background launcher output files (per `train_id`):
  - `train_progress_<train_id>.jsonl`
  - `train_stdout_<train_id>.log`
  - `train_exit_code.txt`
  - `train_pid.txt`
  - `model_<train_id>.pt`
  - `metrics_<train_id>.json`
  - `model_best_<train_id>.pt`
  - `epoch_checkpoints/`
- HF data behavior:
  - Uses `HF_FETCH_LATEST_ALL_DATASETS=1` with `hf_dataset_fetch.py` manifest under:
    - `<REMOTE_RUN_DIR>/hf_dataset_fetch_manifest.json`
  - Fetch policy controlled by `RUNPOD_INTERACTIVE_FETCH_POLICY`.
- Training script path:
  - prefers repo script: `<REMOTE_REPO_DIR>/deploy/runpod_cloud_training/train_baseline_preset.sh`
  - fallback: `/opt/runpod_cloud_training/train_baseline_preset.sh`

## Local State Contract
- Host-tracked state path:
  - `artifacts/runpod_cycles/<RUN_ID>/interactive/latest_manual_train.json`
- Tracks:
  - `train_id`
  - remote manual dir
  - remote progress/log/exit/pid file paths
  - remote checkpoint paths
- `watch`/`train-stop`/`train-status` use this state file for the active manual run target.

## Notes
- `train-start` is intentionally control-oriented for iterative testing and does not auto-stop pod at training completion.
- Full-flow stop/collect/report behavior remains unchanged and is still verified through full-flow scripts.
