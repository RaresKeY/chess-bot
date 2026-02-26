# Chess Bot RunPod CLI Controls

## Responsibility
Document host-side CLI workflows for building/pushing the RunPod image, diagnosing RunPod API auth, provisioning pods from templates, and controlling a running pod over SSH/Jupyter/HTTP.

## Code Ownership
- `scripts/build_runpod_image.sh`
- `scripts/runpod_provision.py`
- `scripts/runpod_cli_doctor.sh`
- `scripts/runpod_quick_launch.sh`
- `scripts/runpod_regression_checks.sh`
- `scripts/runpod_cycle_common.sh`
- `scripts/runpod_cycle_start.sh`
- `scripts/runpod_cycle_push_dataset.sh`
- `scripts/runpod_cycle_train.sh`
- `scripts/runpod_cycle_collect.sh`
- `scripts/runpod_cycle_local_validate.sh`
- `scripts/runpod_cycle_stop.sh`
- `scripts/runpod_cycle_full_smoke.sh`
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

## Image Build / Push Control
- `scripts/build_runpod_image.sh` builds the RunPod image from `deploy/runpod_cloud_training/Dockerfile`
- `IMAGE_REPO` is required; script now normalizes it to lowercase before tagging (GHCR requires lowercase repository names)
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

## Provision Fallback with Explicit GPU Type ID
- `provision --gpu-type-id <id>` normally validates the ID/display name against GraphQL-ranked GPUs
- If GraphQL GPU discovery is denied (`403`) and `--gpu-type-id` is supplied:
  - provisioning helper now proceeds with the explicit GPU type ID without GraphQL validation
  - emits a stderr JSON warning indicating validation was skipped
- This preserves a CLI provisioning path when the operator can pick the GPU type in RunPod UI first but GraphQL API listing is unavailable to the key

## RunPod CLI Helper Scripts
- `scripts/runpod_cli_doctor.sh`
  - checks local key source (env vs keyring)
  - checks REST auth via `template-list`
  - checks GraphQL auth via `gpu-search`
  - prints a remediation hint when REST works but GraphQL fails
- `scripts/runpod_regression_checks.sh`
  - runs RunPod-focused unit tests (`tests/test_runpod_api_helpers.py`, `tests/test_runpod_local_smoke_script.py`)
  - runs `scripts/runpod_cli_doctor.sh`
  - optionally runs direct `template-list`/`gpu-search` probes when `RUNPOD_API_KEY` is set
  - optionally runs Docker local smoke via `RUN_LOCAL_SMOKE=1`
- `scripts/runpod_quick_launch.sh`
  - thin wrapper around `runpod_provision.py provision`
  - driven by env vars (`RUNPOD_TEMPLATE_NAME`, `RUNPOD_POD_NAME`, `RUNPOD_GPU_TYPE_ID`, etc.)
  - supports quick launch from the host terminal with optional explicit GPU type ID
  - inherits current `runpod_provision.py` defaults (notably `--use-runpod-training-preset-env` is now opt-in)
- `scripts/runpod_cycle_common.sh`
  - shared helpers for modular lifecycle scripts (run id paths, keyring token lookup, pod JSON parsing, SSH/connection fields)
- `scripts/runpod_cycle_start.sh`
  - provisions a pod from template using keyring-backed RunPod auth
  - saves provisioning JSON to `artifacts/runpod_cycles/<run_id>/provision.json`
  - initializes a per-run markdown report under `artifacts/runpod_cycles/<run_id>/reports/observations.md`
- `scripts/runpod_cycle_push_dataset.sh`
  - pushes a valid local dataset directory (`train.jsonl`, `val.jsonl`) to the pod via `rsync`/SSH
  - defaults local dataset source to `data/dataset/_smoke_runpod`
- `scripts/runpod_cycle_train.sh`
  - runs `train_baseline_preset.sh` on the pod with explicit output/metrics paths under `artifacts/runpod_cycles/<run_id>/`
  - runs a short remote inference smoke command (`scripts/infer_move.py`) against the produced model
- `scripts/runpod_cycle_collect.sh`
  - pulls remote run artifacts and timing logs into local `artifacts/runpod_cycles/<run_id>/collected/`
- `scripts/runpod_cycle_local_validate.sh`
  - runs local CPU inference (`scripts/infer_move.py`) on the collected `.pt` artifact and saves output
- `scripts/runpod_cycle_stop.sh`
  - cleanly requests pod stop via RunPod GraphQL `podStop` using keyring-backed token and saved `pod_id`
- `scripts/runpod_cycle_full_smoke.sh`
  - orchestration wrapper that composes the modular scripts in order (start -> push -> train -> collect -> local-validate -> stop)

## Running Pod Control (CLI)
- SSH into the pod using RunPod-provided connection info (port `22/tcp`)
- RunPod SSH gateway command (for example `ssh <podid>-<route>@ssh.runpod.io -i ~/.ssh/id_ed25519`) can work even when direct `runner@<public-ip>:<mapped-port>` auth fails due template `AUTHORIZED_KEYS` mismatch or formatting issues
- Modular cycle scripts support SSH gateway overrides via:
  - `RUNPOD_SSH_USER` (for example `<podid>-<route>`)
  - `RUNPOD_SSH_HOST` (for example `ssh.runpod.io`)
  - `RUNPOD_SSH_PORT` (default `22` for gateway)
- Jupyter is exposed on HTTP port `8888`
- Inference API is exposed on HTTP port `8000`
- Common operator commands after connect:
  - `bash /opt/runpod_cloud_training/train_baseline_preset.sh`
  - `curl http://127.0.0.1:8000/healthz`
  - `tail -f /workspace/chess-bot/artifacts/timings/runpod_phase_times.jsonl`
- Data/artifact transfer options:
  - `scp`
  - `rsync`
  - Jupyter file browser
  - `rclone` (installed in image)

## Modular Lifecycle Flow (host-side)
- Full cycle (defaults to a short smoke run and local `_smoke_runpod` dataset):
  - `bash scripts/runpod_cycle_full_smoke.sh`
- Stepwise modular flow:
  - If using the RunPod SSH gateway, export gateway overrides before steps 2-4:
    - `export RUNPOD_SSH_USER='<podid>-<route>'`
    - `export RUNPOD_SSH_HOST='ssh.runpod.io'`
    - `export RUNPOD_SSH_PORT=22`
  1. `bash scripts/runpod_cycle_start.sh`
  2. `bash scripts/runpod_cycle_push_dataset.sh`
  3. `bash scripts/runpod_cycle_train.sh`
  4. `bash scripts/runpod_cycle_collect.sh`
  5. `bash scripts/runpod_cycle_local_validate.sh`
  6. `bash scripts/runpod_cycle_stop.sh`
- Shared run identifier:
  - set `RUNPOD_CYCLE_RUN_ID=<custom-id>` to keep all files under one run directory
- Core outputs per run:
  - `artifacts/runpod_cycles/<run_id>/provision.json`
  - `artifacts/runpod_cycles/<run_id>/reports/observations.md`
  - `artifacts/runpod_cycles/<run_id>/collected/run_artifacts/*`
  - `artifacts/runpod_cycles/<run_id>/local_validation/infer_move_output.txt`

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
- RunPod-provided SSH gateway command succeeded:
  - `ssh 4rgfwq4u8cbodn-64410e24@ssh.runpod.io -i ~/.ssh/id_ed25519`
- Resulting shell landed as `root` inside the container (`root@...:/#`)
- Practical implication:
  - operators can continue the lifecycle via gateway SSH while fixing template `AUTHORIZED_KEYS`
  - for lifecycle scripts, set `RUNPOD_SSH_USER/HOST/PORT` to the gateway endpoint
- Additional root cause found during follow-up debugging:
  - even with correct `/home/runner/.ssh/authorized_keys`, direct mapped `runner@<ip>:<port>` can still be denied when the `runner` account is locked by default (`useradd` on Ubuntu)
  - deployment flow now fixes this automatically by unlocking `runner` at image build/startup while keeping `PasswordAuthentication no`

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
