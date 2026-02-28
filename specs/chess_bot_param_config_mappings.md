# Chess Bot Parameter/Config Mappings

## Responsibility
Central mapping for non-constant runtime controls (env vars, CLI flags, and script parameters) that change execution behavior.

## Scope
- `scripts/runpod_cycle_benchmark_matrix.sh`
- `scripts/runpod_cycle_start.sh`
- `scripts/runpod_provision.py`
- `scripts/runpod_cycle_full_train_hf.sh`
- `scripts/runpod_full_train_easy.sh`
- `scripts/runpod_active_pods_full_status.sh`
- `scripts/train_baseline.py`

## Mapping Rules
- Every runtime control documented here must include: name, default, accepted values, effect, and related command/script.
- Precedence: CLI flag overrides env var defaults when both exist in the same entrypoint.
- For HF game datasets with `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1`, runtime splice cache config must match training runtime splice parameters.

## Precedence Contracts
- Shared secret resolution (`src/chessbot/secrets.py`):
  - default order is caller-defined but canonical project usage is `explicit -> env -> keyring -> dotenv`.
  - env precedence inside a tuple follows declared order (for example `HF_READ_TOKEN` before `HF_TOKEN`).
- RunPod API key (`scripts/runpod_provision.py`):
  - `--api-key` -> `RUNPOD_API_KEY` -> keyring (`runpod`/`RUNPOD_API_KEY`) -> dotenv (`.env.runpod`, `.env`).
- HF read token (`scripts/hf_dataset_fetch.py`):
  - `--token` -> `HF_READ_TOKEN` -> `HF_TOKEN` -> keyring -> dotenv.
- Benchmark runtime splice cap (`scripts/runpod_cycle_benchmark_matrix.sh`):
  - explicit numeric `RUNPOD_BENCH_RUNTIME_MAX_SAMPLES_PER_GAME` overrides all.
  - `auto` resolves from fetched runtime cache manifests.
  - unresolved/mixed cache values fallback to `0` with warning telemetry.
- Active pod status (`scripts/runpod_active_pods_full_status.sh`):
  - `--running-only` requires API lookups/token (cannot be used with missing API token when API lookups are enabled).
  - `--no-api` disables API enrichment and uses local registry-only view.

## RunPod Benchmark Matrix (`scripts/runpod_cycle_benchmark_matrix.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_GPU_TYPE_ID` | `NVIDIA A40` | RunPod GPU type id/display name | Pod GPU selection | `bash scripts/runpod_cycle_benchmark_matrix.sh` |
| `RUNPOD_GPU_COUNT` | `2` | integer `>=1` | GPU count for pod + default `nproc` | same |
| `RUNPOD_CLOUD_TYPE` | start-script default (`COMMUNITY`) unless overridden | `SECURE`, `COMMUNITY` | Cloud tier for provisioning | same |
| `RUNPOD_INTERRUPTIBLE` | `0` | `0`, `1` | Spot/interruptible request via provision helper | same |
| `RUNPOD_BENCH_TRIALS` | `fp32,tf32,fp16,bf16,sparsity` | comma-separated trial list | Precision/sparsity trial matrix | same |
| `RUNPOD_BENCH_EPOCHS` | `1` | integer `>=1` | Epochs per trial | same |
| `RUNPOD_BENCH_BATCH_SIZE` | `auto` | `auto` or integer | Base batch strategy; auto resolves by remote VRAM | same |
| `RUNPOD_BENCH_NUM_WORKERS` | `8` | integer `>=0` | DataLoader workers per rank | same |
| `RUNPOD_BENCH_DISTRIBUTED_BACKEND` | `nccl` | `nccl`, `gloo` | DDP backend for trial launch | same |
| `RUNPOD_BENCH_RUNTIME_MAX_SAMPLES_PER_GAME` | `auto` | `auto` or integer `>=0` | Runtime splice cap. `auto` now resolves from fetched cache manifests; fallback `0` if unresolved | same |
| `RUNPOD_BENCH_MAX_TOTAL_ROWS` | `0` | integer `>=0` | Global row cap passed to training | same |
| `RUNPOD_BENCH_NCCL_SAFE_FALLBACK_ENABLED` | `1` | `0`, `1` | Retry stalled pre-epoch NCCL trials with safer env | same |
| `RUNPOD_BENCH_TRANSFER_TOOL` | `rclone` | `rclone`, `rsync` | Artifact transfer method | same |
| `RUNPOD_BENCH_TRANSFER_STRICT` | `0` | `0`, `1` | Fail instead of rsync fallback if requested transfer tool unavailable | same |
| `RUNPOD_BENCH_SKIP_FINAL_COLLECT` | `0` | `0`, `1` | Skip final `runpod_cycle_collect.sh` | same |

## Provisioning (`scripts/runpod_cycle_start.sh` + `scripts/runpod_provision.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_TEMPLATE_NAME` | `chess-bot-training` | template name | Template selection for pod create | `bash scripts/runpod_cycle_start.sh` |
| `RUNPOD_GPU_TYPE_ID` | `NVIDIA GeForce RTX 3090` (start script) | GPU type id/display name | Explicit GPU selection; bypasses GraphQL discovery | same |
| `RUNPOD_GPU_COUNT` | `1` (start script) | integer `>=1` | Requested GPU count | same |
| `RUNPOD_INTERRUPTIBLE` | `0` | `0`, `1` | Mapped to `runpod_provision.py --interruptible/--no-interruptible` and REST `interruptible` field | same |
| `RUNPOD_INJECT_MANAGED_SSH_KEY_ENV` | `1` | `0`, `1` | Inject managed temp public key into pod env | same |
| `RUNPOD_SET_UNIQUE_REPO_DIR` | `1` | `0`, `1` | Per-run `REPO_DIR` injection to avoid stale volume collisions | same |
| `RUNPOD_REQUIRE_SSH_READY` | `1` | `0`, `1` | Wait for direct SSH readiness before start success | same |
| `RUNPOD_TERMINATE_ON_SSH_NOT_READY` | `1` | `0`, `1` | Auto-terminate if readiness times out | same |

## Full-Train Wrappers (`scripts/runpod_full_train_easy.sh`, `scripts/runpod_cycle_full_train_hf.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_HF_DATASET_REPO_ID` | project default | HF repo id | Source dataset repo for remote fetch | `bash scripts/runpod_full_train_easy.sh` |
| `RUNPOD_HF_DATASET_PATH_PREFIX` | `validated_datasets` | repo prefix | Dataset path root in HF repo | same |
| `RUNPOD_HF_DATASET_SCHEMA_FILTER` | `game_jsonl_runtime_splice_v1` | schema id string | Chooses dataset format from HF manifests | same |
| `RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS` | unset/`0` | integer `>=0` | Training subset cap | same |
| `RUNPOD_FULL_TRAIN_NPROC_PER_NODE` | `${RUNPOD_GPU_COUNT}` | integer `>=1` | Torchrun process count | same |
| `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` | unset | integer `>=0` | Override auto worker policy | same |
| `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE` | `1` in HF flow | `0`, `1` | Force cache-only runtime splice indexing | same |

## Training CLI (`scripts/train_baseline.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--amp` / `--no-amp` | enabled in presets | boolean | Mixed precision enable/disable | `python scripts/train_baseline.py ...` |
| `--amp-dtype` | `auto` | `auto`, `fp16`, `bf16` | Autocast dtype selection | same |
| `--tf32` | preset-controlled | `on`, `off` | TensorFloat32 matmul/cuDNN controls | same |
| `--distributed-backend` | `nccl` in multi-GPU runs | backend id | DDP backend | same |
| `--runtime-max-samples-per-game` | runtime-dependent | integer `>=0` | Runtime splice cap; must match cache config when cache-required | same |
| `--require-runtime-splice-cache` | often enabled in cloud HF flows | boolean | Fail on cache miss/mismatch instead of runtime indexing | same |
| `--max-total-rows` | `0` | integer `>=0` | Row cap for fast subset tests | same |

## Active-Pods Full Status (`scripts/runpod_active_pods_full_status.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--no-api` | disabled | flag | Skip RunPod REST enrichment | `bash scripts/runpod_active_pods_full_status.sh` |
| `--no-ssh` | disabled | flag | Skip SSH remote probes | same |
| `--running-only` | disabled | flag | Keep only `desiredStatus=RUNNING` pods (requires API) | same |
| `--no-write` | disabled | flag | Skip report file write under `artifacts/reports` | same |
| `RUNPOD_STATUS_SSH_TIMEOUT_SECONDS` | `12` | integer `>=1` | Timeout for per-pod SSH probe wrapper | same |
| `RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS` | `10` | integer `>=1` | SSH connect timeout | same |

## Test Coverage
- `tests/test_secrets_resolution.py`
  - dotenv parsing + explicit/env/keyring/dotenv precedence order + dotenv path ordering.
- `tests/test_hf_dataset_fetch.py`
  - HF token resolution precedence (`--token`, `HF_READ_TOKEN`, `HF_TOKEN`, keyring, dotenv).
- `tests/test_runpod_api_helpers.py`
  - RunPod API key precedence and `--interruptible` parser/create payload behavior.
- `tests/test_config_precedence_matrix.py`
  - script-level precedence/override behavior for active-pod status modes and parser toggle order.
- `tests/test_runpod_cycle_scripts.py`
  - benchmark matrix runtime max-samples auto-resolution contract and critical config path assertions.
