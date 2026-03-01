# Chess Bot Parameter/Config Mappings

## Responsibility
Central mapping for non-constant runtime controls (env vars, CLI flags, and script parameters) that change execution behavior.

## Scope
- `scripts/runpod_cycle_benchmark_matrix.sh`
- `scripts/runpod_cycle_start.sh`
- `scripts/runpod_provision.py`
- `scripts/runpod_sdk_component.py`
- `scripts/runpod_sdk_cycle_start.sh`
- `scripts/runpod_sdk_cycle_stop.sh`
- `scripts/runpod_sdk_cycle_full_smoke.sh`
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
| `RUNPOD_SSH_READY_TIMEOUT_SECONDS` | `360` | integer `>=1` | SSH readiness deadline before start failure | same |
| `RUNPOD_TERMINATE_ON_SSH_NOT_READY` | `1` | `0`, `1` | Auto-terminate if readiness times out | same |

## RunPod SDK Component (`scripts/runpod_sdk_component.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--api-key` | unset | string | Highest-precedence RunPod API key input | `python scripts/runpod_sdk_component.py <subcommand> ...` |
| `RUNPOD_API_KEY` | unset | string | Env fallback for API key when CLI flag omitted | same |
| `--keyring-service` | `runpod` | string | Keyring service for API key lookup | same |
| `--keyring-username` | `RUNPOD_API_KEY` | string | Keyring username for API key lookup | same |
| `RUNPOD_SDK_DOTENV_PATH` | unset | dotenv filepath | SDK-component-specific dotenv override for API key fallback | same |
| `--cloud-type` (`gpu-search`, `provision`) | `COMMUNITY` | `SECURE`, `COMMUNITY` | Cloud tier used for GPU ranking/filtering | same |
| `--min-memory-gb` (`gpu-search`, `provision`) | `24` | integer `>=0` | Minimum VRAM filter for SDK GPU ranking | same |
| `--max-hourly-price` (`gpu-search`, `provision`) | `0.0` | float `>=0` (`0` disables cap) | Optional max price filter for SDK GPU ranking | same |
| `--template-id` / `--template-name` (`provision`) | `""` / `chess-bot-training` | template id/name | Template selection for pod create via SDK path | same |
| `--gpu-type-id` (`provision`) | unset | GPU type id/name | Explicit GPU type override; skips auto-pick from ranked SDK GPU list | same |
| `--interruptible` (`provision`) | `False` | boolean flag | Requests spot/interruptible pod in SDK provision payload | same |
| `--wait-ready` (`provision`) | `True` | boolean flag | Poll pod status after create until running/ready or timeout | same |
| `--wait-timeout-seconds` (`provision`) | `900` | integer `>=1` | Wait deadline for `--wait-ready` polling | same |
| `--wait-poll-seconds` (`provision`) | `10` | integer `>=1` | Poll interval for `--wait-ready` | same |
| `--pod-id` (`pod-status`, `pod-stop`, `pod-terminate`) | required | string | Target pod id for status/stop/terminate operations | same |

## RunPod SDK Smoke Flow Wrappers (`scripts/runpod_sdk_cycle_*.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_CYCLE_RUN_ID` | `runpod-sdk-cycle-<utc-ts>` (full smoke) / shared helper default | string | Run id for per-run artifacts and telemetry paths | `bash scripts/runpod_sdk_cycle_full_smoke.sh` |
| `RUNPOD_SDK_TEMPLATE_NAME` | fallback to `RUNPOD_TEMPLATE_NAME` then `chess-bot-training` | template name | SDK-specific template name override for `runpod_sdk_cycle_start.sh` without changing raw start defaults | `bash scripts/runpod_sdk_cycle_start.sh` |
| `RUNPOD_TEMPLATE_NAME` | `chess-bot-training` | template name | Shared fallback template name if SDK-specific override is unset | same |
| `RUNPOD_GPU_TYPE_ID` | `NVIDIA GeForce RTX 3090` | GPU type id/display name | Explicit GPU selection for SDK provision call | same |
| `RUNPOD_GPU_COUNT` | `1` | integer `>=1` | Requested GPU count for SDK provision call | same |
| `RUNPOD_CLOUD_TYPE` | `COMMUNITY` | `SECURE`, `COMMUNITY` | Cloud tier for SDK provision | same |
| `RUNPOD_INTERRUPTIBLE` | `0` | `0`, `1` | Mapped to `--interruptible/--no-interruptible` in SDK start wrapper | same |
| `RUNPOD_INJECT_MANAGED_SSH_KEY_ENV` | `1` | `0`, `1` | Inject managed temp SSH pubkey env vars (`AUTHORIZED_KEYS`, `PUBLIC_KEY`) during SDK provision | same |
| `RUNPOD_SET_UNIQUE_REPO_DIR` | `1` | `0`, `1` | Inject per-run `REPO_DIR` to avoid stale persistent-volume repos | same |
| `RUNPOD_SET_SMOKE_SERVICE_ENVS` | `1` | `0`, `1` | Inject smoke-safe service envs (`START_SSHD=1`, others `0`) for stable smoke SSH runtime | same |
| `RUNPOD_START_ENVS` | unset | whitespace-separated `KEY=VALUE` pairs | Additional env pairs forwarded to SDK provision command as repeated `--env` | same |
| `RUNPOD_REQUIRE_SSH_READY` | `1` | `0`, `1` | Wait for direct SSH readiness before SDK start returns success | same |
| `RUNPOD_SSH_READY_TIMEOUT_SECONDS` | `360` | integer `>=1` | SSH readiness timeout in SDK start wrapper | same |
| `RUNPOD_SSH_READY_POLL_SECONDS` | `8` | integer `>=1` | SSH readiness poll interval | same |
| `RUNPOD_TERMINATE_ON_SSH_NOT_READY` | `1` | `0`, `1` | Auto-terminate pod on SSH readiness timeout | same |
| `RUNPOD_POD_JSON` | `artifacts/runpod_cycles/<run_id>/provision.json` | filepath | Override provision record path consumed by SDK stop and shared cycle scripts | `bash scripts/runpod_sdk_cycle_stop.sh` |
| `RUNPOD_POD_ID` | from provision JSON | string | Explicit pod id override for SDK stop wrapper | same |

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
