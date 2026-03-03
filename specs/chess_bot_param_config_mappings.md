# Chess Bot Parameter/Config Mappings

## Responsibility
Central mapping for non-constant runtime controls (env vars, CLI flags, and script parameters) that change execution behavior.

## Scope
- `scripts/runpod_cycle_benchmark_matrix.sh`
- `scripts/runpod_cycle_start.sh`
- `scripts/runpod_cycle_interactive_test.sh`
- `scripts/runpod_provision.py`
- `scripts/runpod_sdk_component.py`
- `scripts/runpod_sdk_cycle_start.sh`
- `scripts/runpod_sdk_cycle_stop.sh`
- `scripts/runpod_sdk_cycle_full_smoke.sh`
- `scripts/container_ensure_openssh.sh`
- `scripts/runpod_cycle_full_train_hf.sh`
- `scripts/runpod_full_train_easy.sh`
- `scripts/runpod_full_train_easy_smoke_test.sh`
- `scripts/runpod_active_pods_full_status.sh`
- `deploy/runpod_cloud_training/train_baseline_preset.sh`
- `scripts/vast_local_smoke_test.sh`
- `scripts/vast_noauth_deploy_checks.sh`
- `scripts/train_baseline.py`
- `scripts/train_dual_sequence.py`
- `scripts/infer_move.py`
- `scripts/play_model_vs_model.py`

## Mapping Rules
- Every runtime control documented here must include: name, default, accepted values, effect, and related command/script.
- Precedence: CLI flag overrides env var defaults when both exist in the same entrypoint.
- For HF game datasets with `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1`, runtime splice cache config must match training runtime splice parameters.

## Precedence Contracts
- Shared secret resolution (`src/chessbot/secrets.py`):
  - default order is caller-defined but canonical project usage is `explicit -> env -> keyring -> dotenv`.
  - env precedence inside a tuple follows declared order (for example `HF_READ_TOKEN` before `HF_TOKEN`).
  - canonical token/key-provider mapping and container guidance live in `specs/chess_bot_secrets_contract.md`.
  - current `/work` container preference is dotenv provider paths over keyring (keyring module unavailable in this runtime).
- RunPod API key (`scripts/runpod_provision.py`):
  - `--api-key` -> `RUNPOD_API_KEY` -> keyring fallback -> dotenv (`.env.runpod`, `.env`).
- HF read token (`scripts/hf_dataset_fetch.py`):
  - `--token` -> `HF_READ_TOKEN` -> `HF_TOKEN` -> keyring fallback -> dotenv.
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
| `RUNPOD_CLOUD_TYPE` | `SECURE` | `SECURE`, `COMMUNITY` | Cloud tier for provisioning | same |
| `RUNPOD_INTERRUPTIBLE` | `0` | `0`, `1` | Spot/interruptible request via provision helper | same |
| `RUNPOD_BENCH_TRIALS` | `fp32,tf32,fp16,bf16,sparsity,bf16_2to4` | comma-separated trial list (`fp32`, `tf32`, `fp16`, `bf16`, `fp32_sparse`, `fp16_sparse`, `bf16_sparse`, `sparsity`, `fp16_2to4`, `bf16_2to4`) | Precision/sparsity trial matrix; `*_sparse`/`sparsity` use L1 penalty, `*_2to4` uses structured 2:4 sparsity mode | same |
| `RUNPOD_BENCH_EPOCHS` | `1` | integer `>=1` | Epochs per trial | same |
| `RUNPOD_BENCH_BATCH_SIZE` | `auto` | `auto` or integer | Base batch strategy; auto resolves by remote VRAM | same |
| `RUNPOD_BENCH_NUM_WORKERS` | `8` | integer `>=0` | DataLoader workers per rank | same |
| `RUNPOD_BENCH_DISTRIBUTED_BACKEND` | `nccl` | `nccl`, `gloo` | DDP backend for trial launch | same |
| `RUNPOD_BENCH_ROLLOUT_HORIZON` | `8` | integer `>=1` | Baseline rollout horizon used by benchmark trials | same |
| `RUNPOD_BENCH_CLOSENESS_HORIZON` | `${RUNPOD_BENCH_ROLLOUT_HORIZON}` | integer `>=1` | Baseline closeness horizon used by benchmark trials | same |
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
| `RUNPOD_RESUME_STOPPED_POD` | `1` | `0`, `1` | Enable resume-first behavior in `runpod_cycle_start.sh` (attempt `pod-status`/`pod-resume` before new provisioning) | same |
| `RUNPOD_RESUME_POD_ID` | unset | pod id string | Explicit resume candidate pod id override (checked before `RUNPOD_POD_ID` and existing `provision.json`) | same |
| `RUNPOD_RESUME_BID_PER_GPU` | `0.2` | float `>0` | Spot bid-per-GPU used when resume path runs interruptible `podBidResume` | same |
| `RUNPOD_INJECT_MANAGED_SSH_KEY_ENV` | `1` | `0`, `1` | Inject managed temp public key into pod env | same |
| `RUNPOD_SET_UNIQUE_REPO_DIR` | `1` | `0`, `1` | Per-run `REPO_DIR` injection to avoid stale volume collisions | same |
| `RUNPOD_REQUIRE_SSH_READY` | `1` | `0`, `1` | Wait for direct SSH readiness before start success | same |
| `RUNPOD_SSH_READY_TIMEOUT_SECONDS` | `360` | integer `>=1` | SSH readiness deadline before start failure | same |
| `RUNPOD_TERMINATE_ON_SSH_NOT_READY` | `1` | `0`, `1` | Auto-terminate if readiness times out | same |

RunPod cloud-tier deployment policy:
- deployment specs require `RUNPOD_CLOUD_TYPE=SECURE` as the baseline for RunPod cloud deploys.
- `RUNPOD_CLOUD_TYPE=COMMUNITY` is not an approved fallback for standard RunPod deploy/training flows in this project policy.

Explicit-GPU provisioning behavior note (current):
- when `--gpu-type-id` / `RUNPOD_GPU_TYPE_ID` is set, provision now performs a GraphQL stock preflight (`gpuTypes(input:{id})` + `lowestPrice`) and fails fast if the requested `gpu_count` is not listed in `availableGpuCounts` or stock status is out-of-stock/no-capacity.
- this is an existing-control behavior refinement (no new runtime flag), intended to improve resource-availability determination before REST pod create attempts.

Provision start/resume interactions:
- when `RUNPOD_RESUME_STOPPED_POD=1`, `runpod_cycle_start.sh` attempts resume-first behavior:
  - `pod-status` for resume candidate pod id.
  - if state is `EXITED`/`STOPPED`, it calls `pod-resume` with `RUNPOD_GPU_COUNT` and `RUNPOD_RESUME_BID_PER_GPU`.
  - resume interruptible mode defaults to pod-status auto-detect; explicit `RUNPOD_INTERRUPTIBLE` in caller env is forwarded as override (`--interruptible`/`--no-interruptible`).
  - if state is `RUNNING`/`READY`, it reuses the existing pod without creating a new one.
  - otherwise it provisions a new pod as before.
- resume candidate pod id precedence:
  - `RUNPOD_RESUME_POD_ID` -> `RUNPOD_POD_ID` -> existing run `provision.json` pod id.

Direct `runpod_provision.py pod-resume` controls:
- `--pod-id` (required): target pod id to resume.
- `--gpu-count` (default `1`): GPU count passed to resume mutation.
- `--interruptible` / `--no-interruptible` (default auto-detect): force resume mode; default inspects current pod status interruptible fields.
- `--bid-per-gpu` (default `0.2`, float `>0`): required for interruptible spot-bid resume path.
- `--wait-ready` / `--no-wait-ready` (default `--wait-ready`): poll pod status until running/ready after resume.
- `--wait-timeout-seconds` (default `600`) / `--wait-poll-seconds` (default `10`): resume wait timing controls.

## Interactive Testing Flow (`scripts/runpod_cycle_interactive_test.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_TEST_SESSION_ID` | `default` | string | Session namespace used to derive reusable interactive run id when `RUNPOD_CYCLE_RUN_ID` is unset | `bash scripts/runpod_cycle_interactive_test.sh up` |
| `RUNPOD_CYCLE_RUN_ID` | `runpod-interactive-<session_id>` | string | Stable run id/session root reused across `up`/`down`/manual train control commands | same |
| `RUNPOD_INTERRUPTIBLE` | `1` (interactive wrapper default) | `0`, `1` | Interactive `up` default requests spot/interruptible unless caller overrides | same |
| `RUNPOD_CLOUD_TYPE` | `SECURE` (interactive wrapper default) | `SECURE`, `COMMUNITY` | Cloud tier for interactive `up` path | same |
| `RUNPOD_RESUME_STOPPED_POD` | `1` | `0`, `1` | Interactive `up` keeps resume-first behavior (reuse stopped pod before creating new capacity) | same |
| `RUNPOD_TERMINATE_ON_SSH_NOT_READY` | `0` (interactive wrapper default) | `0`, `1` | Interactive `up` avoids auto-terminate on SSH readiness timeout by default | same |
| `RUNPOD_INTERACTIVE_TRAIN_ID` | `manual_<utc-ts>` | string | Manual train run id suffix and remote manual directory name (`manual_*`) | `bash scripts/runpod_cycle_interactive_test.sh train-start` |
| `RUNPOD_INTERACTIVE_STOP_EXISTING` | `1` | `0`, `1` | Before new manual train launch, terminate prior `train_baseline.py`/`torchrun`/preset processes | same |
| `RUNPOD_INTERACTIVE_FETCH_POLICY` | `if_missing` | `if_missing`, `always`, `never` | Controls whether interactive train-start refreshes HF aggregate fetch manifest | same |
| `RUNPOD_HF_DATASET_REPO_ID` | `LogicLark-QuantumQuill/chess-bot-datasets` | HF dataset repo id | Source repo for interactive HF aggregate fetch | same |
| `RUNPOD_HF_DATASET_PATH_PREFIX` | `validated_datasets` | path prefix string | Prefix scanned for `--all-latest` HF dataset fetch in interactive mode | same |
| `RUNPOD_HF_DATASET_SCHEMA_FILTER` | `game_jsonl_runtime_splice_v1` | schema string (`auto` or explicit) | Schema selection forwarded to preset/script for interactive run | same |
| `RUNPOD_INTERACTIVE_TRAIN_EPOCHS` | `1` | integer `>=1` | Epochs for manual interactive train launch | same |
| `RUNPOD_INTERACTIVE_TRAIN_BATCH_SIZE_OVERRIDE` | unset | integer `>=1` | Optional fixed batch override for interactive run (unset keeps preset auto/default behavior) | same |
| `RUNPOD_INTERACTIVE_TRAIN_NUM_WORKERS_OVERRIDE` | unset | integer `>=0` | Optional fixed DataLoader workers override for interactive run | same |
| `RUNPOD_INTERACTIVE_TRAIN_MAX_TOTAL_ROWS` | `0` | integer `>=0` | Optional total-row cap passed via preset extra args (`0` disables cap) | same |
| `RUNPOD_INTERACTIVE_TRAIN_ROLLOUT_HORIZON` | `8` | integer `>=1` | Rollout horizon for interactive train extra args | same |
| `RUNPOD_INTERACTIVE_TRAIN_CLOSENESS_HORIZON` | `${RUNPOD_INTERACTIVE_TRAIN_ROLLOUT_HORIZON}` | integer `>=1` | Closeness horizon for interactive train extra args | same |
| `RUNPOD_INTERACTIVE_TRAIN_NPROC_PER_NODE` | `${RUNPOD_FULL_TRAIN_NPROC_PER_NODE}` or `${RUNPOD_GPU_COUNT}` or `1` | integer `>=1` | Distributed process count for interactive training launch | same |
| `RUNPOD_INTERACTIVE_REQUIRE_RUNTIME_SPLICE_CACHE` | `1` | `0`, `1` | Interactive train cache requirement forwarded to preset (`TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE`) | same |

Interactive flow precedence/interaction notes:
- interactive script defaults are wrapper-level and can be overridden by explicitly exported env vars.
- training-stop and watch target the last tracked manual run from:
  - `artifacts/runpod_cycles/<run_id>/interactive/latest_manual_train.json`
- interactive train launch writes remote `manual_*` subdirectories and does not auto-stop pod compute on completion.

## RunPod SDK Component (`scripts/runpod_sdk_component.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--api-key` | unset | string | Highest-precedence RunPod API key input | `python scripts/runpod_sdk_component.py <subcommand> ...` |
| `RUNPOD_API_KEY` | unset | string | Env fallback for API key when CLI flag omitted | same |
| `--keyring-service` | `runpod` | string | Keyring service for API key lookup | same |
| `--keyring-username` | `RUNPOD_API_KEY` | string | Keyring username for API key lookup | same |
| `RUNPOD_SDK_DOTENV_PATH` | unset | dotenv filepath | SDK-component-specific dotenv override for API key fallback | same |
| `--cloud-type` (`gpu-search`, `provision`) | `SECURE` | `SECURE`, `COMMUNITY` | Cloud tier used for GPU ranking/filtering | same |
| `--min-memory-gb` (`gpu-search`, `provision`) | `24` | integer `>=0` | Minimum VRAM filter for SDK GPU ranking | same |
| `--max-hourly-price` (`gpu-search`, `provision`) | `0.0` | float `>=0` (`0` disables cap) | Optional max price filter for SDK GPU ranking | same |
| `--template-id` / `--template-name` (`provision`) | `""` / `chess-bot-training` | template id/name | Template selection for pod create via SDK path | same |
| `--image-name` (`provision`) | `""` | docker image reference | Explicit image for SDK create call; required by some SDK variants when no `template_id` is set | same |
| `--gpu-type-id` (`provision`) | unset | GPU type id/name | Explicit GPU type override; skips auto-pick from ranked SDK GPU list | same |
| `--gpu-count` (`provision`, `pod-resume`) | `1` | integer `>=1` | Requested GPU count for create or resume operation | same |
| `--interruptible` (`provision`) | `False` | boolean flag | Requests spot/interruptible pod in SDK provision payload | same |
| `--interruptible` (`pod-resume`) | auto-detect (`None`) | boolean flag | Resume-mode override. Default auto-detects from pod status and uses spot-bid resume for interruptible pods | same |
| `--bid-per-gpu` (`pod-resume`) | `0.2` | float `>0` | Spot bid used by GraphQL `podBidResume` for interruptible resume path | same |
| `--wait-ready` (`provision`, `pod-resume`) | `True` | boolean flag | Poll pod status until running/ready or timeout after create/resume | same |
| `--wait-timeout-seconds` (`provision`, `pod-resume`) | `900` | integer `>=1` | Wait deadline for `--wait-ready` polling | same |
| `--wait-poll-seconds` (`provision`, `pod-resume`) | `10` | integer `>=1` | Poll interval for `--wait-ready` | same |
| `--pod-id` (`pod-status`, `pod-stop`, `pod-resume`, `pod-terminate`) | required | string | Target pod id for status/stop/resume/terminate operations | same |

## RunPod SDK Smoke Flow Wrappers (`scripts/runpod_sdk_cycle_*.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_CYCLE_RUN_ID` | `runpod-sdk-cycle-<utc-ts>` (full smoke) / shared helper default | string | Run id for per-run artifacts and telemetry paths | `bash scripts/runpod_sdk_cycle_full_smoke.sh` |
| `RUNPOD_SDK_TEMPLATE_NAME` | fallback to `RUNPOD_TEMPLATE_NAME` then `chess-bot-training` | template name | SDK-specific template name override for `runpod_sdk_cycle_start.sh` without changing raw start defaults | `bash scripts/runpod_sdk_cycle_start.sh` |
| `RUNPOD_SDK_IMAGE_NAME` | unset | docker image reference | SDK-specific image override forwarded to `--image-name` in SDK start wrapper | same |
| `RUNPOD_TEMPLATE_NAME` | `chess-bot-training` | template name | Shared fallback template name if SDK-specific override is unset | same |
| `RUNPOD_GPU_TYPE_ID` | `NVIDIA GeForce RTX 3090` | GPU type id/display name | Explicit GPU selection for SDK provision call | same |
| `RUNPOD_GPU_COUNT` | `1` | integer `>=1` | Requested GPU count for SDK provision call | same |
| `RUNPOD_CLOUD_TYPE` | `SECURE` | `SECURE`, `COMMUNITY` | Cloud tier for SDK provision | same |
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

## Artifact Collection (`scripts/runpod_cycle_collect.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_COLLECT_INCLUDE_EPOCH_CHECKPOINTS` | `1` | `0`, `1` | Include/exclude heavy `epoch_checkpoints` trees during `rsync` collection | `bash scripts/runpod_cycle_collect.sh` |
| `RUNPOD_COLLECT_CONFIRMATION_PROFILE` | `generic` | `generic`, `full_hf` | Chooses sent-back confirmation rules written to `collected/logs_auto/collection_confirmation.json` (`full_hf` requires HF/context/GPU sample artifacts in addition to model/metrics/exit-code) | same |
| `RUNPOD_COLLECT_REQUIRE_TRAIN_ARTIFACTS` | `0` | `0`, `1` | When `1`, collection exits non-zero if confirmation checks fail (artifacts not confirmed sent-back) | same |

Artifact collection confirmation outputs:
- `artifacts/runpod_cycles/<run_id>/collected/logs_auto/collection_confirmation.json` always records `artifacts_sent_back_confirmed`, required checks, and resolved artifact paths.
- full-HF flow runs collect with `RUNPOD_COLLECT_CONFIRMATION_PROFILE=full_hf` and can require strict confirmation before auto-stop.

## Full-Train Wrappers (`scripts/runpod_full_train_easy.sh`, `scripts/runpod_full_train_easy_smoke_test.sh`, `scripts/runpod_cycle_full_train_hf.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_HF_DATASET_REPO_ID` | project default | HF repo id | Source dataset repo for remote fetch | `bash scripts/runpod_full_train_easy.sh` |
| `RUNPOD_HF_DATASET_PATH_PREFIX` | `validated_datasets` | repo prefix | Dataset path root in HF repo | same |
| `RUNPOD_HF_DATASET_SCHEMA_FILTER` | `game_jsonl_runtime_splice_v1` | schema id string | Chooses dataset format from HF manifests | same |
| `RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS` | `5000` (smoke wrapper) / unset-`0` (easy/full-HF wrappers) | integer `>=0` | Training subset cap; smoke wrapper now defaults to a 5K-row quick run (`~4445` train + `~555` val by split ratio), while easy/full-HF keep uncapped behavior unless overridden | same |
| `RUNPOD_FULL_TRAIN_ROLLOUT_HORIZON` | `8` | integer `>=1` | Full-HF rollout horizon (mapped to `TRAIN_ROLLOUT_HORIZON` and baseline `--rollout-horizon`) | same |
| `RUNPOD_FULL_TRAIN_CLOSENESS_HORIZON` | `${RUNPOD_FULL_TRAIN_ROLLOUT_HORIZON}` | integer `>=1` | Full-HF closeness horizon (mapped to `TRAIN_CLOSENESS_HORIZON` and baseline `--closeness-horizon`) | same |
| `RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME` | `auto` (smoke wrapper) / `0` (full-HF flow default) | `auto` or integer `>=0` | Runtime splice per-game cap for game-format datasets. When set to `auto` with cache-required mode, full-HF flow resolves cache-matching runtime config from fetched `runtime_splice_cache/manifest.json` before training | same |
| `RUNPOD_FULL_TRAIN_NPROC_PER_NODE` | `${RUNPOD_GPU_COUNT}` | integer `>=1` | Torchrun process count; smoke wrapper defaults to one process per requested GPU (`2`) | same |
| `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` | unset (smoke wrapper auto mode) | integer `>=0` | Optional manual DataLoader workers-per-rank override; when unset in smoke wrapper, auto worker policy is used | same |
| `RUNPOD_FULL_TRAIN_REQUIRE_ARTIFACT_CONFIRMATION` | `1` | `0`, `1` | Require local sent-back confirmation (`collection_confirmation.json`) after collect. When enabled and checks fail, full-HF flow exits with pod left running for inspection | same |
| `RUNPOD_FULL_TRAIN_STOP_AFTER_ARTIFACT_CONFIRMATION` | `1` | `0`, `1` | Auto-stop compute only after artifact confirmation stage. `0` keeps pod running after successful full flow | same |
| `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE` | `1` in HF flow | `0`, `1` | Force cache-only runtime splice indexing | same |
| `RUNPOD_CYCLE_RUN_ID` | `easy-smoke-<utc-ts>` (smoke wrapper) / shared helper default | string | Per-run artifact/telemetry directory id forwarded through easy/full-train flow | `bash scripts/runpod_full_train_easy_smoke_test.sh` |
| `RUNPOD_GPU_TYPE_ID` | `NVIDIA GeForce RTX 5090` (smoke wrapper) / `NVIDIA GeForce RTX 5090` (easy wrapper) | RunPod GPU type id/display name | Default GPU selection for both easy and smoke wrappers; callers can override per run when targeting specific capacity/cost tiers | `bash scripts/runpod_full_train_easy_smoke_test.sh`, `bash scripts/runpod_full_train_easy.sh` |
| `RUNPOD_GPU_COUNT` | `2` (smoke wrapper/easy wrapper) | integer `>=1` | Multi-GPU pod count; smoke wrapper now defaults to 2-GPU validation runs | same |
| `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE` | unset (smoke wrapper auto mode) | integer `>=1` | Optional manual batch override. Smoke wrapper unsets this by default so full-HF auto batch-attempt policy drives selection | same |
| `RUNPOD_FULL_TRAIN_NCCL_SAFE_DEFAULTS` | `1` | `0`, `1` | Enable safe multi-GPU NCCL env defaults in full-HF flow when `TRAIN_NPROC_PER_NODE>1` | `bash scripts/runpod_full_train_easy.sh` |
| `RUNPOD_FULL_TRAIN_NCCL_IB_DISABLE` | `1` | `0`, `1` | Default value used for `NCCL_IB_DISABLE` when safe defaults are enabled and var is unset in pod env | same |
| `RUNPOD_FULL_TRAIN_NCCL_P2P_DISABLE` | `1` | `0`, `1` | Default value used for `NCCL_P2P_DISABLE` when safe defaults are enabled and var is unset in pod env | same |
| `RUNPOD_FULL_TRAIN_TORCH_NCCL_ASYNC_ERROR_HANDLING` | `1` | `0`, `1` | Default value used for `TORCH_NCCL_ASYNC_ERROR_HANDLING` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_TORCH_NCCL_ENABLE_MONITORING` | `1` | `0`, `1` | Default value used for `TORCH_NCCL_ENABLE_MONITORING` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `1800` | integer `>=1` | Default heartbeat timeout used for `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_NCCL_DEBUG` | `WARN` | NCCL debug level string | Default value used for `NCCL_DEBUG` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_NCCL_P2P_LEVEL` | `LOC` | NCCL P2P level string | Default value used for `NCCL_P2P_LEVEL` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_TORCH_NCCL_BLOCKING_WAIT` | `1` | `0`, `1` | Default value used for `TORCH_NCCL_BLOCKING_WAIT` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_TORCH_NCCL_DUMP_ON_TIMEOUT` | `1` | `0`, `1` | Default value used for `TORCH_NCCL_DUMP_ON_TIMEOUT` under safe defaults | same |
| `RUNPOD_FULL_TRAIN_TORCH_NCCL_TRACE_BUFFER_SIZE` | `2000` | integer `>=1` | Default value used for `TORCH_NCCL_TRACE_BUFFER_SIZE` under safe defaults | same |
| `RUNPOD_SMOKE_EVENT_LOG` | `artifacts/runpod_cycles/<run_id>/logs/full_smoke_events.log` | filepath | Timestamped full stdout/stderr event stream for the smoke wrapper (`awk` UTC prefix + `tee`) | `bash scripts/runpod_full_train_easy_smoke_test.sh` |
| `RUNPOD_SMOKE_SUMMARY_JSON` | `artifacts/runpod_cycles/<run_id>/reports/smoke_summary.json` | filepath | Post-run machine-readable smoke summary (timing, metrics tail, progress tail, GPU sample stats) generated by smoke wrapper | `bash scripts/runpod_full_train_easy_smoke_test.sh` |

Full-train runtime splice interactions:
- `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1` remains enforced in the full-HF flow.
- when `RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME=auto`, the flow resolves runtime splice config from the fetched dataset cache manifest (`runtime_splice_cache/manifest.json`) and uses those resolved values for `TRAIN_RUNTIME_MIN_CONTEXT`, `TRAIN_RUNTIME_MIN_TARGET`, and `TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME` before launching training.
- subset caps (`RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS`, `RUNPOD_FULL_TRAIN_MAX_TRAIN_ROWS`, `RUNPOD_FULL_TRAIN_MAX_VAL_ROWS`) now apply to both single-step (`rollout_horizon=1`) and multistep (`rollout_horizon>1`) training paths, including distributed (DDP) runs.
- rollout controls:
  - `RUNPOD_FULL_TRAIN_ROLLOUT_HORIZON` maps to `TRAIN_ROLLOUT_HORIZON` (default `8`).
  - `RUNPOD_FULL_TRAIN_CLOSENESS_HORIZON` maps to `TRAIN_CLOSENESS_HORIZON` (default rollout value).
- when `RUNPOD_FULL_TRAIN_NCCL_SAFE_DEFAULTS=1` and `TRAIN_NPROC_PER_NODE>1`, the full-HF flow applies safe NCCL defaults only when the target raw env var is currently unset (`NCCL_*` / `TORCH_NCCL_*`), so explicit pod/container env values remain highest precedence.
- smoke-wrapper auto-batch precedence:
  - if `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE` / `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` are unset in caller env, smoke wrapper intentionally unsets them so full-HF auto selection remains active.
  - if caller provides either override env var, smoke wrapper preserves it and forwards it unchanged to `runpod_full_train_easy.sh`.
- artifact confirmation and stop sequencing:
  - full-HF flow collects artifacts with `RUNPOD_COLLECT_CONFIRMATION_PROFILE=full_hf`.
  - when `RUNPOD_FULL_TRAIN_REQUIRE_ARTIFACT_CONFIRMATION=1`, missing confirmation skips failure auto-stop and leaves pod running for manual inspection/recovery.
  - when `RUNPOD_FULL_TRAIN_STOP_AFTER_ARTIFACT_CONFIRMATION=1`, stop is invoked only after `artifacts_sent_back_confirmed=true`.

## Cloud Train Preset (`deploy/runpod_cloud_training/train_baseline_preset.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `TRAIN_NPROC_PER_NODE` | `1` | integer `>=1` | Torchrun process count in container preset; values `>1` enable multi-GPU launch path | `bash "$REPO_DIR/deploy/runpod_cloud_training/train_baseline_preset.sh"` |
| `TRAIN_PROGRESS_JSONL_OUT` | unset (auto-set by watchdog in multi-GPU mode) | filepath | Progress-event JSONL path used for watchdog liveness checks | same |
| `TRAIN_NCCL_HANG_CHECK_ENABLED` | `1` | `0`, `1` | Enable/disable multi-GPU NCCL hang watchdog around training command | same |
| `TRAIN_NCCL_HANG_TIMEOUT_SECONDS` | `900` | integer `>=1` | Max idle time without progress updates before hang detection | same |
| `TRAIN_NCCL_HANG_POLL_SECONDS` | `30` | integer `>=1` | Poll cadence for progress-file mtime checks (`<= timeout`) | same |
| `TRAIN_NCCL_HANG_EXIT_CODE` | `124` | integer `1..255` | Exit code returned when watchdog terminates a stalled run | same |
| `TRAIN_NCCL_HANG_LOG_PATH` | `<dirname(METRICS_OUT)>/nccl_hang_watchdog_<ts>.log` | filepath | Cloud diagnostic log path written on hang detect (NCCL env, process snapshot, GPU status, progress tail) | same |

Cloud train preset NCCL-watchdog interactions:
- watchdog path is active only when `TRAIN_NPROC_PER_NODE>1` and `TRAIN_NCCL_HANG_CHECK_ENABLED=1`.
- when watchdog is active and `TRAIN_PROGRESS_JSONL_OUT` is unset, the preset auto-assigns a per-run progress JSONL path under the run artifacts directory.
- on hang detection, watchdog writes diagnostics to `TRAIN_NCCL_HANG_LOG_PATH`, sends `TERM` then `KILL` to training processes, and returns `TRAIN_NCCL_HANG_EXIT_CODE`.

## Training CLI (`scripts/train_baseline.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--amp` / `--no-amp` | enabled in presets | boolean | Mixed precision enable/disable | `python scripts/train_baseline.py ...` |
| `--amp-dtype` | `auto` | `auto`, `fp16`, `bf16` | Autocast dtype selection | same |
| `--tf32` | `auto` | `auto`, `on`, `off` | TensorFloat32 matmul/cuDNN controls | same |
| `--distributed-backend` | `nccl` in multi-GPU runs | backend id | DDP backend | same |
| `--rollout-horizon` | `1` | integer `>=1` | Multistep rollout horizon (`1` keeps single-step baseline) | same |
| `--closeness-horizon` | `4` (clamped to rollout) | integer `>=1` | Continuation-closeness horizon used in rollout evaluation | same |
| `--runtime-max-samples-per-game` | runtime-dependent | integer `>=0` | Runtime splice cap; must match cache config when cache-required | same |
| `--require-runtime-splice-cache` | often enabled in cloud HF flows | boolean | Fail on cache miss/mismatch instead of runtime indexing | same |
| `--max-total-rows` | `0` | integer `>=0` | Row cap for fast subset tests | same |
| `--sparsity-mode` | `off` | `off`, `l1`, `structured_2to4` | Sparsity behavior: `l1` adds L1 penalty, `structured_2to4` enforces persistent 2:4 masks on linear weights | same |
| `--sparsity-l1-lambda` | `0.0` | float `>=0` | L1 penalty multiplier when `--sparsity-mode=l1` | same |
| `--sparsity-include-bias` | `False` | boolean flag | Include bias tensors in L1/stat tracking (applies to `l1` mode) | same |

Training CLI interactions:
- `sparsity_mode` is currently supported only for `rollout_horizon=1` in training code.
- RunPod benchmark matrix skips sparse/2:4 trials automatically when `RUNPOD_BENCH_ROLLOUT_HORIZON > 1`.

## Dual Sequence Training CLI (`scripts/train_dual_sequence.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--train` | required | repeatable JSONL path | Training input paths (spliced rows or game rows) | `python scripts/train_dual_sequence.py ...` |
| `--val` | required | repeatable JSONL path | Validation input paths | same |
| `--side-mode` | `both` | `white`, `black`, `both` | Train selected side-specific model(s) | same |
| `--model-family` | `dual_side_sequence_lstm` | `dual_side_sequence_lstm`, `dual_side_sequence_board_lstm` | Select move-only or board-conditioned dual sequence architecture | same |
| `--horizon` | `8` | integer `>=1` | One-shot future sequence plies predicted per sample | same |
| `--epochs` | `20` | integer `>=1` | Training epochs | same |
| `--batch-size` | `64` | integer `>=1` | Batch size | same |
| `--lr` | `2e-4` | float `>0` | Adam learning rate | same |
| `--seed` | `7` | integer | Random seed for model/init/shuffle/runtime splice sampling | same |
| `--embed-dim` | `256` | integer `>=1` | Token/step embedding dimension | same |
| `--hidden-dim` | `512` | integer `>=1` | LSTM hidden size | same |
| `--num-layers` | `2` | integer `>=1` | Encoder/decoder LSTM layer count | same |
| `--dropout` | `0.15` | float `>=0` | Embedding/head + inter-layer LSTM dropout (`num_layers>1`) | same |
| `--step-loss-decay` | `0.9` | float (`0,1`) recommended; values `<=0` or `>=1` are normalized to `0.9` | Geometric per-ply loss weighting across horizon with earliest ply highest weight | same |
| `--side-to-move-feature` / `--no-side-to-move-feature` | enabled | boolean | Enable side-to-move conditioning in sequence decoder | same |
| `--side-to-move-embed-dim` | `4` | integer `>=1` | Side-to-move embedding dim when feature enabled | same |
| `--runtime-min-context` | `8` | integer `>=1` | Runtime splice min context for game-row inputs | same |
| `--runtime-min-target` | `1` | integer `>=1` | Runtime splice min target plies for game-row inputs | same |
| `--runtime-max-samples-per-game` | `0` | integer `>=0` | Runtime splice cap per game (`0` = no cap) | same |
| `--mate-bias` / `--no-mate-bias` | enabled | boolean | Enable endgame mate-in-x positive weighting in loss | same |
| `--mate-in-x` | `3` | integer `>=0` | Mate horizon threshold for mate-bias weight application | same |
| `--mate-weight` | `1.25` | float `>=1.0` | Multiplicative weight for mate-bias plies | same |
| `--num-workers` | `0` | integer `>=0` | Train DataLoader worker count | same |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` | Device selection; `auto` resolves to `cuda` when available else `cpu` | same |
| `--verbose` / `--no-verbose` | disabled | boolean | Emit per-epoch train/val summary logs | same |
| `--out-model-white` | `artifacts/model_white.pt` | filepath | White model artifact output | same |
| `--out-model-black` | `artifacts/model_black.pt` | filepath | Black model artifact output | same |
| `--out-metrics` | `artifacts/train_metrics_dual_sequence.json` | filepath | Combined run metrics output | same |

Dual-sequence interactions/precedence notes:
- Winner filtering is fixed by `--side-mode`; rows with non-matching `winner_side` are dropped automatically.
- Draw/unknown winner rows are dropped in current implementation.
- For game-row inputs, runtime splice controls determine sample counts before model-side winner filtering.
- Loss contract uses per-ply target probability distance (`1 - P(actual_move)`), then applies `step_loss_decay` and optional mate-bias multiplicatively.
- `--model-family=dual_side_sequence_board_lstm` enables board-state plane derivation from context (`C=18`) and board-conditioned decoder initialization.

## Inference CLI Dual Routing (`scripts/infer_move.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--model` | empty | artifact path | Single-artifact inference path (legacy next-move or dual-sequence artifact) | `python scripts/infer_move.py ...` |
| `--white-model` | empty | artifact path | White dual-sequence artifact path for side-routed inference | same |
| `--black-model` | empty | artifact path | Black dual-sequence artifact path for side-routed inference | same |
| `--sequence-decode-policy` | `sequence_path` | `sequence_path`, `step1_legal` | Dual-sequence first-move policy. `sequence_path` scores legal step-1 candidates via horizon continuation; `step1_legal` keeps legacy step-1-first legal selection | same |
| `--fallback-topk-multipliers` | `1,2,5` | comma-separated positive integers | Progressive top-k multipliers applied before hard no-legal fallback in next/rollout/sequence decode paths | same |

Inference dual-routing interactions:
- Caller must supply either:
  - `--model`, or
  - both `--white-model` and `--black-model`
- When both side artifacts are provided, side-to-move is inferred from context parity and selects the artifact.
- Progressive top-k attempts are `topk * multiplier` (clamped to vocab size), evaluated in order with deduplication.

## Play-vs-Model Server (`main.py`, `scripts/play_vs_model_server.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--model` | auto-injected latest artifact by `main.py` (or `artifacts/model.pt` when calling server directly) | artifact path | Single-artifact play mode | `python main.py ...`, `python scripts/play_vs_model_server.py ...` |
| `--white-model` | empty | artifact path | White-side artifact for dual pair side-routed play mode | same |
| `--black-model` | empty | artifact path | Black-side artifact for dual pair side-routed play mode | same |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` | Inference device for single or dual pair play runtime | same |
| `--winner-side` | `B` | `W`, `B`, `D`, `?` | Winner token conditioning for single-artifact inference paths | same |
| `--user-color` | `white` | `white`, `black` | Default player side in UI/API turn validation | same |
| `--topk` | `10` | integer `>=1` | Candidate width for model inference | same |
| `--piece-base` | `assets/pieces/cburnett` | URL path | Piece sprite path for browser board rendering | same |
| `--page-path` | `play-vs-model` | URL path fragment | Route for HTML app page | same |
| `--dir` | repo root via `main.py` wrapper | directory path | Static HTTP root for board assets/page | same |

Play-vs-model interactions/precedence notes:
- Model selection requires exactly one mode:
  - single artifact (`--model`), or
  - dual pair (`--white-model` + `--black-model`).
- `main.py` injects latest local `*.pt` as `--model` only when no model-selection flags are provided.
- UI/API `user_color` enforces side-to-move on user actions.
- Dual pair routing auto-selects white/black artifact by side-to-move (context parity) for each model reply.

## Model-vs-Model Arena CLI (`scripts/play_model_vs_model.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--model-a` | empty | artifact path or `latest` | Single-artifact path for participant A | `python scripts/play_model_vs_model.py ...` |
| `--model-b` | empty | artifact path or `latest` | Single-artifact path for participant B | same |
| `--model-a-white` | empty | artifact path | Optional white-side artifact for participant A dual-pair routing | same |
| `--model-a-black` | empty | artifact path | Optional black-side artifact for participant A dual-pair routing | same |
| `--model-b-white` | empty | artifact path | Optional white-side artifact for participant B dual-pair routing | same |
| `--model-b-black` | empty | artifact path | Optional black-side artifact for participant B dual-pair routing | same |
| `--alias-a` | `model_a` | string | Display alias for participant A in summary/PGN headers | same |
| `--alias-b` | `model_b` | string | Display alias for participant B in summary/PGN headers | same |
| `--games` | `20` | integer `>=1` | Number of games in the arena set | same |
| `--topk-a` | `10` | integer `>=1` | Candidate width for participant A inference | same |
| `--topk-b` | `10` | integer `>=1` | Candidate width for participant B inference | same |
| `--winner-side-a` | `W` | `W`, `B`, `D`, `?` | Winner token conditioning for participant A when using single-artifact inference | same |
| `--winner-side-b` | `W` | `W`, `B`, `D`, `?` | Winner token conditioning for participant B when using single-artifact inference | same |
| `--sequence-decode-policy-a` | `sequence_path` | `sequence_path`, `step1_legal` | Dual-sequence decode policy for participant A | same |
| `--sequence-decode-policy-b` | `sequence_path` | `sequence_path`, `step1_legal` | Dual-sequence decode policy for participant B | same |
| `--fallback-topk-multipliers-a` | `1,2,5` | comma-separated positive integers | Progressive top-k retry multipliers for participant A decode | same |
| `--fallback-topk-multipliers-b` | `1,2,5` | comma-separated positive integers | Progressive top-k retry multipliers for participant B decode | same |
| `--device-a` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` | Torch device for participant A | same |
| `--device-b` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` | Torch device for participant B | same |
| `--max-plies` | `300` | integer `>=1` | Hard ply cap per game | same |
| `--alternate-colors` / `--no-alternate-colors` | enabled | boolean | Alternate A/B colors between games or keep A as white | same |
| `--summary-out` | empty | filepath | Optional JSON summary output path | same |
| `--pgn-out` | empty | filepath | Optional PGN output path | same |
| `--progress` / `--no-progress` | enabled | boolean | Show/hide progress bar | same |
| `--verbose` / `--no-verbose` | enabled | boolean | Emit per-game/logging diagnostics | same |

Model-vs-model interactions/precedence notes:
- For each participant (`A`, `B`), caller must provide either:
  - one single artifact (`--model-a` or `--model-b`), or
  - both side artifacts (`--model-*-white` and `--model-*-black`).
- Side artifacts switch runtime mode to `dual_pair`; single artifact mode remains `single`.
- `winner-side-*` only conditions single-artifact inference; dual-pair routing ignores winner token and routes by side-to-move.
- `sequence-decode-policy-*` affects dual-pair participants only.
- `fallback-topk-multipliers-*` applies to both single-artifact and dual-pair participants before hard board-iterator fallback move selection.

## Pod Stop Control (`scripts/runpod_cycle_stop.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_STOP_REQUIRE_CONFIRMATION` | `0` | `0`, `1` | Optional gate for pod-stop mutation. Default `0` keeps auto-stop behavior; when set to `1` and confirmation is missing, stop is skipped and `STOP_SKIPPED_UNCONFIRMED` is recorded | `bash scripts/runpod_cycle_stop.sh` |
| `RUNPOD_STOP_CONFIRMATION` | unset | `YES` | Required confirmation token when `RUNPOD_STOP_REQUIRE_CONFIRMATION=1` | same |

Pod-stop interactions:
- full-flow wrappers that call `runpod_cycle_stop.sh` auto-stop by default only after artifact confirmation succeeds in full-HF flow.
- to require manual confirmation before stop, set `RUNPOD_STOP_REQUIRE_CONFIRMATION=1`; then explicit stop should include `RUNPOD_STOP_CONFIRMATION=YES`.

## Active-Pods Full Status (`scripts/runpod_active_pods_full_status.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--no-api` | disabled | flag | Skip RunPod REST enrichment | `bash scripts/runpod_active_pods_full_status.sh` |
| `--no-ssh` | disabled | flag | Skip SSH remote probes | same |
| `--running-only` | disabled | flag | Keep only `desiredStatus=RUNNING` pods (requires API) | same |
| `--no-write` | disabled | flag | Skip report file write under `artifacts/reports` | same |
| `RUNPOD_STATUS_SSH_TIMEOUT_SECONDS` | `12` | integer `>=1` | Timeout for per-pod SSH probe wrapper | same |
| `RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS` | `10` | integer `>=1` | SSH connect timeout | same |

## Vast No-Auth Deployment Smoke (`scripts/vast_local_smoke_test.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `VAST_SMOKE_RUN_ID` | `vast-local-smoke-<utc-ts>` | string | Run id used in artifact paths under `artifacts/vast_cycles/<run_id>/local_smoke` | `bash scripts/vast_local_smoke_test.sh` |
| `VAST_SMOKE_DATASET_DIR` | `data/dataset/_smoke_fast_game` | dataset dir path with `train.jsonl` + `val.jsonl` | Input dataset source for no-auth local smoke training | same |
| `VAST_SMOKE_VENV_DIR` | `.venv` under repo root | venv dir path | Python environment used by `deploy/vast_cloud_training/train_baseline_preset.sh` during local smoke | same |
| `VAST_SMOKE_EPOCHS` | `1` | integer `>=1` | Smoke epoch count forwarded via preset `TRAIN_EXTRA_ARGS` | same |
| `VAST_SMOKE_MAX_TOTAL_ROWS` | `512` | integer `>=1` | Effective total train+val row cap forwarded to `train_baseline.py --max-total-rows` | same |

Vast local smoke interactions:
- script always routes through `deploy/vast_cloud_training/train_baseline_preset.sh`.
- it injects fixed smoke args (`--batch-size 64 --num-workers 0 --no-progress`) plus configured epochs/max-row-cap.
- no Vast API/auth calls are made; `VAST_API_KEY` is not required.

## Vast No-Auth Deployment Checks (`scripts/vast_noauth_deploy_checks.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `VAST_NOAUTH_SKIP_LOCAL_SMOKE` | `0` | `0`, `1` | Skip or run the local smoke training step after test/connectivity checks | `bash scripts/vast_noauth_deploy_checks.sh` |

Vast no-auth checks interactions:
- always runs `python -m unittest discover -s tests -p "test_vast*.py" -v`.
- always runs `bash scripts/cloud_connectivity_health_checks.sh --provider vast --no-live`.
- when `VAST_NOAUTH_SKIP_LOCAL_SMOKE=0`, runs `bash scripts/vast_local_smoke_test.sh`.

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
  - benchmark matrix runtime max-samples auto-resolution contract, precision/sparsity trial defaults, and structured-2:4 trial wiring assertions.
- `tests/test_runpod_direct_api_guardrails.py`
  - direct-API-only guardrails for start/stop/full-smoke wiring and shared-script backend neutrality.
- `tests/test_runpod_sdk_guardrails.py`
  - SDK-only guardrails for SDK wrapper wiring and protection against accidental fallback to direct API entrypoints.
- `tests/test_runpod_container_openssh_flow.py`
  - container-specific OpenSSH bootstrap behavior in shared RunPod helpers (`apt-get openssh-client` path + keypair prep integration).
- `tests/test_training_precision_controls.py`
  - `scripts/train_baseline.py --help` exposes precision/sparsity controls (including `structured_2to4`) and autocast dtype resolver guardrails.
- `tests/test_train_dual_sequence_cli.py`
  - `scripts/train_dual_sequence.py --help` exposes dual-sequence controls and side-mode run wiring.
- `tests/test_play_model_vs_model_cli.py`
  - `scripts/play_model_vs_model.py --help` exposes dual-pair arena controls and runtime-mode wiring.
- `tests/test_vast_cycle_scripts.py`
  - Vast script contract checks include no-auth deployment smoke/check wrapper wiring assertions.
