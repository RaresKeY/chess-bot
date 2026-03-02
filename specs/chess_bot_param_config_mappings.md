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
- `scripts/container_ensure_openssh.sh`
- `scripts/runpod_cycle_full_train_hf.sh`
- `scripts/runpod_full_train_easy.sh`
- `scripts/runpod_full_train_easy_smoke_test.sh`
- `scripts/runpod_active_pods_full_status.sh`
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
| `RUNPOD_CLOUD_TYPE` | start-script default (`COMMUNITY`) unless overridden | `SECURE`, `COMMUNITY` | Cloud tier for provisioning | same |
| `RUNPOD_INTERRUPTIBLE` | `0` | `0`, `1` | Spot/interruptible request via provision helper | same |
| `RUNPOD_BENCH_TRIALS` | `fp32,tf32,fp16,bf16,sparsity,bf16_2to4` | comma-separated trial list (`fp32`, `tf32`, `fp16`, `bf16`, `fp32_sparse`, `fp16_sparse`, `bf16_sparse`, `sparsity`, `fp16_2to4`, `bf16_2to4`) | Precision/sparsity trial matrix; `*_sparse`/`sparsity` use L1 penalty, `*_2to4` uses structured 2:4 sparsity mode | same |
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

Explicit-GPU provisioning behavior note (current):
- when `--gpu-type-id` / `RUNPOD_GPU_TYPE_ID` is set, provision now performs a GraphQL stock preflight (`gpuTypes(input:{id})` + `lowestPrice`) and fails fast if the requested `gpu_count` is not listed in `availableGpuCounts` or stock status is out-of-stock/no-capacity.
- this is an existing-control behavior refinement (no new runtime flag), intended to improve resource-availability determination before REST pod create attempts.

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
| `--image-name` (`provision`) | `""` | docker image reference | Explicit image for SDK create call; required by some SDK variants when no `template_id` is set | same |
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
| `RUNPOD_SDK_IMAGE_NAME` | unset | docker image reference | SDK-specific image override forwarded to `--image-name` in SDK start wrapper | same |
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

## Full-Train Wrappers (`scripts/runpod_full_train_easy.sh`, `scripts/runpod_full_train_easy_smoke_test.sh`, `scripts/runpod_cycle_full_train_hf.sh`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `RUNPOD_HF_DATASET_REPO_ID` | project default | HF repo id | Source dataset repo for remote fetch | `bash scripts/runpod_full_train_easy.sh` |
| `RUNPOD_HF_DATASET_PATH_PREFIX` | `validated_datasets` | repo prefix | Dataset path root in HF repo | same |
| `RUNPOD_HF_DATASET_SCHEMA_FILTER` | `game_jsonl_runtime_splice_v1` | schema id string | Chooses dataset format from HF manifests | same |
| `RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS` | `5000` (smoke wrapper) / unset-`0` (easy/full-HF wrappers) | integer `>=0` | Training subset cap; smoke wrapper now defaults to a 5K-row quick run (`~4445` train + `~555` val by split ratio), while easy/full-HF keep uncapped behavior unless overridden | same |
| `RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME` | `auto` (smoke wrapper) / `0` (full-HF flow default) | `auto` or integer `>=0` | Runtime splice per-game cap for game-format datasets. When set to `auto` with cache-required mode, full-HF flow resolves cache-matching runtime config from fetched `runtime_splice_cache/manifest.json` before training | same |
| `RUNPOD_FULL_TRAIN_NPROC_PER_NODE` | `${RUNPOD_GPU_COUNT}` | integer `>=1` | Torchrun process count; smoke wrapper defaults to one process per requested GPU (`2`) | same |
| `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` | unset (smoke wrapper auto mode) | integer `>=0` | Optional manual DataLoader workers-per-rank override; when unset in smoke wrapper, auto worker policy is used | same |
| `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE` | `1` in HF flow | `0`, `1` | Force cache-only runtime splice indexing | same |
| `RUNPOD_CYCLE_RUN_ID` | `easy-smoke-<utc-ts>` (smoke wrapper) / shared helper default | string | Per-run artifact/telemetry directory id forwarded through easy/full-train flow | `bash scripts/runpod_full_train_easy_smoke_test.sh` |
| `RUNPOD_GPU_TYPE_ID` | `NVIDIA GeForce RTX 5090` (smoke wrapper) / `NVIDIA GeForce RTX 5090` (easy wrapper) | RunPod GPU type id/display name | Default GPU selection for both easy and smoke wrappers; callers can override per run when targeting specific capacity/cost tiers | `bash scripts/runpod_full_train_easy_smoke_test.sh`, `bash scripts/runpod_full_train_easy.sh` |
| `RUNPOD_GPU_COUNT` | `2` (smoke wrapper/easy wrapper) | integer `>=1` | Multi-GPU pod count; smoke wrapper now defaults to 2-GPU validation runs | same |
| `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE` | unset (smoke wrapper auto mode) | integer `>=1` | Optional manual batch override. Smoke wrapper unsets this by default so full-HF auto batch-attempt policy drives selection | same |
| `RUNPOD_SMOKE_EVENT_LOG` | `artifacts/runpod_cycles/<run_id>/logs/full_smoke_events.log` | filepath | Timestamped full stdout/stderr event stream for the smoke wrapper (`awk` UTC prefix + `tee`) | `bash scripts/runpod_full_train_easy_smoke_test.sh` |
| `RUNPOD_SMOKE_SUMMARY_JSON` | `artifacts/runpod_cycles/<run_id>/reports/smoke_summary.json` | filepath | Post-run machine-readable smoke summary (timing, metrics tail, progress tail, GPU sample stats) generated by smoke wrapper | `bash scripts/runpod_full_train_easy_smoke_test.sh` |

Full-train runtime splice interactions:
- `TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1` remains enforced in the full-HF flow.
- when `RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME=auto`, the flow resolves runtime splice config from the fetched dataset cache manifest (`runtime_splice_cache/manifest.json`) and uses those resolved values for `TRAIN_RUNTIME_MIN_CONTEXT`, `TRAIN_RUNTIME_MIN_TARGET`, and `TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME` before launching training.
- smoke-wrapper auto-batch precedence:
  - if `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE` / `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE` are unset in caller env, smoke wrapper intentionally unsets them so full-HF auto selection remains active.
  - if caller provides either override env var, smoke wrapper preserves it and forwards it unchanged to `runpod_full_train_easy.sh`.

## Training CLI (`scripts/train_baseline.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--amp` / `--no-amp` | enabled in presets | boolean | Mixed precision enable/disable | `python scripts/train_baseline.py ...` |
| `--amp-dtype` | `auto` | `auto`, `fp16`, `bf16` | Autocast dtype selection | same |
| `--tf32` | `auto` | `auto`, `on`, `off` | TensorFloat32 matmul/cuDNN controls | same |
| `--distributed-backend` | `nccl` in multi-GPU runs | backend id | DDP backend | same |
| `--runtime-max-samples-per-game` | runtime-dependent | integer `>=0` | Runtime splice cap; must match cache config when cache-required | same |
| `--require-runtime-splice-cache` | often enabled in cloud HF flows | boolean | Fail on cache miss/mismatch instead of runtime indexing | same |
| `--max-total-rows` | `0` | integer `>=0` | Row cap for fast subset tests | same |
| `--sparsity-mode` | `off` | `off`, `l1`, `structured_2to4` | Sparsity behavior: `l1` adds L1 penalty, `structured_2to4` enforces persistent 2:4 masks on linear weights | same |
| `--sparsity-l1-lambda` | `0.0` | float `>=0` | L1 penalty multiplier when `--sparsity-mode=l1` | same |
| `--sparsity-include-bias` | `False` | boolean flag | Include bias tensors in L1/stat tracking (applies to `l1` mode) | same |

## Dual Sequence Training CLI (`scripts/train_dual_sequence.py`)
| Control | Default | Accepted | Effect | Related Command |
|---|---|---|---|---|
| `--train` | required | repeatable JSONL path | Training input paths (spliced rows or game rows) | `python scripts/train_dual_sequence.py ...` |
| `--val` | required | repeatable JSONL path | Validation input paths | same |
| `--side-mode` | `both` | `white`, `black`, `both` | Train selected side-specific model(s) | same |
| `--horizon` | `8` | integer `>=1` | One-shot future sequence plies predicted per sample | same |
| `--epochs` | `20` | integer `>=1` | Training epochs | same |
| `--batch-size` | `64` | integer `>=1` | Batch size | same |
| `--lr` | `2e-4` | float `>0` | Adam learning rate | same |
| `--seed` | `7` | integer | Random seed for model/init/shuffle/runtime splice sampling | same |
| `--embed-dim` | `256` | integer `>=1` | Token/step embedding dimension | same |
| `--hidden-dim` | `512` | integer `>=1` | LSTM hidden size | same |
| `--num-layers` | `2` | integer `>=1` | Encoder/decoder LSTM layer count | same |
| `--dropout` | `0.15` | float `>=0` | Embedding/head + inter-layer LSTM dropout (`num_layers>1`) | same |
| `--step-loss-decay` | `1.0` | float `>0` | Geometric per-ply loss weighting across horizon | same |
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
- full-flow wrappers that call `runpod_cycle_stop.sh` stop compute by default (auto-stop).
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
