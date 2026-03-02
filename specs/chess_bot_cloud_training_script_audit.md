# Chess Bot Cloud Training Script Audit

## Purpose
Provide a code-truth audit of cloud training scripts (RunPod + Vast), including scope, freshness (mtime + last commit), and whether each script is aligned with the current training regime.

## Audit Date
- 2026-03-02 (UTC)

## Method
- Inventory and grep scan over:
  - `scripts/runpod*.sh`
  - `scripts/vast*.sh`
  - `deploy/runpod_cloud_training/*.sh`
  - `deploy/vast_cloud_training/*.sh`
- Verified launch/invocation lines directly in script code.
- Recorded file mtime (`stat`) and last git commit (`git log -1`).

## Current Regime (Code Truth)
- Cloud training scripts currently launch `scripts/train_baseline.py` (baseline next-move training).
- No cloud training launcher currently invokes `scripts/train_dual_sequence.py`.
- Cloud launchers now default to forced multistep baseline output (`--rollout-horizon 8`, `--closeness-horizon 8`) via preset/runtime controls.

## Inventory + Status

| Script | Scope | mtime (UTC) | Last Commit | Code Truth | Status |
|---|---|---|---|---|---|
| `scripts/runpod_full_train_easy.sh` | top-level easy full-train wrapper | 2026-02-28 06:48:27 | 2026-02-28T08:53:45+02:00 `d4d52a3` | Sets env defaults and delegates to full HF flow, including forced baseline multistep defaults (`RUNPOD_FULL_TRAIN_ROLLOUT_HORIZON=8`, closeness inherited). | current |
| `scripts/runpod_cycle_full_train_hf.sh` | primary full RunPod lifecycle + remote training launcher | 2026-03-02 09:48:11 | 2026-03-02T11:48:27+02:00 `bbfbdd5` | Launches `train_baseline_preset.sh` (or direct `train_baseline.py` fallback), supports DDP (`torchrun`), cache-required runtime splice, batch retry ladder, and forwards rollout/closeness horizons (default 8/8). | current |
| `deploy/runpod_cloud_training/train_baseline_preset.sh` | in-container training preset | 2026-02-27 10:36:41 | 2026-02-27T12:48:28+02:00 `96a266e` | Trains baseline model, supports HF aggregate inputs, schema-aware path selection, repeated `--train/--val`, runtime splice/cache flags, optional `torchrun`, and rollout/closeness controls (default 8/8). | current |
| `scripts/runpod_cycle_watch_progress.sh` | remote progress watcher + checkpoint sync | 2026-02-27 08:47:24 | 2026-02-27T10:48:17+02:00 `70b78c7` | Monitors progress JSONL/stdout, tracks exit sentinel, syncs best/epoch checkpoints, emits local ETA report. | current (monitor-only) |
| `scripts/runpod_cycle_train.sh` | smoke train orchestration lane | 2026-03-01 11:08:56 | 2026-03-01T13:09:51+02:00 `5ad0c09` | Runs remote `train_baseline_preset.sh` and smoke inference check, forwarding rollout/closeness defaults (`8/8`). Optional HF aggregate mode. | current |
| `scripts/runpod_cycle_full_smoke.sh` | smoke E2E wrapper (start/push/train/collect/stop) | 2026-02-28 06:48:19 | 2026-02-28T08:53:45+02:00 `d4d52a3` | Wraps smoke lifecycle and delegates training to `runpod_cycle_train.sh`. | current |
| `scripts/runpod_sdk_cycle_full_smoke.sh` | SDK-based smoke E2E wrapper | 2026-03-01 05:18:22 | 2026-03-01T07:21:06+02:00 `6614963` | Same smoke behavior as full smoke, with SDK start/stop wrappers. | current |
| `scripts/runpod_full_train_easy_smoke_test.sh` | easy-flow smoke with verify + terminate | 2026-03-02 09:37:15 | 2026-03-02T11:43:34+02:00 `72cbcb6` | Uses full easy flow with smoke-safe overrides (`epochs=1`, row cap, `runtime_max_samples_per_game=auto`) plus rollout/closeness defaults (`8/8`), then verifies artifacts and termination. | current |
| `scripts/runpod_local_smoke_test.sh` | local Docker smoke harness for RunPod module | 2026-02-28 06:23:05 | 2026-02-28T08:28:27+02:00 `fa1db00` | Launches containerized `train_baseline_preset.sh` with tiny dataset, timing logs, and rollout/closeness smoke defaults (`8/8`). | current |
| `scripts/runpod_cycle_benchmark_matrix.sh` | benchmark matrix/trials lane | 2026-03-01 05:35:32 | 2026-03-01T07:52:54Z `f0eecf1` | Runs multi-trial baseline training with precision variants, rollout/closeness defaults (`8/8`), NCCL fallback logic, and sparse-trial skip guard when rollout horizon >1. | current |
| `scripts/runpod_cycle_benchmark_10k_sixpack.sh` | benchmark preset wrapper | 2026-02-28 11:16:20 | 2026-02-28T13:17:29+02:00 `562546f` | Preconfigures benchmark matrix env and invokes `runpod_cycle_benchmark_matrix.sh`. | current |
| `deploy/vast_cloud_training/train_baseline_preset.sh` | Vast training preset | 2026-02-27 12:29:11 | 2026-02-27T14:39:00+02:00 `a711fec` | Calls `scripts/train_baseline.py` using `--output` and forwards rollout/closeness defaults (`8/8`). | current |
| `scripts/vast_cycle_start.sh` | Vast provision/start lifecycle | 2026-02-27 12:28:49 | 2026-02-27T14:39:00+02:00 `a711fec` | Provisioning only, no training launch. | out of training scope |
| `scripts/vast_cycle_status.sh` | Vast status lifecycle | 2026-02-27 12:28:49 | 2026-02-27T14:39:00+02:00 `a711fec` | Status only, no training launch. | out of training scope |
| `scripts/vast_cycle_stop.sh` | Vast stop lifecycle | 2026-02-27 12:28:49 | 2026-02-27T14:39:00+02:00 `a711fec` | Stop only, no training launch. | out of training scope |
| `scripts/vast_cycle_terminate.sh` | Vast terminate lifecycle | 2026-02-27 12:28:49 | 2026-02-27T14:39:00+02:00 `a711fec` | Terminate only, no training launch. | out of training scope |

## Key Drift / Risk Findings

1. RunPod easy wrapper defaults differ from preferred flow assumptions:
   - Code defaults: `epochs=100`, schema filter `auto`.
   - Preferred operator flow may still pin different values via explicit env exports.

2. Cloud launchers now enforce baseline multistep 8-ply rollout, but still do not use dual-sequence trainers:
   - No cloud launcher calls `scripts/train_dual_sequence.py`.

## Immediate Recommendations

1. Keep canonical cloud regime explicit in docs/specs:
   - current: baseline multistep (`--rollout-horizon 8`)
   - future option: dual-sequence (`train_dual_sequence.py`)

2. Align RunPod defaults and preferred-flow docs where still intentionally different:
   - either update wrapper defaults to match preferred flow
   - or keep wrapper defaults and update preferred-flow spec language as "recommended exports".

3. Add/refresh a Vast training smoke test lane that exercises the preset script directly.
