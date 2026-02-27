# Chess Bot Training Runbook

## Responsibility
Provide direct, copy/paste commands for running training (full run and smoke run) against compact game datasets.

## Multi-Month Input Behavior
- You can repeat `--train` and `--val` to combine multiple month shards.
- Combined training/validation exposure is size-proportional to each file's indexed sample rows.
- Runtime behavior is unchanged: no month sampler, no file-level weighting flags, and no API changes.
- `artifacts/train_metrics*.json` records `train_inputs`/`val_inputs` plus `train_rows_by_file`/`val_rows_by_file` for per-file accounting.

## Top-Level Full Training Command (Low Batch)

Run from repo root:

```bash
.venv/bin/python scripts/train_baseline.py \
  --train data/dataset/elite_2025-11_game/train.jsonl \
  --val data/dataset/elite_2025-11_game/val.jsonl \
  --output artifacts/model_full_elite_2025-11_game.pt \
  --metrics-out artifacts/train_metrics_full_elite_2025-11_game.json \
  --progress-jsonl-out artifacts/train_progress_full_elite_2025-11_game.jsonl \
  --device auto \
  --epochs 20 \
  --batch-size 512 \
  --num-workers 8 \
  --amp \
  --runtime-min-context 8 \
  --runtime-min-target 1 \
  --runtime-max-samples-per-game 0 \
  --lr 2e-4 \
  --embed-dim 256 \
  --hidden-dim 512 \
  --num-layers 2 \
  --dropout 0.15 \
  --phase-feature \
  --side-to-move-feature \
  --lr-scheduler plateau \
  --lr-scheduler-metric val_loss \
  --lr-plateau-factor 0.5 \
  --lr-plateau-patience 3 \
  --lr-plateau-threshold 1e-4 \
  --early-stopping-patience 0 \
```

## Fast Smoke Command

```bash
.venv/bin/python scripts/train_baseline.py \
  --train data/dataset/_smoke_fast_game/train.jsonl \
  --val data/dataset/_smoke_fast_game/val.jsonl \
  --runtime-max-samples-per-game 1 \
  --epochs 1 \
  --batch-size 512 \
  --num-workers 0 \
  --output artifacts/model_smoke.pt \
  --metrics-out artifacts/train_metrics_smoke.json \
  --progress-jsonl-out artifacts/train_progress_smoke.jsonl \
  --no-progress
```

## Multi-Month Example (Repeated Flags)

```bash
.venv/bin/python scripts/train_baseline.py \
  --train data/dataset/elite_2025-09_game/train.jsonl \
  --train data/dataset/elite_2025-10_game/train.jsonl \
  --train data/dataset/elite_2025-11_game/train.jsonl \
  --val data/dataset/elite_2025-09_game/val.jsonl \
  --val data/dataset/elite_2025-10_game/val.jsonl \
  --val data/dataset/elite_2025-11_game/val.jsonl \
  --output artifacts/model_full_elite_2025-09_to_2025-11_game.pt \
  --metrics-out artifacts/train_metrics_full_elite_2025-09_to_2025-11_game.json \
  --device auto \
  --epochs 40 \
  --batch-size 2048 \
  --num-workers 8 \
  --amp \
  --no-progress
```

## Multi-GPU (Single Node, Torchrun/DDP)

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_baseline.py \
  --train data/dataset/elite_2025-11_game/train.jsonl \
  --val data/dataset/elite_2025-11_game/val.jsonl \
  --output artifacts/model_full_elite_2025-11_game_ddp.pt \
  --metrics-out artifacts/train_metrics_full_elite_2025-11_game_ddp.json \
  --progress-jsonl-out artifacts/train_progress_full_elite_2025-11_game_ddp.jsonl \
  --distributed auto \
  --device auto \
  --epochs 20 \
  --batch-size 512 \
  --num-workers 8 \
  --amp \
  --runtime-min-context 8 \
  --runtime-min-target 1 \
  --runtime-max-samples-per-game 0
```

Notes:
- Keep `--distributed auto` for backward compatibility; the same command still works in single-process mode when not launched with `torchrun`.
- Only rank 0 writes final model/metrics/progress files.

## Dataset Expectations
- Input format: compact game dataset (`game_jsonl_runtime_splice_v1`)
- Required files:
  - `train.jsonl`
  - `val.jsonl`
- Optional but recommended:
  - `runtime_splice_cache/manifest.json` and split cache binaries (`train`, `val`)

## Runtime Cache Behavior
- Training auto-loads compatible runtime splice caches when present.
- If cache is missing or config-mismatched, training falls back to runtime index building from JSONL.

## Output Artifacts
- Model: path passed to `--output`
- Metrics: path passed to `--metrics-out`
- Progress stream: path passed to `--progress-jsonl-out`
