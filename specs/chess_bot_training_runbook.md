# Chess Bot Training Runbook

## Responsibility
Provide direct, copy/paste commands for running training (full run and smoke run) against compact game datasets.

## Top-Level Full Training Command

Run from repo root:

```bash
.venv/bin/python scripts/train_baseline.py \
  --train data/dataset/elite_2025-11_game/train.jsonl \
  --val data/dataset/elite_2025-11_game/val.jsonl \
  --output artifacts/model_full_elite_2025-11_game.pt \
  --metrics-out artifacts/train_metrics_full_elite_2025-11_game.json \
  --progress-jsonl-out artifacts/train_progress_full_elite_2025-11_game.jsonl \
  --device auto \
  --epochs 100 \
  --batch-size 2048 \
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
  --no-progress
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
