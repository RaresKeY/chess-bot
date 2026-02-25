# Chess Bot Training Component

## Responsibility
Train a baseline winner-aware next-move predictor from splice samples and save a reusable model artifact.

## Code Ownership
- CLI: `scripts/train_baseline.py`
- Core logic: `src/chessbot/training.py`
- Model definition: `src/chessbot/model.py`
- Shared IO: `src/chessbot/io_utils.py`

## Model (current baseline)
- Token embedding over UCI move vocabulary
- Configurable-depth LSTM sequence encoder over context moves (`--num-layers`)
- Configurable dropout (`--dropout`) applied to embedding/head; also used as LSTM inter-layer dropout when `num_layers > 1`
- Optional winner embedding concatenated before classifier head
- Cross-entropy next-move objective

## Winner-Aware Behavior
- Winner side encoded (`W`, `B`, `D`, `?`)
- Winner examples (`W`/`B`) receive configurable loss upweighting (`--winner-weight`)

## Runtime / Device Controls
- `scripts/train_baseline.py` supports explicit `--device` (`auto`, `cpu`, `cuda`, `cuda:N`)
- CUDA request fails fast if `torch.cuda.is_available()` is false
- CUDA-oriented controls:
  - `--num-workers` (DataLoader workers)
  - `--pin-memory/--no-pin-memory` (auto-disabled on CPU)
  - `--amp/--no-amp` (CUDA mixed precision via `torch.amp`)
- Model/training controls:
  - `--num-layers` (LSTM layer count)
  - `--dropout` (embedding/head dropout and LSTM inter-layer dropout when multilayer)
  - `--restore-best/--no-restore-best` (restore best validation checkpoint before saving)
  - `--verbose/--no-verbose` (toggle startup/epoch/checkpoint logs)
  - `--progress/--no-progress` (toggle per-epoch batch progress bar; useful with `--verbose`)
- Memory/loader safeguards:
  - train DataLoader disables `persistent_workers` and uses reduced prefetch (`prefetch_factor=1`) when `--num-workers > 0`
  - validation DataLoader runs single-process (`num_workers=0`) to avoid a second worker pool and reduce host RAM growth
- CLI prints a small CUDA preflight summary (`torch` version, CUDA availability, device count, `CUDA_VISIBLE_DEVICES`)
- CLI can emit verbose startup logs including resolved input/output paths, requested hyperparameters, and loaded dataset row counts
- Core training loop can emit epoch start/end summaries, a per-epoch batch progress bar, and best-checkpoint update/restore messages

## Output Artifact Contract
`artifacts/model.pt` stores:
- `state_dict`
- `vocab`
- `config` (`embed_dim`, `hidden_dim`, `num_layers`, `dropout`, `use_winner`)
- `runtime` (`device`, `amp`, `best_checkpoint`) from the training run

## Metrics Output
`artifacts/train_metrics.json` stores:
- dataset row counts
- epoch history (train_loss, val_loss, top1, top5)
- model path
- runtime request fields (`device_requested`, `num_layers`, `dropout`, `num_workers`, `pin_memory`, `amp`, `restore_best`, `verbose`, `progress`)
- best-checkpoint summary copied from artifact runtime metadata

## Current Limitation
- CLI still loads `train`/`val` JSONL fully into memory before training starts (not yet streaming/iterable).
