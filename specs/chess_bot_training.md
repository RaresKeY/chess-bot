# Chess Bot Training Component

## Responsibility
Train a baseline winner-aware next-move predictor from splice samples and save a reusable model artifact.

## Dataset Inputs
- CLI accepts one or more `--train` JSONL paths and one or more `--val` JSONL paths (repeatable flags)
- Repeated `--train` / `--val` inputs are combined via file-backed JSONL indexing (line offsets) instead of concatenating row dicts in memory
- Existing single-path usage remains supported (one `--train`, one `--val`)

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
- Optional phase embedding concatenated before classifier head (`phase` label from splice rows)
- Optional side-to-move embedding concatenated before classifier head (derived from context ply parity)
- Cross-entropy next-move objective

## Winner-Aware Behavior
- Winner side encoded (`W`, `B`, `D`, `?`)
- Winner examples (`W`/`B`) receive configurable loss upweighting (`--winner-weight`)

## Phase-Aware Behavior (optional)
- Training reads splice-row `phase` labels when present (`opening`, `middlegame`, `endgame`; fallback `unknown`)
- CLI exposes per-phase loss multipliers:
  - `--phase-weight-opening`
  - `--phase-weight-middlegame`
  - `--phase-weight-endgame`
  - `--phase-weight-unknown`
- Effective per-example loss weight is multiplicative: `winner_weight_component * phase_weight_component`
- Default phase weights are `1.0`, preserving prior training behavior unless explicitly changed

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
  - `--phase-feature/--no-phase-feature`, `--phase-embed-dim`
  - `--side-to-move-feature/--no-side-to-move-feature`, `--side-to-move-embed-dim`
  - `--restore-best/--no-restore-best` (restore best validation checkpoint before saving)
  - `--lr-scheduler {none,plateau}` and plateau controls (`--lr-scheduler-metric`, `--lr-plateau-factor`, `--lr-plateau-patience`, `--lr-plateau-threshold`, `--lr-plateau-min-lr`)
  - `--early-stopping-patience`, `--early-stopping-metric`, `--early-stopping-min-delta`
  - `--verbose/--no-verbose` (toggle startup/epoch/checkpoint logs)
  - `--progress/--no-progress` (toggle per-epoch batch progress bar; useful with `--verbose`)
- CLI defaults (current tuned baseline preset):
  - `epochs=40`, `lr=2e-4`
  - `embed_dim=256`, `hidden_dim=512`, `num_layers=2`, `dropout=0.15`
  - `lr_scheduler=plateau` remains default; early stopping remains opt-in (`patience=0`)
- Memory/loader safeguards:
  - CLI streams train JSONL files to build vocabulary/counts, indexes train/val JSONL line offsets, and loads rows on-demand in `Dataset.__getitem__` (reduces host RAM vs eager row loading)
  - train DataLoader disables `persistent_workers` and uses reduced prefetch (`prefetch_factor=1`) when `--num-workers > 0`
  - validation DataLoader runs single-process (`num_workers=0`) to avoid a second worker pool and reduce host RAM growth
- CLI prints a small CUDA preflight summary (`torch` version, CUDA availability, device count, `CUDA_VISIBLE_DEVICES`)
- CLI can emit verbose startup logs including resolved input/output paths, requested hyperparameters, and loaded dataset row counts
- Verbose logs include per-file train/val input lists and loaded row counts per input file when multiple datasets are provided
- Core training loop can emit epoch start/end summaries, a per-epoch batch progress bar, and best-checkpoint update/restore messages

## Output Artifact Contract
`artifacts/model.pt` stores:
- `state_dict`
- `vocab`
- `config` (`embed_dim`, `hidden_dim`, `num_layers`, `dropout`, `use_winner`, `use_phase`, `phase_embed_dim`, `use_side_to_move`, `side_to_move_embed_dim`)
- `runtime` (`device`, `amp`, `best_checkpoint`, `early_stopping`, `lr_scheduler`) from the training run
- `runtime.phase_weights` (resolved per-phase multipliers used during training)

## Metrics Output
`artifacts/train_metrics.json` stores:
- dataset row counts
- train/val input file lists and row counts per input file (when provided)
- `data_loading` mode metadata (currently `indexed_jsonl_on_demand`)
- epoch history (train_loss, val_loss, top1, top5)
- model path
- runtime request fields (`device_requested`, `num_layers`, `dropout`, `num_workers`, `pin_memory`, `amp`, `restore_best`, `verbose`, `progress`)
- requested `phase_weights`
- feature-toggle and embedding-dim settings for phase/side-to-move head inputs
- LR scheduler request fields and runtime scheduler summary (including final LR)
- early-stopping request fields and runtime early-stopping summary
- best-checkpoint summary copied from artifact runtime metadata

## Current Limitation
- Training still stores per-row JSONL line offsets in RAM (much smaller than full row dicts/strings, but not fully streaming/iterable training).

## Regression Tests (current)
- `tests/test_training_features.py` covers:
  - side-to-move ID derivation parity/fallback
  - `collate_train()` phase + side-to-move tensor outputs
  - `NextMoveLSTM` forward path with phase/side feature head inputs enabled
  - scheduler/early-stopping runtime metadata and early-stop behavior in a tiny synthetic training run
