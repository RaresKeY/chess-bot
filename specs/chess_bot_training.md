# Chess Bot Training Component

## Responsibility
Train a baseline winner-aware next-move predictor from splice samples and save a reusable model artifact.

## Dataset Inputs
- CLI accepts one or more `--train` JSONL paths and one or more `--val` JSONL paths (repeatable flags)
- Training auto-detects input schema per path set:
  - legacy splice-row JSONL (`context` + `target` + `next_move`)
  - compact game-level JSONL (`moves` or `moves_uci`) with runtime splicing
- Repeated `--train` / `--val` inputs are combined via file-backed JSONL indexing (line offsets) instead of concatenating row dicts in memory
- Multi-month repeated-input behavior is size-proportional by row count/sample count: larger monthly files naturally contribute more training/validation examples because all indexed rows are consumed uniformly (no extra month-level weighting/sampling layer)
- Existing single-path usage remains supported (one `--train`, one `--val`)
- For compact game-level datasets, training builds a sample index of `(game row offset, splice_index)` and derives `context`/`target` at runtime

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

## Training Objective Modes (current)
- Default compatibility mode remains single-step next-move training (`rollout_horizon=1`)
- Optional multistep mode is enabled by setting `--rollout-horizon > 1`
  - current implementation uses teacher-forced recursive rollout training (repeated next-move predictions)
  - loss is a weighted sum of per-step cross-entropy terms across the rollout horizon
  - rollout loss weights decay geometrically from step 1 using `--rollout-loss-decay` (step 1 starts at weight `1.0`)
  - continuation "closeness" metrics are reported over the first `--closeness-horizon` rollout plies (clamped to rollout horizon)

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
  - `--rollout-horizon` (future plies predicted during training objective; `1` preserves baseline behavior)
  - `--closeness-horizon` (validation continuation-closeness horizon; clamped to rollout horizon)
  - `--rollout-loss-decay` (geometric decay factor for multistep rollout loss weights)
  - runtime splicing controls for compact game datasets:
    - `--runtime-min-context`
    - `--runtime-min-target`
    - `--runtime-max-samples-per-game` (0 = no cap)
    - `--require-runtime-splice-cache/--no-require-runtime-splice-cache` (when enabled, game-dataset training fails if runtime cache cannot be loaded; no runtime index fallback)
  - `--num-layers` (LSTM layer count)
  - `--dropout` (embedding/head dropout and LSTM inter-layer dropout when multilayer)
  - `--phase-feature/--no-phase-feature`, `--phase-embed-dim`
  - `--side-to-move-feature/--no-side-to-move-feature`, `--side-to-move-embed-dim`
  - `--restore-best/--no-restore-best` (restore best validation checkpoint before saving)
  - `--lr-scheduler {none,plateau}` and plateau controls (`--lr-scheduler-metric`, `--lr-plateau-factor`, `--lr-plateau-patience`, `--lr-plateau-threshold`, `--lr-plateau-min-lr`)
  - `--early-stopping-patience`, `--early-stopping-metric`, `--early-stopping-min-delta`
  - `--verbose/--no-verbose` (toggle startup/epoch/checkpoint logs)
  - `--progress/--no-progress` (toggle per-epoch batch progress bar; useful with `--verbose`)
  - `--progress-jsonl-out` (optional epoch-level machine-readable JSONL event stream for external monitoring/polling)
- CLI defaults (current tuned baseline preset):
  - `epochs=40`, `lr=2e-4`
  - `embed_dim=256`, `hidden_dim=512`, `num_layers=2`, `dropout=0.15`
  - `lr_scheduler=plateau` remains default; early stopping remains opt-in (`patience=0`)
- Memory/loader safeguards:
  - legacy splice-row mode: streams train JSONL to build vocabulary/counts, indexes train/val line offsets, loads rows on-demand in `Dataset.__getitem__`
  - compact game-level mode: streams train JSONL to build vocabulary and runtime-splice sample counts, builds sample index `(game offset, splice_i)`, loads game rows on-demand and splices per sample in `Dataset.__getitem__`
  - when compatible `runtime_splice_cache` artifacts are present for game-level train/val paths, training loads precomputed packed indexes from cache instead of rebuilding them from JSONL
  - if cache files are missing/mismatched (for example runtime splice config mismatch), training safely falls back to runtime JSONL indexing
  - when `--require-runtime-splice-cache` is enabled for game datasets, training checks cache-loadability first and raises an error on cache miss/mismatch instead of falling back
  - train DataLoader disables `persistent_workers` and uses reduced prefetch (`prefetch_factor=1`) when `--num-workers > 0`
  - validation DataLoader runs single-process (`num_workers=0`) to avoid a second worker pool and reduce host RAM growth
- CLI prints a small CUDA preflight summary (`torch` version, CUDA availability, device count, `CUDA_VISIBLE_DEVICES`)
- CLI can emit verbose startup logs including resolved input/output paths, requested hyperparameters, and loaded dataset row counts
- Verbose logs include per-file train/val input lists and loaded row counts per input file when multiple datasets are provided
- Core training loop can emit epoch start/end summaries, a per-epoch batch progress bar, and best-checkpoint update/restore messages
- When `--progress-jsonl-out` is set, the CLI appends JSONL events (for example `script_start`, `train_setup`, `epoch_start`, `epoch_end`, `train_complete`, `script_complete`) with timestamps for host-side progress polling

## Output Artifact Contract
`artifacts/model.pt` stores:
- root metadata for inference compatibility dispatch:
  - `artifact_format_version` (current writer uses `2`)
  - `model_family` (current baseline family: `next_move_lstm`)
  - `training_objective` (root copy; e.g. `single_step_next_move` or multistep objective)
- `state_dict`
- `vocab`
- `config` (`embed_dim`, `hidden_dim`, `num_layers`, `dropout`, `use_winner`, `use_phase`, `phase_embed_dim`, `use_side_to_move`, `side_to_move_embed_dim`)
- `runtime` (`device`, `amp`, `best_checkpoint`, `early_stopping`, `lr_scheduler`) from the training run
- `runtime.training_objective` and, in multistep mode, rollout settings (`rollout_horizon`, `closeness_horizon`, `rollout_loss_decay`, resolved `rollout_loss_weights`)
- `runtime.phase_weights` (resolved per-phase multipliers used during training)

## Metrics Output
`artifacts/train_metrics.json` stores:
- dataset row counts
- train/val input file lists and row counts per input file (when provided)
- `data_loading` mode metadata (currently `indexed_jsonl_on_demand`)
- `dataset_schema` metadata (`spliced` or `game`)
- in compact game-level mode, metrics also include game counts (`train_games`, `val_games`) and `runtime_splice` settings used during indexing
- in compact game-level mode, metrics include `cache_load_reason_by_split` with per-split cache status (`hit`) or fallback reason string (for example `cache_config_mismatch`, `cache_file_missing:...`)
- epoch history (train_loss, val_loss, top1, top5)
- when multistep mode is enabled (`rollout_horizon > 1`), epoch history additionally includes rollout metrics such as `rollout_step{n}_acc`, `rollout_prefix_match_len_avg`, `rollout_legal_rate`, and `rollout_weighted_continuation_score`
- model path
- runtime request fields (`device_requested`, `num_layers`, `dropout`, `num_workers`, `pin_memory`, `amp`, `restore_best`, `verbose`, `progress`)
- requested `phase_weights`
- feature-toggle and embedding-dim settings for phase/side-to-move head inputs
- LR scheduler request fields and runtime scheduler summary (including final LR)
- early-stopping request fields and runtime early-stopping summary
- best-checkpoint summary copied from artifact runtime metadata
- training-objective metadata (`training_objective`, rollout horizon/closeness/decay and resolved rollout loss weights when multistep mode is used)

## Optional Progress JSONL Output
- `--progress-jsonl-out <path>` writes newline-delimited JSON events for long-running training observability without scraping stdout
- Event rows include `ts_epoch_ms` and an `event` field; epoch-end rows include metric snapshots (`train_loss`, `val_loss`, `top1`, `top5`, `lr`)
- in multistep mode, epoch-end progress events also include rollout summary metrics (for example `rollout_step4_acc`, `rollout_legal_rate`, `rollout_weighted_continuation_score`)
- Intended use case: remote/RunPod training where a host-side watcher polls the file and renders a local progress bar

## Current Limitation
- Training still stores per-row JSONL line offsets in RAM (much smaller than full row dicts/strings, but not fully streaming/iterable training).
- Compact game-level runtime splicing still computes phase labels in-loader for fallback/no-cache paths; precomputed cache paths avoid index-build replay work but do not eliminate per-sample row decode/splice cost.
- Runtime-splice optimization (current):
  - training now caches per-sample phase IDs during runtime splice-index construction (single replay pass per game at index-build time)
  - runtime splice indexes use packed arrays (`array`-backed path IDs / offsets / splice indices / phase IDs) instead of Python int lists to reduce RAM usage substantially
  - metrics summary may include `runtime_splice_index_bytes_train` / `runtime_splice_index_bytes_val` for visibility into index memory footprint

## Regression Tests (current)
- `tests/test_training_features.py` covers:
  - side-to-move ID derivation parity/fallback
  - `collate_train()` phase + side-to-move tensor outputs
  - `collate_train_rollout()` rollout-target/mask tensor outputs for multistep batches
  - `NextMoveLSTM` forward path with phase/side feature head inputs enabled
  - scheduler/early-stopping runtime metadata and early-stop behavior in a tiny synthetic training run
  - runtime splice cache index loading for game datasets, plus fallback to runtime indexing on cache config mismatch
  - tiny game-dataset training path that confirms cache-backed data loading mode is used when cache is present
  - per-split cache-load reason reporting (`cache_load_reason_by_split`) in dataset metrics/progress setup events for game datasets (hit vs fallback reason)
  - multistep file-backed training path emits rollout metrics/progress fields and multistep runtime metadata
- `tests/test_game_dataset_architecture.py` covers:
  - compact game-dataset builder CLI output schema (`moves`, no duplicated splice rows)
  - single-step training from compact game-level JSONL via runtime splicing
  - multistep training from compact game-level JSONL via runtime splicing
