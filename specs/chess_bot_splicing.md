# Chess Bot Splicing / Dataset-Prep Component

## Responsibility
Prepare training datasets from validated games while preserving leakage-safe train/val/test splits by `game_id`.

Current repo supports two dataset architectures:
- compact game-level datasets (new preferred path; runtime splicing in training)
- legacy materialized splice datasets (still supported for compatibility)

## Code Ownership
- Compact game-level dataset CLI: `scripts/build_game_dataset.py`
- Legacy splice dataset CLI: `scripts/build_splice_dataset.py`
- End-to-end monthly prep orchestrator: `scripts/acquire_and_prepare_elite_month.py` (builds both legacy and compact game datasets by default, plus runtime caches)
- Batch monthly prep orchestrator: `scripts/batch_prepare_elite_months.py`
- Batch compact cache orchestrator: `scripts/batch_build_compact_caches.py`
- Core splice logic (legacy + shared split/eligibility helpers): `src/chessbot/splicing.py`
- Shared IO: `src/chessbot/io_utils.py`

## Inputs
- `valid_games.jsonl` from validation component
- Validation rows may contain `moves_uci` and/or canonical alias `moves`; builders/training should accept either
- By default, legacy `scripts/build_splice_dataset.py` rejects live bot archive inputs unless `--allow-live-bot-games` is passed intentionally

## Output Architectures

## 1. Compact Game-Level Dataset (preferred)
Writes:
- `data/dataset/train.jsonl`
- `data/dataset/val.jsonl`
- `data/dataset/test.jsonl`
- `data/dataset/stats.json`

Row shape (compact):
- one game per row
- `game_id`, `winner_side`, `result`, `plies`, `moves`
- optional `source_file`
- optional `headers` only when requested (`--keep-headers`)

Training behavior:
- `src/chessbot/training.py` performs runtime splicing (`context`/`target` generation) from these rows
- rollout horizon/cap becomes a training-time setting instead of a dataset-directory naming convention

## 2. Legacy Materialized Splice Dataset (compatibility path)
Typical outputs (often month/cap-specific dirs):
- `data/dataset/elite_<month>_cap<k>/train.jsonl`
- `.../val.jsonl`
- `.../test.jsonl`
- `.../stats.json`

Row shape (expanded samples):
- `context`, `target`, `next_move`
- per-splice phase metadata
- `ply`, `plies_remaining`, buckets, `winner_side`, `game_id`

## Legacy Sample Construction (materialized splice rows)
For each valid game and splice index `i`:
- `context = moves[:i+1]`
- `target = moves[i+1 : i+1+K]`
- `next_move = target[0]`
- attach phase metadata derived from reconstructed board state at splice context:
  - `phase`, `phase_reason`, `phase_rule_version`
  - `ply`, `plies_remaining`, `plies_remaining_bucket`, `relative_progress_bucket`
- attach `winner_side` and `game_id`

## Split Rules
- Split by game only (never by sample)
- Detect and fail on any game-id overlap across splits
- Default filtering favors decisive games (`1-0`, `0-1`)

## Compact Game Dataset Build Notes (`scripts/build_game_dataset.py`)
- Two-pass streaming build:
  - Pass 1: collect eligible spliceable `game_id`s using runtime-splice eligibility defaults (`min_context`, `min_target`)
  - Pass 2: write compact game rows directly to split JSONL outputs
- This avoids holding all games or all splice samples in memory
- Output `stats.json` records dataset format, split counts, average plies, and runtime splice defaults used for eligibility

## Batch Cache Generation (`scripts/batch_build_compact_caches.py`)
- Automatically scans a directory containing multiple validated month directories (e.g., `data/validated/elite_2025-01`).
- For each month, generates the compact game-level dataset into a `_game` suffixed output directory using `scripts/build_game_dataset.py`.
- Immediately runs `scripts/build_runtime_splice_cache.py` on that new dataset to produce the binary cache needed for high-performance training.
- Intelligently skips directories that already have a complete cache unless `--overwrite` is specified.

## End-to-end Orchestration updates (`scripts/acquire_and_prepare_elite_month.py`)
- By default, it now builds **both** the legacy materialized dataset AND the new preferred compact game dataset (along with its runtime cache).
- Supports `--no-legacy` to skip generating the huge, expanded legacy datasets completely.
- Uses `scripts/batch_build_compact_caches.py` for massive, multi-month backwards-compatible processing.

## Runtime Splice Cache Precompute (`scripts/build_runtime_splice_cache.py`)
- Precomputes the CPU-heavy runtime-splice artifacts for compact game datasets and stores them under `runtime_splice_cache/` inside each dataset directory.
- Per split (`train`, `val`, `test`) writes packed files:
  - `paths.json`
  - `path_ids.u32.bin`
  - `offsets.u64.bin`
  - `splice_indices.u32.bin`
  - `sample_phase_ids.u8.bin`
- Writes `runtime_splice_cache/manifest.json` with:
  - config (`min_context`, `min_target`, `max_samples_per_game`, `seed`)
  - per-split game/sample counts
  - per-split and total cache byte sizes
  - binary layout metadata
- `--dataset-dir` is repeatable, so one invocation can build caches for multiple dataset directories.
- Supports `--jobs N` for task-level parallelism across all scheduled `(dataset, split)` builds in one run (for example, two datasets x three splits can use up to six workers).
- Supports a TTY-only progress bar via `--progress-bar/--no-progress-bar` (stderr output).
- Supports `--verbose/--no-verbose` for per-split summary logs.
- If process-based parallelism is unavailable in constrained environments (for example semaphore/socket restrictions), the script automatically falls back to single-process indexing instead of failing.

## Runtime Cache Validation Utility (`scripts/validate_runtime_splice_cache.py`)
- Validates cache presence and structure for one or many compact game datasets.
- Supports direct dataset paths (`--dataset-dir` repeatable) or root scanning (`--dataset-root` + `--dataset-glob`).
- For each required split (default `train,val,test`), verifies required cache files exist and packed binary row counts align across:
  - `path_ids.u32.bin`
  - `offsets.u64.bin`
  - `splice_indices.u32.bin`
  - `sample_phase_ids.u8.bin`
- Ensures `paths.json` is readable and points to existing source split JSONL files.

## Legacy Materialized Splice Build Notes (`scripts/build_splice_dataset.py`)
- Two-pass streaming build for large corpora
- Optional pass-2 concurrency (`--workers`, `--batch-size`, `--worker-backend`)
- Per-game sample capping (`--max-samples-per-game`) applied during sample generation
- Pass-2 stats include split phase counts and remaining-plies bucket counts

## Verified Legacy Materialized Build (elite_2025-11_cap4)
- Input validated corpus: `data/validated/elite_2025-11/valid_games.jsonl`
- Output directory: `data/dataset/elite_2025-11_cap4`
- Params:
  - `k=4`
  - `min_context=8`
  - `min_target=1`
  - `max_samples_per-game=4`
  - decisive-only filtering enabled
- Result stats:
  - `input_games_total=279613`
  - `input_games_after_filters=235543`
  - `spliceable_games=235488`
  - split games: train `186649`, val `23331`, test `23332`
  - split samples: train `753284`, val `94045`, test `94122`

## Verified Compact Game Build (`data/dataset`, elite_2025-11)
- Input validated corpus: `data/validated/elite_2025-11/valid_games.jsonl`
- Builder: `scripts/build_game_dataset.py`
- Output format: `game_jsonl_runtime_splice_v1`
- Result stats:
  - `input_games_total=279613`
  - `input_games_after_filters=235543`
  - `spliceable_games=235488`
  - split games: train `188378`, val `23557`, test `23553`
- Output file sizes (compact game rows):
  - `data/dataset/train.jsonl` ~ `164 MB`
  - `data/dataset/val.jsonl` ~ `21 MB`
  - `data/dataset/test.jsonl` ~ `21 MB`
