# Chess Bot Splicing Component

## Responsibility
Transform validated games into supervised splice samples and perform leakage-safe train/val/test splitting by `game_id`.

## Code Ownership
- CLI: `scripts/build_splice_dataset.py`
- End-to-end monthly prep orchestrator (invokes splicing): `scripts/acquire_and_prepare_elite_month.py`
- Batch monthly prep orchestrator (iterates month/URL lists): `scripts/batch_prepare_elite_months.py`
- Core logic: `src/chessbot/splicing.py`
- Shared IO: `src/chessbot/io_utils.py`

## Inputs
- `valid_games.jsonl` from validation component
- By default, `scripts/build_splice_dataset.py` rejects live bot archive inputs (`source_file=lichess_live_bot`, `LichessGameId` headers, or `data/live_play/...` paths) to prevent accidental mixing with elite corpora.
- Override only intentionally with `--allow-live-bot-games`.

## Outputs
- `data/dataset/train.jsonl`
- `data/dataset/val.jsonl`
- `data/dataset/test.jsonl`
- `data/dataset/stats.json`

## Sample Construction
For each valid game and splice index `i`:
- `context = moves[:i+1]`
- `target = moves[i+1 : i+1+K]`
- `next_move = target[0]`
- attach phase metadata derived from reconstructed board state at splice context:
  - `phase` (`opening`, `middlegame`, `endgame`, fallback `unknown`)
  - `phase_reason`
  - `phase_rule_version` (current: `material_castling_v1`)
  - `ply`, `plies_remaining`, `plies_remaining_bucket`, `relative_progress_bucket`
- attach `winner_side` and `game_id`

## Split Rules
- Split by game only (never by sample)
- Detect and fail on any game-id overlap across splits
- Default filtering favors decisive games (`1-0`, `0-1`)

## Execution / Performance Notes
- `scripts/acquire_and_prepare_elite_month.py` now defaults `--max-samples-per-game` to `8`, and its default dataset output directory suffix is derived dynamically as `elite_<month>_cap<max-samples-per-game>`.
- `scripts/build_splice_dataset.py` uses a two-pass streaming build for large corpora:
  - Pass 1: stream validated games to collect spliceable `game_id`s and compute train/val/test split assignment
  - Pass 2: stream validated games again and write samples directly to split JSONL outputs
- This avoids holding all games or all splice samples in memory.
- Per-game sample capping (`--max-samples-per-game`) is applied within each game during streaming.
- Pass-2 stats now also record `split_phase_counts` and `split_remaining_bucket_counts` in `stats.json` based on written sample rows.
- Optional concurrency for pass 2: `--workers N --batch-size M`
  - Uses parallel batch processing for per-game sample generation while preserving single-writer output files
  - `--worker-backend auto` (default) prefers `process` when `workers > 1` because splicing is CPU-bound Python work (better CPU utilization than threads)
  - `--worker-backend thread` remains available as an override
  - `--workers 0` (default) uses all available CPU cores for pass-2 parallel batches
  - Split assignment remains deterministic and leakage-safe (computed before parallel pass 2)
- Progress visibility:
  - `--progress-every N` prints pass-1 and pass-2 counters while the dataset build streams through input games
  - available in both single-threaded and threaded pass-2 modes
- Safety guard:
  - live bot archive corpora are excluded from splice builds by default (`--no-allow-live-bot-games`)
  - pass `--allow-live-bot-games` only when intentionally building a dataset from live-played games

## Verified Real Dataset Build (elite_2025-11_cap4)
- Input validated corpus: `data/validated/elite_2025-11/valid_games.jsonl`
- Output directory: `data/dataset/elite_2025-11_cap4`
- Params:
  - `k=4`
  - `min_context=8`
  - `min_target=1`
  - `max_samples_per_game=4`
  - decisive-only filtering enabled
- Result stats:
  - `input_games_total=279613`
  - `input_games_after_filters=235543`
  - `spliceable_games=235488`
  - split games: train `186649`, val `23331`, test `23332`
  - split samples: train `753284`, val `94045`, test `94122`
