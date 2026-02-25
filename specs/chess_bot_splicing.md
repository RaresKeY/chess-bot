# Chess Bot Splicing Component

## Responsibility
Transform validated games into supervised splice samples and perform leakage-safe train/val/test splitting by `game_id`.

## Code Ownership
- CLI: `scripts/build_splice_dataset.py`
- Core logic: `src/chessbot/splicing.py`
- Shared IO: `src/chessbot/io_utils.py`

## Inputs
- `valid_games.jsonl` from validation component

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
- attach `winner_side` and `game_id`

## Split Rules
- Split by game only (never by sample)
- Detect and fail on any game-id overlap across splits
- Default filtering favors decisive games (`1-0`, `0-1`)

## Execution / Performance Notes
- `scripts/build_splice_dataset.py` uses a two-pass streaming build for large corpora:
  - Pass 1: stream validated games to collect spliceable `game_id`s and compute train/val/test split assignment
  - Pass 2: stream validated games again and write samples directly to split JSONL outputs
- This avoids holding all games or all splice samples in memory.
- Per-game sample capping (`--max-samples-per-game`) is applied within each game during streaming.
- Optional concurrency for pass 2: `--workers N --batch-size M`
  - Uses threaded batch processing for per-game sample generation while preserving single-writer output files
  - Split assignment remains deterministic and leakage-safe (computed before threaded pass 2)
- Progress visibility:
  - `--progress-every N` prints pass-1 and pass-2 counters while the dataset build streams through input games
  - available in both single-threaded and threaded pass-2 modes

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
