# Chess Bot Validation Component

## Responsibility
Parse and replay PGN games from initial position, reject malformed/illegal/noisy games, and emit accepted/rejected artifacts.

## Code Ownership
- CLI: `scripts/validate_games.py`
- Companion data download utility: `scripts/download_lichess_elite_month.py`
- Core logic: `src/chessbot/validation.py`
- Shared IO: `src/chessbot/io_utils.py`

## Inputs
- PGN file, glob, or directory (`--input`)

## Outputs
- `valid_games.jsonl`: canonicalized accepted games (UCI move lists + metadata)
- `invalid_games.csv`: rejected games with reason and offending ply
- `summary.json`: aggregate counts and failure-mode histogram

## Execution / Performance Notes
- `scripts/validate_games.py` writes `valid_games.jsonl` and `invalid_games.csv` in a streaming fashion (incremental writes), rather than accumulating all rows in memory.
- `src/chessbot/validation.py` exposes an iterator-style validation event stream (`iter_validation_events`) to support large monthly corpora.
- This change is required for full-month elite datasets where in-memory accumulation can exceed practical memory limits.
- Optional concurrency: `scripts/validate_games.py --workers N`
  - Uses file-level threading (each PGN file validated in its own thread to shard outputs, then merged)
  - Effective when validating a directory/glob with multiple PGN files
  - A single large PGN file remains effectively sequential (PGN parsing/replay is stream-based)
- Progress visibility:
  - `scripts/validate_games.py --progress-every N` prints periodic counters during streaming validation
  - threaded mode prints per-file completion summaries as shards finish

## Verified Real Dataset Build (2025-11 Elite Month)
- Source archive: `https://database.nikonoel.fr/lichess_elite_2025-11.zip` (Lichess-linked elite database)
- Extracted PGN: `data/raw/elite/2025-11/lichess_elite_2025-11.pgn`
- Validation outputs:
  - `data/validated/elite_2025-11/valid_games.jsonl`
  - `data/validated/elite_2025-11/invalid_games.csv`
  - `data/validated/elite_2025-11/summary.json`
- Result counts (full month, `min_plies=8`):
  - `total_games=280246`
  - `valid_games=279613`
  - `invalid_games=633`
  - `valid_ratio=0.9977412701697794`

## Validation Rules
- Replay every move legally from initial board state
- Reject parse errors and PGN parser errors
- Reject empty games
- Reject result mismatches when terminal board result is available and disagrees with PGN result
- Optional minimum plies filter (`--min-plies`)

## Valid Record Schema (current)
- `game_id`
- `source_file`
- `headers`
- `result`
- `winner_side`
- `plies`
- `moves_uci`
