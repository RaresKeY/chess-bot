# Chess Bot Validation Component

## Responsibility
Parse and replay PGN games from initial position, reject malformed/illegal/noisy games, and emit accepted/rejected artifacts.

## Code Ownership
- CLI: `scripts/validate_games.py`
- Companion data download utility: `scripts/download_lichess_elite_month.py`
- End-to-end monthly prep orchestrator (invokes validation): `scripts/acquire_and_prepare_elite_month.py`
- Batch monthly prep orchestrator from month/URL lists: `scripts/batch_prepare_elite_months.py`
- Core logic: `src/chessbot/validation.py`
- Shared IO: `src/chessbot/io_utils.py`

## Inputs
- PGN file, glob, or directory (`--input`)
- For month-based acquisition workflows, `scripts/batch_prepare_elite_months.py` accepts a text file of `YYYY-MM` entries and/or Lichess elite ZIP URLs (one per line, comments allowed) and invokes `scripts/acquire_and_prepare_elite_month.py` per month.
- `scripts/acquire_and_prepare_elite_month.py` now skips months that already have complete validated + dataset outputs when `--overwrite` is omitted; partial existing outputs still fail fast and require `--overwrite` for rebuild safety.
- `scripts/download_lichess_elite_month.py` now validates cached ZIPs before reuse, automatically deletes/re-downloads invalid cached files, and rejects HTML/error-page responses so unavailable months do not get cached as fake `.zip` files.
- As of **February 26, 2026**, the `database.nikonoel.fr` elite monthly archive is publishing dumps through **2025-11** (archive begins at **2020-06**). The repo’s default `config/elite_month_validator_links.txt` is a curated subset (currently 2025-01..2025-11).

## Outputs
- `valid_games.jsonl`: canonicalized accepted games (UCI move lists + metadata)
- `invalid_games.csv`: rejected games with reason and offending ply
- `summary.json`: aggregate counts and failure-mode histogram

## Execution / Performance Notes
- `scripts/validate_games.py` writes `valid_games.jsonl` and `invalid_games.csv` in a streaming fashion (incremental writes), rather than accumulating all rows in memory.
- `src/chessbot/validation.py` exposes an iterator-style validation event stream (`iter_validation_events`) to support large monthly corpora.
- `src/chessbot/validation.py` now parses PGN with a custom `python-chess` visitor that captures headers + mainline UCI moves directly during parse, skipping variation parsing and avoiding full `Game` tree construction plus a second replay pass.
- This change is required for full-month elite datasets where in-memory accumulation can exceed practical memory limits.
- Optional concurrency: `scripts/validate_games.py --workers N`
  - Uses file-level multiprocessing (each PGN file validated in its own worker process to shard outputs, then merged)
  - `--workers 0` (default) uses all available CPU cores
  - `--all-cores` is a convenience alias for `--workers 0`
  - Effective when validating a directory/glob with multiple PGN files
  - For a single PGN file, the CLI now auto-shards the file into temporary chunk PGNs by game header (`[Event ...]`) when `workers>1` (disable with `--no-auto-shard-single-file`) so file-level multiprocessing still uses all cores
  - Summary JSON includes `input_source`, `auto_sharded_single_file`, and `auto_shard_game_count` metadata
- Progress visibility:
  - `scripts/validate_games.py` shows a live TTY progress bar/status line by default (`--progress-bar`, disable with `--no-progress-bar`)
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
- Parse and legally apply mainline PGN SAN moves using `python-chess` during PGN parsing (custom visitor path)
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
- `moves_uci` (legacy/original field name)
- `moves` (canonical alias; same UCI move list, added for game-level runtime-splice dataset builders)

## Schema Compatibility Note
- Validation now emits both `moves_uci` and `moves` for the same UCI mainline list.
- Existing downstream code that reads `moves_uci` remains compatible.
- New compact game-level dataset tooling can read either field and writes `moves` as the canonical move-list field.

## Schema Reuse (current)
- The Lichess live bot archive (`scripts/lichess_bot.py`, `src/chessbot/lichess_bot.py`) appends live-played games to `data/live_play/lichess_bot_archive/valid_games.jsonl` using the same valid-record JSONL schema for downstream compatibility.
- Despite schema compatibility, live-played bot games are intentionally stored outside the elite validation tree and are excluded from splice dataset builds by default unless explicitly allowed.
