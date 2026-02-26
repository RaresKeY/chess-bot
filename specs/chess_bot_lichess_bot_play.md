# Chess Bot Lichess Bot Play

## Responsibility
Provide a separate module for connecting the local move model to the Lichess Bot API, including challenge handling, game-stream move play, and offline preview of bot decisions from a recorded stream fixture.

## Code Ownership
- CLI wrapper: `scripts/lichess_bot.py`
- Preview server utility: `scripts/serve_lichess_preview.py`
- Combined flow wrapper: `scripts/run_lichess_bot_with_preview.py`
- Core logic / HTTP transport / preview renderer: `src/chessbot/lichess_bot.py`
- Tests: `tests/test_lichess_bot.py`
- Reused model runtime: `src/chessbot/play_vs_model.py` (`LoadedMoveModel`)

## Architecture
- `LichessBotRunner` orchestrates event stream processing and per-game stream handling.
- `LichessTransport` abstracts network transport for testability.
- `LichessHTTPTransport` implements Lichess Bot API calls over stdlib `urllib`.
- `MoveProvider` abstraction allows model-backed move selection or test doubles.
- `ModelMoveProvider` uses the existing model artifact runtime and falls back to a legal move when needed.

## Lichess API Behavior (current)
- Reads incoming events from `/api/stream/event` (NDJSON).
- Accepts/declines challenges based on variant/rated/time-control policy.
- Can create outbound user challenges via `POST /api/challenge/{username}` (one-shot CLI mode and live preview control endpoint).
- Opens game stream `/api/bot/game/stream/{gameId}` and plays moves via `/api/bot/game/{gameId}/move/{uci}`.
- Determines bot color from `gameFull.white.title` / `gameFull.black.title` (`BOT`).
- Applies a conservative global request rate limit (default `1500 ms`) between API requests.

## Policy Controls (current)
- `accept_rated` gate for rated challenges
- `allow_variants` allowlist (default `standard`)
- `min_initial_seconds` minimum initial clock
- `dry_run` mode to avoid posting moves while still logging intended actions
- Outbound challenge controls: username, rated flag, clock limit/increment, preferred color, variant

## Token Resolution (current)
- Token source priority for live mode:
  1. `--token`
  2. system keyring lookup (defaults: service `lichess`, username `lichess_api_token`)
  3. `LICHESS_BOT_TOKEN` environment variable
- `keyring` is included in project requirements; lookup failures are still handled gracefully (for example missing backend or missing credential).

## Preview Mode (current)
- `--preview-fixture` reads a local JSON fixture containing a `transcript` list of game-stream events.
- Runs the same game-play logic offline and emits structured JSON logs to stdout.
- Optional `--preview-html` writes a standalone HTML report with board (when state is present), bot actions, and transcript for visual inspection.

## Live Preview (current)
- `--preview-live-dir` enables real-time-ish local preview artifact updates during live bot play.
- Persists per-game files under `games/<game_id>/`:
  - `state.json` (latest streamed game state)
    - includes derived board metadata when moves are present (`fen`, `turn`, `result`, `move_rows` with SAN/UCI)
  - `actions.json` (bot action/error logs for that game)
  - `transcript.json` (raw game stream events observed so far)
- Persists root files:
  - `index.json` (game list / summary)
  - `logs.json` (global structured logs)
- `scripts/serve_lichess_preview.py` serves an interactive live UI (`/index.html`) over local HTTP with client-side polling controls (auto-refresh toggle, interval, follow-newest game, manual refresh) and reads JSON artifacts from the preview dir.
- The live UI renders a visual board using local SVG piece assets (`assets/pieces/cburnett`) plus a compact SAN move table for the selected game, driven by the derived board data in `state.json`.
- The selected-game board highlights the last move (`from` and `to` squares) using a classic green square highlight treatment.
- The live UI also exposes an outbound challenge form that POSTs to the local preview server (`/api/challenge`), which invokes a one-shot `scripts/lichess_bot.py --challenge-user ...` call using the same token/keyring settings.
- The live UI also exposes an opponent-search panel with an `Actively search` toggle that polls a local `/api/online-bots` endpoint (backed by Lichess `/api/bot/online`) and renders clickable online bot candidates with `Prefill` / `Challenge` actions (default search interval `15000 ms`, server-side cache ~`15 s`).
- The opponent-search list can be filtered client-side by the auto-challenge ELO range controls (rating bucket + min/max ELO) to narrow the visible candidates to the target strength window.
- The live UI also exposes an `Auto Challenge` control row that can continuously trigger server-side auto-challenge ticks with configurable interval, rating bucket (`bullet`/`blitz`/`rapid`/`classical`), ELO min/max range, retry cooldown, `include playing bots` behavior, and an `only if no active game` guard.
- Auto-challenge ticks POST to the local preview server endpoint `/api/auto-challenge-tick`, which refreshes/caches online bots, filters by the requested ELO range, skips recently attempted usernames for the configured cooldown window, optionally pauses while a live game is active (using preview `index.json` statuses), and then issues a one-shot outbound challenge via the existing `scripts/lichess_bot.py --challenge-user ...` flow.
- Manual and auto-challenge responses include parsed challenge outcome metadata (challenge status/id/url when available plus a best-effort human-readable outcome/error message), which the UI surfaces in the challenge status text.
- The opponent-search list is a discovery list of online bots, not a guarantee that a bot will accept the chosen challenge settings (policy, variant, time control, or current load may still cause rejection).
- The live preview path no longer rewrites HTML files on each event; the bot writes JSON artifacts only.
- `scripts/run_lichess_bot_with_preview.py` starts both the bot and preview server together and shuts them down as a pair.
- Wrapper shutdown uses isolated child process groups and sends group-level terminate/kill signals so preview/bot descendants are cleaned up more reliably on Ctrl+C or wrapper exit; on Linux, child processes also get a parent-death `SIGTERM` fallback to reduce orphaned preview/bot processes if the wrapper dies unexpectedly.
- Live-played games are appended (default) to `data/live_play/lichess_bot_archive/valid_games.jsonl` via `--played-games-out`, using the same valid-record JSONL schema as dataset validation (`game_id`, `source_file`, `headers`, `result`, `winner_side`, `plies`, `moves_uci`) for potential later training reuse.
- The default path intentionally lives outside `data/validated/elite_*` so bot-vs-bot / live-play data is clearly separated from curated elite corpora.
- The live archive skips empty games and de-duplicates by Lichess game ID within a single bot process run.

## Logging (current)
- Structured JSON logs are always written to stdout.
- Optional `--log-jsonl` appends the same structured events to a local JSONL file.
- Error events include `network_error`, `game_event_error`, `runner_event_error`, and `unexpected_error`.
- Combined wrapper defaults `--log-jsonl` to `artifacts/lichess_bot/bot.jsonl`.
- One-shot outbound challenge mode prints a structured JSON result (`event=challenge_create_result`) and exits without loading the model.
- Live game archival emits `live_game_archived` on success and `live_game_archive_error` on archive failures.
- Preview-server HTTP access logs are written to `artifacts/lichess_live_preview/preview_server_access.log` (or `<preview-dir>/preview_server_access.log`), while routine `200` request lines are suppressed from terminal output; HTTP errors still print to stderr.

## Validation / Tests
- Unit tests cover challenge filtering, turn-taking logic, move posting vs dry-run behavior, illegal stream move detection, preview JSON persistence, outbound challenge request encoding, live transcript-to-valid-record archive conversion, request-rate throttling behavior, and preview-server auto-challenge candidate/range filtering helpers.
- Tests depend on `python-chess` being installed in the runtime.
