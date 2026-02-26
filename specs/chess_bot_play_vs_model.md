# Chess Bot Play-vs-Model Component

## Responsibility
Provide an interactive browser UI to play chess against the trained move model, with server-side legality checks and model move generation.

## Code Ownership
- Convenience launcher: `main.py`
- CLI server: `scripts/play_vs_model_server.py`
- Core logic: `src/chessbot/play_vs_model.py`
- Dependencies reused:
  - `src/chessbot/model.py`
  - `src/chessbot/inference.py`
- Piece assets: `assets/pieces/cburnett/*.svg`

## Architecture
- Browser UI renders board and move list from server-provided FEN snapshots
- Python backend validates user moves using `python-chess`
- Backend requests model move from loaded artifact and applies it if legal
- loaded-artifact inference now uses compatibility auto-dispatch:
  - legacy artifacts (no multistep metadata) use next-move inference
  - multistep-trained artifacts default to rollout-first-move inference using the artifact rollout horizon
- If model has no legal prediction, backend falls back to a legal move so play can continue
- HTTP server uses threaded handling with graceful shutdown on `SIGINT`/`SIGTERM`, explicit socket close, and address reuse enabled for quick restarts

## Endpoints (current)
- `GET /play-vs-model` -> interactive HTML page
- `POST /api/state` -> returns serialized state from provided `context`
- `POST /api/move` -> applies user move and model response

## Inputs
Server CLI:
- `main.py` convenience wrapper launches play-vs-model and, when `--model` is omitted, resolves the newest `*.pt` artifact under `artifacts/` and injects repo root as `--dir` (unless explicitly provided)
- `--model` model artifact path
- `--dir` HTTP document root for static assets
- `--piece-base` URL path to piece images
- `--winner-side` model conditioning token
- `--topk` model top-k candidate count

API payload (move):
- `context` (UCI list)
- `user_move` (UCI)
- `winner_side`
- `topk`
- `user_color` (currently UI uses `white`)

## UI Behavior (current)
- User plays White by clicking source then destination square
- Viewer-style board with responsive 1:1 squares
- Move history + snapshot navigation (`|<`, `←`, `→`, `>|`)
- `Undo Pair` removes last user+model plies
- `New Game` resets to starting position
- Toggleable `Log` panel (Show/Hide) records move events, the exact raw model prediction UCI, and model fallback/error messages in a consistent per-turn order
- Illegal model predictions are surfaced in the UI log as `ERROR` entries and include the attempted model UCI (fallback may still be applied so play continues)

## Known Limitations (current)
- Model quality may be weak; fallback legal move is used when no legal prediction exists
- Promotion UI defaults to server-side auto-queen fallback for 4-char pawn promotion UCIs
- Only user-as-White flow is wired in current UI
