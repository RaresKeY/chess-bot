# Chess Bot Game Viewer Component

## Responsibility
Render a local HTML visual for a PGN game with board graphics and left/right move navigation.

## Code Ownership
- CLI: `scripts/render_game_viewer.py`
- Core logic: `src/chessbot/viewer.py`
- Piece assets: `assets/pieces/cburnett/*.svg`

## Inputs
- `--pgn` PGN file path
- `--game-index` 0-based game index
- `--out-html` output HTML path
- `--piece-base` optional relative asset path override (default auto-resolves relative to repo `assets/pieces/cburnett`)

## Output
- Self-contained HTML viewer with embedded game positions/moves (except external local piece SVG asset references)

## UI Behavior (current)
- Renders chessboard from FEN snapshots for each ply
- Board squares must remain 1:1 aspect ratio (no rectangular stretching across responsive layouts)
- On-screen navigation buttons: first / previous / next / last
- Keyboard navigation: `ArrowLeft`, `ArrowRight`, `Home`, `End`
- Sidebar shows game metadata and move list
- Clicking a move in the move list jumps to that ply

## Asset Source
- Piece set: `cburnett` SVGs downloaded from the open-source Lichess `lila` repository
- Local storage path: `assets/pieces/cburnett/`
- Default generated HTML references piece assets via a path computed relative to the output HTML and repo root (not container-specific `/work` paths)
- Keep source/license attribution in README and commit history when refreshing assets
