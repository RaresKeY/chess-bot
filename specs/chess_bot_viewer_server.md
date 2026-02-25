# Chess Bot Viewer Server Utility

## Responsibility
Serve generated HTML viewer files and local piece assets over HTTP for browser viewing.

## Code Ownership
- CLI: `scripts/serve_viewer.py`
- Runtime dependency: Python stdlib `http.server`

## Behavior
- Serves a chosen directory as the HTTP document root (default `/work`)
- Prints base URL and optional example URL when `--example-path` is provided
- Runs until interrupted (`Ctrl+C`)

## Typical Use
1. Generate viewer HTML with `scripts/render_game_viewer.py`
2. Start local server with `scripts/serve_viewer.py` (`--dir` can be any filesystem path)
3. Open `http://127.0.0.1:<port>/artifacts/viewer/game_viewer.html`

## Portability
- Utility is pure Python stdlib (`http.server`) and not container-specific.
- It can serve any directory on any machine with Python, including a cloned GitHub repo checkout.
