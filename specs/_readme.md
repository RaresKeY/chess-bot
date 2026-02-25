# work - Specs Index

**Tech Stack**: Python

---

**IMPORTANT** Before making changes or researching any part of the codebase, use the table below to find and read the relevant spec first. This ensures you understand existing patterns and constraints.

## Documentation

Populate this table following template style.

| Spec | Code | Purpose |
|------|------|---------|
| [Commit Workflow](./commit_workflow.md) | `AGENTS.md`, git workflow tasks | Canonical commit process, safeguards, and skill routing. |
| [Chess Bot MVP Plan](./chess_bot_mvp_plan.md) | `implement.md`, `README.md`, `scripts/*.py`, `src/chessbot/*` | Umbrella chess-bot spec linking component specs, validation gates, and rollout notes. |
| [Chess Bot Overview](./chess_bot_overview.md) | `README.md`, `scripts/*.py`, `src/chessbot/*` | High-level scope, architecture pattern, decisions, and artifact contract. |
| [Chess Bot README Contract](./chess_bot_readme_contract.md) | `README.md` | Rules for component-oriented README sections and spec/README sync. |
| [Chess Bot Environment](./chess_bot_environment.md) | `AGENTS.md`, `.venv`, runtime checks | Container verification, venv guidance, and runtime assumptions. |
| [Chess Bot Validation](./chess_bot_validation.md) | `scripts/validate_games.py`, `src/chessbot/validation.py` | PGN replay validation behavior, outputs, and record schema. |
| [Chess Bot Splicing](./chess_bot_splicing.md) | `scripts/build_splice_dataset.py`, `src/chessbot/splicing.py` | Splice sample generation and game-level split guarantees. |
| [Chess Bot Training](./chess_bot_training.md) | `scripts/train_baseline.py`, `src/chessbot/training.py`, `src/chessbot/model.py` | Baseline model training behavior, winner-aware loss, and artifact format. |
| [Chess Bot Evaluation](./chess_bot_evaluation.md) | `scripts/eval_model.py`, `src/chessbot/evaluation.py` | Offline metrics and legality-rate evaluation behavior. |
| [Chess Bot Inference](./chess_bot_inference.md) | `scripts/infer_move.py`, `src/chessbot/inference.py` | Top-k decoding and best-legal move selection behavior. |
| [Chess Bot Game Viewer](./chess_bot_game_viewer.md) | `scripts/render_game_viewer.py`, `src/chessbot/viewer.py`, `assets/pieces/cburnett/*` | HTML board viewer with left/right navigation and local piece assets. |
| [Chess Bot Play-vs-Model](./chess_bot_play_vs_model.md) | `scripts/play_vs_model_server.py`, `src/chessbot/play_vs_model.py` | Interactive browser UI + API server for playing against the model. |
| [Chess Bot Viewer Server Utility](./chess_bot_viewer_server.md) | `scripts/serve_viewer.py` | Local HTTP server utility for viewing generated HTML and piece assets in-browser. |
