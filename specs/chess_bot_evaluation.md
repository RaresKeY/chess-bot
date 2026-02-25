# Chess Bot Evaluation Component

## Responsibility
Evaluate a trained model artifact on held-out splice rows and report quality + legality metrics, and provide a benchmark helper for head-to-head games versus a known UCI engine.

## Code Ownership
- CLI: `scripts/eval_model.py`
- Engine match benchmark CLI: `scripts/play_model_vs_engine.py`
- Core logic: `src/chessbot/evaluation.py`
- Model dependency: `src/chessbot/model.py`

## Runtime / Device Controls
- `scripts/eval_model.py` supports `--device` (default `cpu`; `cuda:N` supported)
- CLI prints a small CUDA preflight summary before evaluation
- Model weights are loaded on CPU first (`torch.load(..., map_location="cpu")`) and then moved to requested device

## Metrics (current)
- `top1`
- `top5`
- `legal_rate_top1`
- `by_phase` grouped metrics (`opening`, `middlegame`, `endgame`, `unknown` when needed)
- `by_remaining_plies` grouped metrics (`<=10`, `11-20`, `>20`, fallback `unknown`)
- `phase_rule_version` (current phase-classification heuristic version)

## Legality Metric Behavior
- Reconstruct board from each sample context
- Decode top-1 predicted token to UCI move
- Count legal predictions only if predicted move is legal in reconstructed position
- Overall `legal_rate_top1` is aggregated from exact numerator/denominator counts across rows (not batch-averaged rates)

## Engine Match Benchmark Helper (current)
- `scripts/play_model_vs_engine.py` runs the model against a UCI engine binary (e.g. Stockfish) using `python-chess.engine`
- Alternates colors across games and reports W/D/L, score percentage, and average plies
- Tracks model fallback move usage (when no legal model prediction is found and a legal fallback move is used)
- Supports optional summary JSON and PGN export for later analysis
