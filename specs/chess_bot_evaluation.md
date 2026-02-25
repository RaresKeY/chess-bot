# Chess Bot Evaluation Component

## Responsibility
Evaluate a trained model artifact on held-out splice rows and report quality + legality metrics.

## Code Ownership
- CLI: `scripts/eval_model.py`
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

## Legality Metric Behavior
- Reconstruct board from each sample context
- Decode top-1 predicted token to UCI move
- Count legal predictions only if predicted move is legal in reconstructed position
