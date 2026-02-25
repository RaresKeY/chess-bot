# Chess Bot Inference Component

## Responsibility
Generate top-k candidate move tokens from a model artifact and return the best legal move for a supplied UCI move context.

## Code Ownership
- CLI: `scripts/infer_move.py`
- Core logic: `src/chessbot/inference.py`
- Model dependency: `src/chessbot/model.py`

## Inputs
- `--model` artifact path
- `--context` space-separated UCI moves
- `--winner-side` conditioning token (`W`, `B`, `D`, `?`)
- `--device` torch device for inference (default `cpu`; `cuda:N` supported)

## Output (current)
Printed object containing:
- `topk` candidate UCI tokens
- `best_legal` selected legal move from candidates (or empty string)
- `device` actually used for inference

## Constraints
- Context must be legal from starting position; illegal context raises an error.
