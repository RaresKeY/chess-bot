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
- `--policy-mode {auto,next,rollout}` (`auto` preserves old-model compatibility and prefers rollout for multistep-trained artifacts)
- `--rollout-plies` optional continuation rollout length (`>0` switches CLI to rollout mode)
- `--rollout-fallback-legal/--no-rollout-fallback-legal` optional legal fallback during rollout generation
- `--device` torch device for inference (default `cpu`; `cuda:N` supported)

## Output (current)
Printed object containing:
- `topk` candidate UCI tokens
- `best_legal` selected legal move from candidates (or empty string)
- `device` actually used for inference

When `--rollout-plies > 0`, printed object instead contains rollout-oriented fields:
- `rollout` predicted continuation UCI list (up to requested plies, may stop early)
- `first_move` first predicted rollout move (empty if none)
- `steps_generated`
- `fallback_moves` count (only non-zero when `--rollout-fallback-legal` is enabled)
- `step_debug` per-step top-k / legality / chosen move details
- `device`

## Compatibility / Dispatch Behavior (current)
- Inference helpers now detect artifact metadata when present:
  - root/runtime `training_objective`
  - root `model_family`
  - runtime `rollout_horizon`
- Old artifacts that lack these fields are treated as legacy single-step next-move models (`single_step_next_move`) and continue using legacy next-move inference behavior
- `--policy-mode auto` prioritizes new multistep artifacts by using rollout-first-move inference when metadata indicates a multistep objective, while keeping older artifacts on next-move logic

## Constraints
- Context must be legal from starting position; illegal context raises an error.
