# Chess Bot Inference Component

## Responsibility
Generate top-k candidate move tokens from a model artifact and return the best legal move for a supplied UCI move context.
Inference now applies legal-move masking at decode time so illegal tokens are removed before candidate ranking.

## Code Ownership
- CLI: `scripts/infer_move.py`
- Core logic: `src/chessbot/inference.py`
- Model dependency: `src/chessbot/model.py`

## Inputs
- `--model` single artifact path (legacy `next_move_lstm`, move-only dual-sequence artifact, or board-conditioned dual-sequence artifact)
- optional dual pair inputs for side-routed sequence inference:
  - `--white-model`
  - `--black-model`
- `--context` space-separated UCI moves
- `--winner-side` conditioning token (`W`, `B`, `D`, `?`)
- `--policy-mode {auto,next,rollout}` (`auto` preserves old-model compatibility and prefers rollout for multistep-trained artifacts)
- `--sequence-decode-policy {sequence_path,step1_legal}` controls dual-sequence first-move selection policy
- `--rollout-plies` optional continuation rollout length (`>0` switches CLI to rollout mode)
- `--rollout-fallback-legal/--no-rollout-fallback-legal` optional legal fallback during rollout generation
- `--fallback-topk-multipliers` comma-separated progressive top-k retry multipliers before hard fallback behavior
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

When dual-sequence inference is used (single dual artifact or dual pair routing), output includes sequence-oriented fields:
- `topk_step1` top-k candidates for ply-1
- `predicted_sequence` greedy per-ply sequence tokens across configured horizon
- `predicted_sequence_with_probs` structured greedy per-ply entries:
  - `ply` (1-based index), `move_uci`, `probability`, `log_probability`
- `legal_sequence` legality-filtered continuation sequence generated step-wise from per-ply top-k
- `legal_sequence_with_probs` structured legality-filtered per-ply entries:
  - `ply` (1-based index), `move_uci`, `probability`, `log_probability`
- `best_legal` legal first move selected from step-1 top-k
- `move_uci` selected first move
- `selected_model_side` when dual pair routing is used (`white` or `black`)
- `horizon`
- `model_side` artifact-side metadata when present
- `sequence_decode_policy_used` active dual-sequence decode policy
- `decode_topk_attempts` progressive top-k attempts actually used for legality search
- `decode_topk_used` top-k attempt where legal decode succeeded (or last attempted)

## Compatibility / Dispatch Behavior (current)
- Inference helpers now detect artifact metadata when present:
  - root/runtime `training_objective`
  - root `model_family`
  - runtime `rollout_horizon`
- Old artifacts that lack these fields are treated as legacy single-step next-move models (`single_step_next_move`) and continue using legacy next-move inference behavior
- `--policy-mode auto` prioritizes new multistep artifacts by using rollout-first-move inference when metadata indicates a multistep objective, while keeping older artifacts on next-move logic
- `model_family` values below dispatch to one-shot sequence inference and return `policy_mode_used=sequence`:
  - `dual_side_sequence_lstm`
  - `dual_side_sequence_board_lstm`
  - `allplay_bootstrap_dualhead_curriculum_lstm`
  - `allplay_bootstrap_dualhead_board_curriculum_lstm`
- board-conditioned dual artifacts derive board-state planes from provided context before forward pass
- when both `--white-model` and `--black-model` are provided, inference routes to artifact by side-to-move derived from context length parity (white on even plies, black on odd plies)
- dual-sequence first-move selection supports:
  - `sequence_path` (default): score legal first-move candidates using horizon-aware continuation path score (log-prob sum over chosen legal sequence tokens)
  - `step1_legal`: legacy behavior selecting first legal candidate from step-1 top-k
- next-move and rollout decode now support progressive top-k retries (`--fallback-topk-multipliers`) before reporting no legal model move
- legal-move masking details:
  - next-move decode masks logits to legal moves from current board before selecting top-k/best move
  - dual-sequence decode masks each ply against legal moves for the board state at that ply (based on chosen continuation path)

## Constraints
- Context must be legal from starting position; illegal context raises an error.
- CLI requires either:
  - `--model`, or
  - both `--white-model` and `--black-model`
