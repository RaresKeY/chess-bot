# Chess Bot Dual Sequence Board Architecture

## Status
Implemented.

## Architecture Name
- `dual_side_sequence_board_lstm`

## Responsibility
Train and serve side-specific one-shot future-sequence models that condition on:
- move-context token sequence
- derived board-state planes from context position

Like baseline dual sequence:
- white model trains on `winner_side == "W"`
- black model trains on `winner_side == "B"`

## Code Ownership
- CLI: `scripts/train_dual_sequence.py`
- Core logic: `src/chessbot/training_dual_sequence.py`
- Model: `src/chessbot/model.py` (`NextMoveSeqBoardLSTM`)
- Board features: `src/chessbot/board_features.py`
- Inference routing: `src/chessbot/inference.py`, `scripts/infer_move.py`

## Inputs
- move context tokens (`context`) encoded with vocab
- board-state planes `[C,8,8]` derived from replaying context (`C=18`)

Board feature planes:
- 12 piece planes (white + black piece types)
- side-to-move plane
- castling rights planes (white K/Q, black K/Q)
- en-passant target square plane

## Model Contract
- `NextMoveSeqBoardLSTM`:
  - LSTM move-context encoder (same token pathway as `dual_side_sequence_lstm`)
  - MLP board encoder over flattened board planes
  - board-conditioned initialization of decoder hidden/cell state
  - one-shot sequence decoder head producing logits `[B,H,V]`

## Loss and Metrics
- Train/val loss uses masked per-ply target-probability distance:
  - `loss_ply = 1 - P(actual_move)`
- sequence weighting:
  - geometric step weighting from ply 1 via `step_loss_decay` (default `0.9`, earlier plies highest)
- optional mate-bias weighting multiplies per-ply weights in endgame mate-in-x cases
- reported sequence metrics:
  - `ply_match_rate`
  - `full_seq_exact_hit_rate`

## Artifact Contract
Per-side artifacts remain:
- `artifacts/model_white.pt`
- `artifacts/model_black.pt`

Board family markers:
- `model_family = "dual_side_sequence_board_lstm"`
- `training_objective = "one_shot_sequence_match"`
- `config.board_feature_planes = 18`
- `runtime.use_board_state_feature = true`

## Inference Compatibility
- `src/chessbot/inference.py` accepts this family in sequence path dispatch.
- context is replayed at inference time to derive board-state planes before forward pass.
- dual pair routing (`--white-model` + `--black-model`) works unchanged.
