# Chess Bot Dual Sequence Architecture

## Status
Implemented (initial version).
This spec describes the move-only dual-sequence family.
Board-conditioned variant is tracked separately in `specs/chess_bot_dual_sequence_board_architecture.md`.
All-play bootstrap architecture is tracked separately in `specs/chess_bot_allplay_bootstrap_dual_architecture.md`.

## Architecture Name
- Human name: `WinnerSplit Dual Tracks`
- Artifact family id: `dual_side_sequence_lstm`

## Responsibility
Train and serve side-specific one-shot future-sequence models:
- White model trained from White-win games only
- Black model trained from Black-win games only

## Code Ownership
- CLI: `scripts/train_dual_sequence.py`
- Core logic: `src/chessbot/training_dual_sequence.py`
- Model: `src/chessbot/model.py` (`NextMoveSeqLSTM`)
- Inference routing: `src/chessbot/inference.py`, `scripts/infer_move.py`

## Objective
Given preceding context moves, predict a sequence of future plies in one forward pass.

```text
context moves (UCI) -> encoder -> sequence decoder head -> logits [B, H, V]
```

`H` defaults to `8` and includes both sides' plies in order.

## Data Contract
- Supported input schemas:
  - spliced rows (`context`, `target`, `winner_side`)
  - game rows (`moves`/`moves_uci`, `winner_side`) with runtime splice indexing
- Winner filtering per model:
  - white model keeps `winner_side == "W"`
  - black model keeps `winner_side == "B"`
- Draw/unknown rows are dropped (current behavior).

## Model Contract
- `NextMoveSeqLSTM`:
  - context token embedding + encoder LSTM
  - learned step embeddings for `0..H-1`
  - decoder LSTM producing one logit vector per future ply
- Output tensor shape: `[batch, horizon, vocab_size]`
- Optional side-to-move conditioning in the decoder input.

## Loss and Metrics
- Train/val loss: masked per-ply target-probability distance over horizon (`1 - P(actual_move)` per ply).
- Sequence cutoff:
  - targets shorter than `H` are padded with `mask=0`.
- Step-weight decay:
  - geometric decay controlled by `step_loss_decay` (default `0.9`) so earlier plies are weighted more than later plies.
- Mate-bias hook:
  - optional multiplicative boost for endgame rows where ground-truth continuation reaches checkmate within `mate_in_x`.

Tracked sequence metrics:
- `ply_match_count`
- `ply_match_rate`
- `full_seq_exact_hit_count`
- `full_seq_exact_hit_rate`

## Artifact Contract
Per-side artifacts:
- `artifacts/model_white.pt`
- `artifacts/model_black.pt`

Artifact fields:
- `model_family = "dual_side_sequence_lstm"`
- `training_objective = "one_shot_sequence_match"`
- `model_side` (`white` or `black`)
- `state_dict`, `vocab`, `config`, `runtime`

Combined run metrics:
- `artifacts/train_metrics_dual_sequence.json`

## Inference Compatibility
- Existing next-move and rollout inference remain supported.
- New dual routing:
  - provide both `--white-model` and `--black-model` in `scripts/infer_move.py`
  - selector uses side-to-move derived from context parity
- Single-artifact dual-sequence inference is also supported when `--model` points to a dual-sequence artifact.
- Dual-sequence inference exposes per-ply probability outputs for both greedy predicted sequence and legality-filtered chosen sequence (`*_with_probs` fields).
- Dual-sequence decode applies legal-move masking per ply against the current board state before ranking candidates.

## Visual Flow
```text
                    +---------------------------+
                    | input train/val JSONL(s) |
                    +-------------+-------------+
                                  |
                +-----------------+-----------------+
                |                                   |
      filter winner_side == W              filter winner_side == B
                |                                   |
          train white model                    train black model
                |                                   |
       artifacts/model_white.pt            artifacts/model_black.pt
                +-----------------+-----------------+
                                  |
            artifacts/train_metrics_dual_sequence.json
```

## Known Limits (current)
- Sequence prediction is one-shot neural continuation, not explicit tree search.
- Winner-only filtering intentionally biases style toward winning continuations and may reduce defensive coverage.
