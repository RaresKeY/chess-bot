# Chess Bot Baseline Next-Move Architecture

## Status
Implemented (current baseline path).

## Architecture Name
- Human name: `Pulse-1 Next-Move Oracle`
- Artifact family id: `next_move_lstm`

## Responsibility
Train and serve a next-move policy model from chess move-history context.

This architecture is the default model family used by:
- `scripts/train_baseline.py`
- cloud training launchers (`deploy/*/train_baseline_preset.sh`, `scripts/runpod_*`, `scripts/vast_*`)
- `scripts/infer_move.py` when loading baseline artifacts

## Code Ownership
- Training CLI: `scripts/train_baseline.py`
- Training core: `src/chessbot/training.py`
- Model: `src/chessbot/model.py` (`NextMoveLSTM`)
- Inference dispatch/runtime: `src/chessbot/inference.py`, `scripts/infer_move.py`

## Model Contract
- Encoder: token embedding + LSTM over context UCI moves.
- Optional conditioning features:
  - winner feature (`W/B/D/?`)
  - phase feature (`opening/middlegame/endgame/unknown`)
  - side-to-move feature (derived from context parity)
- Output: logits over move vocabulary for next-move prediction.

## Training Objective Modes
- `single_step_next_move`:
  - one-step next-token cross-entropy objective.
  - default mode (`--rollout-horizon 1`).
- `multistep_teacher_forced_recursive`:
  - recursive teacher-forced rollout over `H` plies when `--rollout-horizon > 1`.
  - weighted per-step loss via `rollout_loss_decay`.

## Data Contract
- Supported input schemas:
  - spliced rows (`context`, `target`, `next_move`, optional labels)
  - game rows (`moves`/`moves_uci`) with runtime splicing
- Game-row mode builds runtime splice indexes and supports cache-backed indexing (`runtime_splice_cache`).

## Artifact Contract
- `model_family = "next_move_lstm"`
- `training_objective` is stored at artifact root/runtime and reflects selected objective mode.
- Artifact includes:
  - `state_dict`
  - `vocab`
  - `config` (dims/layers/dropout + feature toggles)
  - `runtime` metadata (device, amp, scheduler, checkpoint, stopping, etc.)

## Inference Compatibility
- Supported by default in `scripts/infer_move.py`.
- Inference dispatch falls back to baseline mode when `model_family` is missing/legacy.

## Known Limits (current)
- Data coverage is concentrated on elite games, so robustness against low-quality/blunder move patterns is limited.
- Runtime index structures are optimized but still in-memory structures, not fully streaming training.
