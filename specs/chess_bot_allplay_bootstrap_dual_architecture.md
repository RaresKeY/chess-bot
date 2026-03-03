# Chess Bot All-Play Bootstrap Dual Architecture

## Status
Implemented.

## Architecture Name
- Human name: `ForgeFirst All-Play SideTune Curriculum`
- Training architecture id: `allplay_bootstrap_side_finetune_curriculum`
- Output artifact family ids:
  - `allplay_bootstrap_dualhead_curriculum_lstm`
  - `allplay_bootstrap_dualhead_board_curriculum_lstm`

## Responsibility
Train dual white/black sequence models with a two-stage method:
1. Bootstrap a shared model on all valid games (`winner_side in W/B/D`).
2. Fine-tune side-specific models (`white`, `black`) from the shared initialization.

This architecture is intended to reduce over-specialization from winner-only cold starts and keep side-specialized end artifacts.

## Code Ownership
- CLI: `scripts/train_dual_sequence.py`
- Core logic: `src/chessbot/training_dual_sequence.py`
  - `train_bootstrap_dual_head_curriculum_from_jsonl_paths(...)`
  - `train_dual_sequence_model_from_jsonl_paths(...)`
- Inference dispatch: `src/chessbot/inference.py`

## Training Flow
```text
train/val JSONL -> stage 1 (model_side=all) shared bootstrap model
                          -> shared vocab + shared state_dict
                          -> stage 2 white fine-tune (model_side=white)
                          -> stage 2 black fine-tune (model_side=black)
```

Stage details:
- Stage 1 (`bootstrap_allplay`):
  - winner filter: all valid `W/B/D` rows
  - unknown winner rows (`?`) dropped
  - outputs shared artifact (typically `artifacts/model_shared_bootstrap.pt`)
- Stage 2 (`finetune_white`, `finetune_black`):
  - winner filter per side (`W` or `B`)
  - fixed vocab from shared stage
  - full model weight initialization from shared stage
  - outputs side artifacts used in dual routing inference

## Loss / Curriculum Controls
Loss core remains masked per-ply target-probability distance (`1 - P(actual_move)`).

Additional controls:
- `step1_loss_multiplier`: scales first-ply weight before geometric decay.
- `curriculum_start_horizon`, `curriculum_end_horizon`, `curriculum_ramp_epochs`:
  - training mask can progressively unlock later plies across epochs.
  - validation defaults to full-horizon metrics for checkpoint selection.

## Artifact Contract
Shared bootstrap artifact:
- family id matches selected backbone (`dual_side_sequence_lstm` or `dual_side_sequence_board_lstm`)
- `model_side = all`
- runtime includes stage/architecture metadata

Final side artifacts:
- move-only: `model_family = allplay_bootstrap_dualhead_curriculum_lstm`
- board-conditioned: `model_family = allplay_bootstrap_dualhead_board_curriculum_lstm`
- `model_side = white` or `black`

Merged metrics output includes:
- `training_mode = dual_bootstrap_curriculum`
- `training_architecture = allplay_bootstrap_side_finetune_curriculum`
- `bootstrap` metrics + per-side metrics

## Inference Compatibility
`src/chessbot/inference.py` treats both all-play bootstrap family ids as dual-sequence artifacts and routes through sequence inference automatically.
