# Chess Bot Dual Sequence Implementation Plan

## Status
Phase 1-4 implemented in initial form. This document now tracks delivered scope and remaining hardening tasks.

## Objective
Deliver a dual-model one-shot sequence training stack:
- White model trained on White-win games only
- Black model trained on Black-win games only
- Default horizon `H=8` with mask-based cutoff
- Draw rows dropped
- Endgame mate-in-x positive bias support

## Phase Plan
### Phase 1: Data and Row Construction
1. Add winner-filtered split utilities in `src/chessbot/training.py` or a new module (`src/chessbot/training_dual_sequence.py`).
2. Build one-shot row extraction:
   - input: `context`, `target`, `winner_side`
   - output: `seq_targets[H]`, `seq_mask[H]`
3. Enforce draw/unknown drop in dual-sequence mode.
4. Add deterministic row counters for white/black retained/dropped stats.

Status:
- Complete (initial implementation in `src/chessbot/training_dual_sequence.py`).

Acceptance gates:
- Deterministic row counts from fixed fixture.
- White model rows contain only `winner_side == W`.
- Black model rows contain only `winner_side == B`.

### Phase 2: Model and Loss
1. Add one-shot sequence model class (new module or extension in `src/chessbot/model.py`):
   - returns logits shape `[B, H, V]`.
2. Add masked sequence CE objective.
3. Add sequence-match metrics:
   - `ply_match_count`, `ply_match_rate`, `full_seq_exact_hit_rate`.
4. Add endgame mate-in-x bias weighting hook with conservative default (off).

Status:
- Complete (initial implementation in `NextMoveSeqLSTM` + masked sequence loss + metrics).

Acceptance gates:
- Tensor shape assertions pass for varying `B/T/H`.
- Loss ignores masked plies exactly.
- Match metrics align with synthetic deterministic logits fixtures.

### Phase 3: New Training CLI
1. Add `scripts/train_dual_sequence.py`.
2. CLI should support:
   - train/val input paths (repeatable)
   - horizon control (default 8)
   - side mode (`white`, `black`, `both`)
   - draw-drop enforced default
   - mate-bias toggle + weight controls
3. Emit dual artifacts and combined metrics JSON.
4. Keep baseline script untouched (`scripts/train_baseline.py` remains compatible).

Status:
- Complete (`scripts/train_dual_sequence.py`).

Acceptance gates:
- `--help` exposes dual-sequence flags.
- `side-mode=both` writes both model artifacts.
- `side-mode=white` and `side-mode=black` write only requested artifact.

### Phase 4: Inference Compatibility
1. Add helper to route side-to-play to correct artifact.
2. Preserve existing inference API behavior for current baseline models.
3. Add optional dual-sequence inference mode returning:
   - first move
   - predicted continuation sequence

Status:
- Complete (dual-artifact routing added to `scripts/infer_move.py` and `src/chessbot/inference.py`).

Acceptance gates:
- Existing inference tests remain green.
- New tests confirm artifact routing by side-to-play.

### Phase 5: Docs + Specs + Runbook
1. Update training/inference specs for implemented behavior.
2. Update parameter/config mappings for all new runtime controls.
3. Add runbook examples for:
   - white-only training
   - black-only training
   - dual training in one run

Status:
- In progress (specs updated for architecture/inference/mappings; additional runbook examples can be expanded later).

Acceptance gates:
- `specs/_readme.md` includes all new component docs.
- Runtime controls documented with defaults/types/effects/precedence.

## Test Plan (Unit + Regression)
### Unit Tests (new file targets)
- `tests/test_dual_sequence_training.py`
  - winner filtering correctness
  - horizon cutoff + mask construction
  - one-shot output tensor shape contract
  - masked CE correctness
  - sequence-match metric correctness
  - mate-bias weight application bounded/targeted

- `tests/test_train_dual_sequence_cli.py`
  - parser flags and defaults
  - side-mode artifact output expectations
  - draw-drop behavior in small fixture

- `tests/test_dual_sequence_inference.py`
  - side-to-play artifact routing
  - continuation output shape/length checks

### Regression Tests (required)
- Add focused regressions for each discovered bug during implementation:
  - incorrect winner filtering
  - mask off-by-one at end of sequence
  - side routing selecting wrong model
  - mate-bias applying outside configured scope

Regression checklist (must run):
1. Targeted new test module(s) first via venv.
2. Any touched legacy component test modules.
3. Full regression script (`bash scripts/test.sh`) before merge.

Execution note:
- In this environment, `torch` is not installed for `/usr/bin/python3`, so runtime test execution is skipped/blocked locally.
- New tests are added and guarded to skip when `torch` is unavailable:
  - `tests/test_dual_sequence_training.py`
  - `tests/test_train_dual_sequence_cli.py`
  - `tests/test_dual_sequence_inference.py`

## Commands (planned)
Use project venv:

```sh
.venv/bin/python -m unittest -q tests.test_dual_sequence_training
.venv/bin/python -m unittest -q tests.test_train_dual_sequence_cli
.venv/bin/python -m unittest -q tests.test_dual_sequence_inference
bash scripts/test.sh
```

## Compatibility Guardrails
- Do not break existing artifact loading for `next_move_lstm`.
- Keep new artifacts clearly versioned/typed (new `model_family` value).
- Baseline training/eval/inference paths remain default unless explicitly using dual-sequence script/mode.

## Deliverables Checklist
- New training script: `scripts/train_dual_sequence.py`
- New/updated core module(s) under `src/chessbot/`
- New tests + regressions in `tests/`
- Updated specs:
  - architecture doc
  - training component behavior
  - inference compatibility notes
  - parameter/config mapping entries

## Visual Rollout Roadmap
```text
[Phase 1 Data] -> [Phase 2 Model/Loss] -> [Phase 3 CLI] -> [Phase 4 Inference] -> [Phase 5 Docs/Runbook]
       |                 |                     |                    |                    |
   filter/mask       one-shot [B,H,V]      train script        side routing         finalized specs
   correctness       + match metrics        + artifacts         + dual inference     + mapping tables
```
