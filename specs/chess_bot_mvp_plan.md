# Chess Bot MVP Plan

## Scope
Build a reproducible supervised chess move-prediction pipeline that runs:
0. Optional monthly orchestration wrapper (`scripts/acquire_and_prepare_elite_month.py`) for download + validation + splicing
1. PGN validation (`scripts/validate_games.py`)
2. Splice dataset generation (`scripts/build_splice_dataset.py`)
3. Baseline training (`scripts/train_baseline.py`)
4. Evaluation (`scripts/eval_model.py`)
5. Inference (`scripts/infer_move.py`)

## Logical Section Specs
- Overview: `specs/chess_bot_overview.md`
- README contract: `specs/chess_bot_readme_contract.md`
- Environment/runtime: `specs/chess_bot_environment.md`
- Validation component: `specs/chess_bot_validation.md`
- Splicing component: `specs/chess_bot_splicing.md`
- Training component: `specs/chess_bot_training.md`
- Evaluation component: `specs/chess_bot_evaluation.md`
- Inference component: `specs/chess_bot_inference.md`
- Game viewer component: `specs/chess_bot_game_viewer.md`
- Play-vs-model component: `specs/chess_bot_play_vs_model.md`
- Viewer local server utility: `specs/chess_bot_viewer_server.md`

## Purpose of This Umbrella Spec
- Preserve end-to-end scope and rollout checkpoints in one place.
- Link out to the component-level specs for current implementation truth.

## Decisions
- Move encoding: UCI tokens.
- Training target: next move (`target[0]`) from splice target window.
- Winner-aware training: optional winner embedding and winner-side weighted loss.
- Splits: by `game_id` only to prevent leakage.

## Artifacts
- Validation outputs:
  - `data/validated/valid_games.jsonl`
  - `data/validated/invalid_games.csv`
  - `data/validated/summary.json`
- Dataset outputs:
  - `data/dataset/train.jsonl`
  - `data/dataset/val.jsonl`
  - `data/dataset/test.jsonl`
  - `data/dataset/stats.json`
- Model outputs:
  - `artifacts/model.pt`
  - `artifacts/train_metrics.json`
  - `artifacts/eval_metrics.json`
- Viewer outputs:
  - `artifacts/viewer/game_viewer.html`
- Play UI:
  - served dynamically at `/play-vs-model` by `scripts/play_vs_model_server.py`

## Constraints
- Require legal replay from initial state for accepted games.
- Require decisive games by default for winner-side tasks (`1-0` or `0-1`).
- Keep all outputs machine-readable (`jsonl/csv/json`).

## Environment Notes
- In this container, canonical setup is `python3 -m venv --clear /work/.venv`.
- Avoid relying on externally-synced venv shims (for example `uv`-generated `pip` wrappers) because absolute shebang paths can break across environments.
- Prefer module invocation for tooling reliability: `/work/.venv/bin/python -m pip ...`.

### Container Verification
Run these checks before assuming host and container environments are the same:

```sh
test -f /run/.containerenv && echo containerenv=yes || echo containerenv=no
cat /proc/1/cgroup
cat /proc/1/comm
pwd
ls -a /work
ls /home/mintmainog/workspace/vs_code_workspace/chess_bot 2>/dev/null || echo "host path not present here"
```

Expected for this workspace:
- `/run/.containerenv` present.
- `pwd` at `/work`.
- Host-local path above not present in container.

If these checks fail, warn the user immediately that commands are running outside the expected containerized `/work` context.

### Smoke Test Snapshot (2026-02-22)
- Runtime used: `/work/.venv` rebuilt in-container with `python3 -m venv --clear /work/.venv`.
- End-to-end commands executed successfully: `validate_games.py` -> `build_splice_dataset.py` -> `train_baseline.py` -> `eval_model.py` -> `infer_move.py`.
- Sample-run artifacts generated under `data/validated`, `data/dataset`, and `artifacts`.
- Small sample corpus caveat: only 2 games validated cleanly (others rejected by PGN quality/parser issues), so validation split had 0 rows and metrics are not meaningful for quality claims.

## Current Implementation Notes
- `scripts/*.py` are thin orchestration CLIs; reusable logic lives in `src/chessbot/*`.
- Model artifacts are currently loaded/saved directly inside CLI scripts (no dedicated `artifacts` module yet).
- Training/eval/inference all rely on the same serialized artifact shape: `{state_dict, vocab, config}`.

## Validation Gates
- Validator reports non-zero valid games.
- No game-id overlap across train/val/test.
- Training script logs decreasing loss trend and top-k metrics.
- Evaluation reports top-k and legal-prediction rate.
