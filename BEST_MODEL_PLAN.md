# Best Model Training Plan

## Goal
Train the strongest practical model for arena performance while reducing fallback/legality failures, using cost-aware multi-GPU RunPod runs.

## Target Objective
- Primary: improve arena strength (head-to-head outcomes).
- Secondary: reduce fallback moves / legality failures.
- Constraint: keep run cost and runtime manageable.

## Recommended Baseline Recipe
- Dataset: full HF aggregate under `game_jsonl_runtime_splice_v1` (no `max_total_rows` cap).
- Training mode: DDP (`RUNPOD_GPU_COUNT=N`, `RUNPOD_FULL_TRAIN_NPROC_PER_NODE=N`).
- Architecture:
  - `embed_dim=512`
  - `hidden_dim=1024`
  - `num_layers=4`
  - `dropout=0.1`
- Features:
  - `phase_feature=on`
  - `side_to_move_feature=on`
- Optimization:
  - AMP enabled
  - plateau scheduler (`metric=val_loss`, factor/patience as current defaults)
  - runtime splice cache required (`TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1`)
- Duration: start with 20 epochs; extend to 30-40 only if arena gains are confirmed.

## Why This Shape First
- Recent 1M-subset deeper run did not outperform prior references in arena score (all draws) and showed weaker fallback metrics in some pairings.
- Highest expected gains now are from broader data exposure and stability before further depth/width escalation.

## Priority Order (Highest Impact First)
1. Full-data coverage (or at least very large uncapped subset).
2. Stronger evaluation protocol (20+ games, `winner_side=?`, opening diversity).
3. Batch/LR stability tuning on multi-GPU DDP.
4. Architecture scaling sweeps after 1-3 are stable.

## 3-Run Sweep Plan

### 1) Safe (stability-first)
- Shape: `384/768`, `layers=4`, `dropout=0.15`
- Epochs: 20
- Goal: establish stable full-data baseline with low fallback drift.

### 2) Balanced (recommended default)
- Shape: `512/1024`, `layers=4`, `dropout=0.1`
- Epochs: 20
- Goal: best expected quality/cost tradeoff.

### 3) Max (capacity test)
- Shape: `640/1280`, `layers=5`, `dropout=0.1`
- Epochs: 20 initially
- Goal: check if larger capacity improves arena/fallback enough to justify cost.

## Acceptance Criteria Per Run
- Training completes with `train_exit_code=0`.
- Cache usage reports `train=hit`, `val=hit`.
- Arena eval vs references (`base`, `combined`, `cloud_fulltrain`):
  - at least 20 games per pairing,
  - use `winner_side=?` for both models,
  - no fallback regression vs current best reference profile.

## Stop/Advance Rules
- If fallback worsens and arena score is flat: do not scale architecture up; improve data/optimization first.
- If fallback improves and arena score holds/improves: proceed to next capacity tier or longer epochs.
- Only extend epochs when short-run arena signal is positive.

## Operational Notes
- Use `bash scripts/runpod_cycle_status.sh --watch --auto-collect` during supervised/manual runs.
- Generate concise status snapshots with:
  - `python scripts/runpod_cycle_report_style.py --run-id <run_id>`
