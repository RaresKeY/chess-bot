# Chess Bot Model Olympics

## Responsibility
Track a curated set of model artifacts ("participants") and the repeatable method used to compare them head-to-head.

This spec is intended to be edited as participants are added/removed and as match methodology evolves.

## Code Ownership
- Match runner: `scripts/play_model_vs_model.py`
- Inference compatibility dispatch used during matches: `src/chessbot/inference.py`, `src/chessbot/play_vs_model.py`
- Output summaries / PGNs: `artifacts/reports/*.json`, `artifacts/reports/*.pgn`

## Canonical Match Method (current)

Primary method:
- run head-to-head matches with `scripts/play_model_vs_model.py`
- alternate colors
- collect:
  - W/D/L
  - score %
  - average plies
  - fallback moves per game (high-signal secondary quality metric)

Current default-style settings used in recent comparisons:
- `games=6` (quick pair checks) or `games=12/20` (longer checks)
- `topk_a=10`, `topk_b=10`
- `winner_side_a='W'`, `winner_side_b='W'` (not ideal for ranking; see caveats)
- `alternate_colors=true`
- `max_plies=300`
- device: GPU when available (`cuda`)
- policy mode:
  - multistep artifacts auto-use `rollout`
  - legacy single-step artifacts auto-use `next`

## Important Caveats
- These matches are often highly deterministic and draw-heavy.
- Outcome score alone is a weak ranking signal in this setup.
- Fallback rate is currently an important secondary metric (lower is usually better move-generation quality).
- Training metrics across different datasets/objectives are not directly comparable ("not apples-to-apples").

## Participant Registry (mutable)

Each participant entry should include:
- `id` (stable short name used in reports)
- artifact path
- model family / objective (if known)
- notes/status (active, archived, experimental, baseline, etc.)

### Active / Recent Participants (as of February 26, 2026)

1. `base_model`
- Artifact: `artifacts/model.pt`
- Type: legacy single-step next-move model
- Role: baseline / reference

2. `combined_e20_b2048`
- Artifact: `artifacts/model_combined_elite_2025_10_2025_11_e20_b2048.pt`
- Type: legacy single-step next-move model
- Role: stronger local legacy reference

3. `cloud_fulltrain`
- Artifact: `artifacts/runpod_cycles/fulltrain-20260226T113920Z/collected/run_artifacts/model_fulltrain-20260226T113920Z.pt`
- Type: legacy single-step next-move model (RunPod full train)
- Role: strongest recent single-step reference in local comparisons

4. `subset_2layer_128_256`
- Artifact: `artifacts/experiments/model_multistep_cap4_subset_20e_2080ti.pt`
- Type: multistep teacher-forced recursive (`rollout_horizon=4`)
- Role: local multistep subset experiment (stronger than `subset_8x64` in metrics/fallbacks)

5. `subset_8x64`
- Artifact: `artifacts/experiments/model_multistep_cap4_subset_20e_2080ti_l8_h64.pt`
- Type: multistep teacher-forced recursive (`rollout_horizon=4`)
- Role: depth experiment (`8x64`), known weaker than `subset_2layer_128_256`

6. `game_runtime_20k20e`
- Artifact: `artifacts/experiments/model_game_runtime_splice_20k20e_gpu.pt`
- Type: multistep teacher-forced recursive (`rollout_horizon=4`) trained from compact game-level dataset via runtime splicing
- Role: first 20-epoch GPU model on new compact game dataset architecture

## Add / Remove Procedure (manual spec maintenance)

To add a participant:
1. Add a new registry entry above (ID, artifact path, type, role)
2. Run at least one head-to-head vs a reference participant (`base_model`, `combined_e20_b2048`, or `cloud_fulltrain`)
3. Save JSON + PGN under `artifacts/reports/`
4. Record findings in a report (or append a short results note here)

To remove/deprecate a participant:
1. Mark it as archived/deprecated in the registry (preferred)
2. Optionally remove from active list after artifacts are moved/deleted
3. Preserve historical report references when possible

## Recommended Pairing Matrix (quick tournament)

For `N` participants:
- run round-robin pairings (`N*(N-1)/2`)
- use `6` games per pairing for quick scans
- escalate interesting pairings to `20+` games

Example command template:

```bash
.venv/bin/python scripts/play_model_vs_model.py \
  --model-a <path_a> \
  --model-b <path_b> \
  --alias-a <id_a> \
  --alias-b <id_b> \
  --games 6 \
  --summary-out artifacts/reports/<id_a>_vs_<id_b>_g6.json \
  --pgn-out artifacts/reports/<id_a>_vs_<id_b>_g6.pgn \
  --no-progress
```

## Recent Verified Comparison Artifacts (examples)
- `artifacts/reports/game_runtime20k20e_vs_base_g6.json`
- `artifacts/reports/game_runtime20k20e_vs_combined_g6.json`
- `artifacts/reports/game_runtime20k20e_vs_cloudfull_g6.json`
- `artifacts/reports/model_vs_model_subset2layer_vs_8x64_g12.json`
- `artifacts/reports/rr_cloudfull_vs_base_g6.json`

## Next Method Improvements (not yet canonical)
1. Use `winner_side='?'` to reduce conditioning bias in rankings
2. Add controlled randomness / opening diversity to break deterministic draw loops
3. Add engine-assisted intermediate-position evaluation for stronger ranking signal
