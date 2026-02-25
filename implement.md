# Chess Bot — Implementation Plan

## Goal
Train a move-prediction model using **high-level (high-quality) chess matches**, with a training setup focused on the **winning side**. The model should learn to predict the **next move sequence** from a partial game history.

---

## Core Training Idea (from sketch)
1. Take a full game move list from a strong match.
2. Identify the game winner (`white` or `black`).
3. Choose a splice index `i` in the move timeline.
4. Use all moves up to pointer `i` as input context.
5. Predict following moves for **both sides** after `i`.
6. Repeat over many splice points (`loop: splice i+1`) and many games.

This turns one game into many supervised samples.

---

## Data Representation

### Input (context)
- Sequence of moves from game start to splice index `i`.
- Recommended format:
  - SAN tokens (simple start), or
  - UCI tokens (more deterministic), or
  - board-state tensors per ply (advanced).

### Target (future)
- Next move token (single-step prediction), and optionally:
- Next `K` moves (multi-step sequence prediction), alternating white/black.

### Winner Feature
Because the concept emphasizes training on the **winning side**, add:
- `winner_side` feature (`W` or `B`) and/or
- sample weighting favoring games/moves from winner perspective.

---

## Dataset Construction Pipeline

### 1) Source Selection
- Use high-level PGN data (e.g., titled-player or high-Elo games).
- Filter invalid/short/noisy games.

### 1.1) Research Notes: High-Quality Match Databases
Potential high-quality sources to evaluate and compare:
- **Lichess Elite Database** (monthly dumps, very large, strong-player subsets available).
- **FIDE / major tournament PGNs** (high data quality, lower volume, curated events).
- **TWIC (The Week in Chess)** archives (broad tournament coverage, PGN format).
- **Chess.com published event/game exports** where licensing allows usage.

Selection criteria for final dataset:
- Mean/median Elo threshold (e.g., 2200+ or titled-only pools).
- Time-control filter (prefer rapid/classical for higher move quality; exclude ultra-bullet for baseline).
- Completion quality (exclude aborted/unfinished games).
- License and redistribution constraints documented per source.

### 2) Parse & Normalize
- Parse PGN.
- Normalize move text format.
- Keep metadata: result, players, Elo, event type.

### 2.1) Validation Script (start -> finish, legal moves)
Create a `validate_games.py` data-quality script that:
1. Replays each game from initial board state to terminal state.
2. Verifies every move is legal at the moment it is played.
3. Detects parse errors, illegal SAN/UCI tokens, and broken move numbers.
4. Confirms game result consistency when possible (`1-0`, `0-1`, `1/2-1/2`).
5. Outputs a clean index of valid games plus rejection reasons for invalid games.

Suggested output artifacts:
- `valid_games.jsonl` (or parquet) with canonicalized move lists.
- `invalid_games.csv` with `game_id`, `reason`, `offending_ply`.
- Validation summary report (% valid, top failure modes).

### 3) Splice Sampling
For each game with move list `M = [m0, m1, ..., mn]`:
- Iterate splice index `i` from `min_context` to `n - min_target`.
- Build sample:
  - `x = [m0 ... mi]`
  - `y = [m(i+1) ... m(i+K)]` (or only `m(i+1)`)
  - `winner = side(result)`
- Optional balancing:
  - Cap number of samples per game.
  - Avoid over-representing openings by stratified splice selection.

### 4) Split
- Split by **game**, not by sample, to avoid leakage.
- Train/val/test recommended: 80/10/10.

---

## Modeling Strategy

### Baseline (recommended first)
- Token embedding + sequence model (LSTM/Transformer).
- Objective: next-move cross-entropy.
- Teacher forcing for sequence targets.

### Winner-Aware Extension
- Concatenate winner embedding with context representation.
- Add loss weighting for winner-side moves.
- Optional auxiliary head: predict game outcome from context.

### Inference Modes
- Greedy next move.
- Beam search for move sequence generation.
- Legality filter using chess engine/library (reject illegal outputs).

---

## Training Loop (high level)
1. Load batch of `(context, target, winner_side)`.
2. Forward pass.
3. Compute loss:
   - primary next-move/sequence loss,
   - plus optional weighted winner-side component.
4. Backprop + optimizer step.
5. Validate periodically on held-out games.

Track:
- Top-1 / Top-k next-move accuracy.
- Legal-move rate after decoding.
- Sequence exact-match (for small `K`).

---

## Evaluation Plan
- **Offline**:
  - Next-move accuracy by phase (opening/middlegame/endgame).
  - Accuracy conditioned on winner side.
  - Robustness on unseen openings.
- **Practical**:
  - Playouts vs baseline bot.
  - Blunder rate under fixed depth legality checker.

---

## Phased Execution Plan (Goals + Verification)

### Phase 1 — Source Research & Dataset Decision
**Goals to reach**
- Select 1–2 primary high-quality PGN sources.
- Freeze dataset acceptance criteria (Elo floor, time control, completion rules, licensing notes).
- Produce a reproducible source manifest (URLs/archives/months used).

**Ways to verify**
- A checked-in research note with chosen sources and rejected alternatives.
- A machine-readable config (e.g., `dataset_sources.yaml`) with all filters.
- A reproducible fetch command/log that matches the manifest.

### Phase 2 — Validation Pipeline (start -> finish legality)
**Goals to reach**
- Implement `validate_games.py` to replay each game from initial position.
- Reject games with parse errors, illegal moves, or inconsistent results.
- Emit valid/invalid outputs with reason codes.

**Ways to verify**
- Run validator on a sample corpus and confirm non-zero valid games.
- Check `invalid_games.csv` includes structured reasons and offending ply.
- Spot-check random valid games by replaying with a chess library without errors.

### Phase 3 — Splice Dataset Generation
**Goals to reach**
- Build splice samples from valid games only.
- Ensure winner-side metadata is present for each sample.
- Enforce leakage-safe split by game id.

**Ways to verify**
- Dataset stats report: total samples, per-split counts, avg context length.
- Assert no game id overlap across train/val/test.
- Quick schema check confirms fields: `context`, `target`, `winner_side`, `game_id`.

### Phase 4 — Baseline Training
**Goals to reach**
- Train a baseline next-move model end-to-end.
- Achieve stable loss decrease and usable top-k accuracy.
- Save checkpoints and training metadata.

**Ways to verify**
- Training curves show convergence trend.
- Validation top-1/top-k exceeds random/chance baseline.
- Repro run with fixed seed yields comparable metrics (within tolerance).

### Phase 5 — Evaluation & Inference Hardening
**Goals to reach**
- Evaluate by phase (opening/middlegame/endgame) and winner conditioning.
- Add legality filter at inference and measure legal-move rate.
- Benchmark against a simple baseline bot in controlled playouts.

**Ways to verify**
- Evaluation report with segmented metrics and failure examples.
- Inference test shows near-100% legal output after legality filtering.
- Playout summary (win/draw/loss, blunder proxy) against baseline.

### Phase 6 — Delivery Readiness
**Goals to reach**
- Package scripts/docs so another engineer can reproduce pipeline.
- Finalize minimal deliverables and known limitations.
- Define next-step roadmap (search hybrid, stronger models, more data).

**Ways to verify**
- Clean run from raw PGN -> validated data -> trained model -> inference demo.
- README checklist completed with commands and expected outputs.
- Peer handoff test: independent rerun succeeds without ad-hoc fixes.

---

## Minimal Deliverables
1. Source-research note comparing high-quality PGN databases and selection criteria.
2. `validate_games.py` script that verifies legal move progression from start to finish.
3. PGN preprocessing script producing splice-based samples.
4. Train script for baseline next-move predictor.
5. Eval script with top-k and legality metrics.
6. Inference script returning best legal move from context.

---

## Pseudocode
```text
for game in high_level_games:
    moves = parse_moves(game)
    winner = get_winner(game.result)

    for i in range(min_context, len(moves) - min_target):
        context = moves[: i + 1]
        target  = moves[i + 1 : i + 1 + K]   # or single next move

        add_sample(
            x=context,
            y=target,
            winner_side=winner
        )
```

---

## Notes / Assumptions
- This plan focuses on supervised move prediction, not full search-based engine design.
- If quality needs exceed supervised learning, a hybrid approach can combine:
  - neural move prior (this project),
  - tactical search (minimax/MCTS),
  - and engine-based evaluation.
