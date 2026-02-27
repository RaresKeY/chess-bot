# Build Caches For All Validated Months

Use the dedicated batch flow script that is built for this exact job.

## Canonical command

```bash
.venv/bin/python scripts/batch_build_compact_caches.py \
  --validated-dir data/validated \
  --dataset-out-dir data/dataset \
  --jobs 0
```

## Rebuild everything (force overwrite)

```bash
.venv/bin/python scripts/batch_build_compact_caches.py \
  --validated-dir data/validated \
  --dataset-out-dir data/dataset \
  --jobs 0 \
  --overwrite
```

## What this flow does

- Discovers every `valid_games.jsonl` under `data/validated`.
- Runs `scripts/build_game_dataset.py` for each month.
- Then runs `scripts/build_runtime_splice_cache.py` for each generated `_game` dataset (`train,val,test`).
- `--jobs 0` means auto CPU core count (recommended default).
