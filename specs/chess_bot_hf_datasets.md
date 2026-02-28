# Chess Bot HF Datasets

## Responsibility
Define the canonical Hugging Face dataset repo layout, publish/fetch auth contract, versioning expectations, and stale-version cleanup policy for chess-bot datasets.

## Canonical Repo
- HF dataset repo id: `LogicLark-QuantumQuill/chess-bot-datasets`
- Path prefix: `validated_datasets`
- Dataset layout:
  - `validated_datasets/<dataset_name>/<version>/`
  - current compact dataset names are monthly `_game` folders (for example `elite_2025-11_game`)

## Current Online State (2026-02-27)
- Each monthly compact dataset under `validated_datasets/elite_2025-*_game` has one current version:
  - `20260227T044455Z`
- Previous `validated-*` version folders were cleaned up as stale.

## Auth Contract (Publish/Fetch)
- Token lookup order:
  1. explicit `--token`
  2. env `HF_TOKEN`
  3. keyring lookup
  4. dotenv fallback (`HF_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.hf_dataset`, then `.env`)
- Canonical HF keyring identities:
  - read/fetch: `service=huggingface`, `username=codex_hf_read_token`
  - write/publish: `service=huggingface`, `username=codex_hf_write_token`
- Equivalent explicit CLI flags:
  - `--keyring-service huggingface`
  - `--keyring-username codex_hf_write_token`

## Publish Flow (Canonical)
- Script: `scripts/hf_dataset_publish.py`
- Multi-dataset publish command (new structure):
  - `.venv/bin/python scripts/hf_dataset_publish.py --repo-id LogicLark-QuantumQuill/chess-bot-datasets --repo-path-prefix validated_datasets --dataset-root data/dataset --dataset-glob 'elite_*_game' --keyring-service huggingface --keyring-username codex_hf_write_token`
  - `scripts/hf_dataset_fetch.py` defaults to read profile (`HF_READ_TOKEN` / `codex_hf_read_token`) and only falls back to legacy `HF_TOKEN` for compatibility
- Behavior:
  - validates `train.jsonl`/`val.jsonl` by default
  - for compact game datasets, validates `runtime_splice_cache` by default
  - publishes archive payload (`tar.gz`) + `manifest.json` + `checksums.sha256`

## Fetch Flow (Canonical)
- Script: `scripts/hf_dataset_fetch.py`
- Typical aggregate fetch:
  - `.venv/bin/python scripts/hf_dataset_fetch.py --repo-id LogicLark-QuantumQuill/chess-bot-datasets --repo-path-prefix validated_datasets --all-latest --dest-dir data/hf_datasets`

## Stale Version Policy
- Keep one active version per monthly `_game` dataset unless rollback history is explicitly required.
- Stale-version cleanup target:
  - older `validated-*` folders or any superseded version not selected as active.
- Cleanup method:
  - use HF API folder deletion (`HfApi.delete_folder`) per stale path under `validated_datasets/<dataset>/<version>`.
