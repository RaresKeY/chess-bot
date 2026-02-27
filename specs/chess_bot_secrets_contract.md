# Chess Bot Secrets Contract

## Responsibility
Define one canonical contract for secret/token resolution so all CLI and runtime flows behave consistently across local, sandboxed, and containerized environments.

## Code Ownership
- Shared resolver: `src/chessbot/secrets.py`
- Current consumers:
  - `scripts/runpod_provision.py`
  - `scripts/hf_dataset_publish.py`
  - `scripts/hf_dataset_fetch.py`
  - `src/chessbot/lichess_bot.py`
  - `scripts/runpod_cycle_common.sh` (delegates dotenv lookup to Python helper)

## Secret Resolution Contract
- Resolver API must support ordered sources and return `(value, source_label)`.
- Source labels are informational only (`explicit`, `env`, `keyring`, `dotenv`).
- Empty/whitespace-only values are treated as missing.
- Missing secret errors must state which sources were checked.

## Source Orders (Current Ground Truth)
- RunPod API key (`RUNPOD_API_KEY`):
  1. explicit CLI (`--api-key`)
  2. environment
  3. keyring (`service=runpod`, `username=RUNPOD_API_KEY`)
  4. dotenv fallback
- HF token (`HF_TOKEN`):
  1. explicit CLI (`--token`)
  2. environment
  3. keyring (`service=huggingface`, `username=codex_hf_write_token`)
  4. dotenv fallback
- Lichess token (`LICHESS_BOT_TOKEN`):
  1. explicit CLI (`--token`)
  2. keyring (`service=lichess`, `username=lichess_api_token`)
     - credential description label: `Lichess API Token`
  3. environment
  4. dotenv fallback

## Keyring Metadata (Non-Sensitive)
- RunPod:
  - service: `runpod`
  - username/key: `RUNPOD_API_KEY`
- Hugging Face datasets:
  - service: `huggingface`
  - username/key: `codex_hf_write_token`
- Lichess:
  - service: `lichess`
  - username/key: `lichess_api_token`
  - credential description: `Lichess API Token`

## Layout Rules
- `.env` is secrets-only:
  - allowed keys: secret values used by runtime resolution (currently `RUNPOD_API_KEY`, `HF_TOKEN`, `LICHESS_BOT_TOKEN`)
  - values should default to empty placeholders in repo-local setup unless explicitly populated by the operator
- Do not place non-sensitive operational metadata in `.env`:
  - keyring service names/usernames
  - credential descriptions
  - image repository defaults
  - workflow notes or usage instructions
- Non-sensitive secret-management metadata belongs in specs:
  - this file is the canonical location for keyring identities, source-order contract, and dotenv behavior
- Documentation defaults (for example image repos, dataset repo IDs) belong in README or feature specs, not in secrets-only `.env`.

## Dotenv Contract
- Override path env vars are checked first:
  - RunPod: `RUNPOD_DOTENV_PATH`, `CHESSBOT_DOTENV_PATH`
  - HF: `HF_DOTENV_PATH`, `CHESSBOT_DOTENV_PATH`
  - Lichess: `LICHESS_DOTENV_PATH`, `CHESSBOT_DOTENV_PATH`
- Default fallback filenames:
  - RunPod: `.env.runpod`, `.env`
  - HF: `.env.hf_dataset`, `.env`
  - Lichess: `.env.lichess`, `.env`
- Parser behavior:
  - supports `export KEY=...`
  - supports quoted values
  - supports inline comments after whitespace (`KEY=value # comment`)
  - ignores invalid key names

## Keyring -> Dotenv Populate Utility
- Script: `scripts/populate_env_from_keyring.py`
- Purpose: generate a repo-local secrets-only dotenv file from canonical keyring entries.
- Canonical mapping:
  - `RUNPOD_API_KEY` <- keyring `runpod` / `RUNPOD_API_KEY`
  - `HF_TOKEN` <- keyring `huggingface` / `codex_hf_write_token`
  - `LICHESS_BOT_TOKEN` <- keyring `lichess` / `lichess_api_token`
- Behavior:
  - default output path: `.env`
  - default mode fails if any required keyring entry is missing
  - `--allow-missing` writes empty placeholder values for missing entries
  - never prints secret values to stdout/stderr

## Security Requirements
- Never print secret values in logs, reports, or errors.
- Avoid mutating global process environment as part of lookup.
- Use source labels for diagnostics instead of raw token content.
- Agent policy: do not read `.env`/`.env*` contents during routine operation; treat them as secrets-only.

## Regression Testing Requirements
- Add/update tests for:
  - precedence order per consumer flow
  - keyring unavailable/unconfigured behavior
  - dotenv fallback path behavior
  - missing-secret error clarity
- Current regression coverage includes:
  - `tests/test_secrets_resolution.py`
  - `tests/test_runpod_api_helpers.py`
  - `tests/test_runpod_cycle_scripts.py`
  - `tests/test_hf_dataset_fetch.py`
  - `tests/test_hf_dataset_publish.py`
  - `tests/test_lichess_bot.py` (token resolution cases)
