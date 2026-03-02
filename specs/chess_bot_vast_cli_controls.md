# Chess Bot Vast.ai CLI Controls

## Purpose
Define the Vast.ai host-side provisioning and lifecycle workflow added as a parallel cloud provider path. This flow must stay isolated from RunPod logic and files.

## Scope
- `scripts/vast_provision.py`
- `scripts/vast_cli_doctor.sh`
- `scripts/vast_regression_checks.sh`
- `scripts/vast_cycle_common.sh`
- `scripts/vast_cycle_start.sh`
- `scripts/vast_cycle_stop.sh`
- `scripts/vast_cycle_terminate.sh`
- `scripts/vast_cycle_status.sh`
- `scripts/vast_local_smoke_test.sh`
- `scripts/vast_noauth_deploy_checks.sh`
- `scripts/cloud_connectivity_health_checks.sh` (provider framework)
- `scripts/cloud_checks/providers/vast.sh` (Vast provider checks)
- `deploy/vast_cloud_training/README.md`
- `deploy/vast_cloud_training/PLAN.md`
- `deploy/vast_cloud_training/env.example`
- `deploy/vast_cloud_training/train_baseline_preset.sh`

## Separation Contract
- All new Vast logic uses `vast_` script names and `VAST_*` env variables.
- Vast state tracking file is `config/vast_tracked_instances.jsonl`.
- Existing RunPod scripts/specs remain authoritative and unchanged for RunPod flows.

## Official Vast.ai API Conventions (researched 2026-02-27)
- Auth: Bearer token header (`Authorization: Bearer <token>`).
- Offer search endpoint: `POST /api/v0/bundles/`.
- Instance creation from ask/offer: `PUT /api/v0/asks/{id}/`.
- Instance management (state/label): `PUT /api/v0/instances/{id}/`.
- Instance deletion: `DELETE /api/v0/instances/{id}/`.
- Template behavior: if both template and explicit create args are set, explicit args override template values.

## Current CLI Commands
- `offer-search`: query bundles and rank by price/reliability/VRAM filters.
- `instance-list`: list authenticated user instances.
- `provision`: choose offer (or explicit `--offer-id`), then create instance.
- `manage-instance`: set `state` (`running`/`stopped`) and/or label.
- `destroy-instance`: delete instance.

## Cycle Scripts
- `vast_cycle_start.sh`
: provisions a Vast instance and writes `artifacts/vast_cycles/<run_id>/provision.json`.
- `vast_cycle_stop.sh`
: transitions instance to stopped state via `manage-instance --state stopped`.
- `vast_cycle_terminate.sh`
: destroys instance via `destroy-instance`.
- `vast_cycle_status.sh`
: dumps current instance list to `artifacts/vast_cycles/<run_id>/status_response.json`.

## No-Auth Deployment Smoke Scripts
- `vast_local_smoke_test.sh`
: runs local smoke training through `deploy/vast_cloud_training/train_baseline_preset.sh` using default dataset `data/dataset/_smoke_fast_game` (no Vast API calls).
  - local smoke wrapper sets `REPO_DIR` to repo root and `VENV_DIR` to `${VAST_SMOKE_VENV_DIR:-<repo>/.venv}` to avoid `/workspace` path assumptions on non-cloud hosts.
- `vast_noauth_deploy_checks.sh`
: runs no-auth Vast validation lane:
  - `python -m unittest discover -s tests -p "test_vast*.py" -v`
  - `bash scripts/cloud_connectivity_health_checks.sh --provider vast --no-live`
  - `bash scripts/vast_local_smoke_test.sh` (unless `VAST_NOAUTH_SKIP_LOCAL_SMOKE=1`)

## Cloud-Run Preflight
- Before running Vast cloud provisioning/training lifecycle scripts, push the intended local code changes to the GitHub repo/branch so remote clone/pull uses the latest committed code.

## Token Resolution
Vast scripts resolve API key in this order:
1. explicit `--api-key`
2. env `VAST_API_KEY`
3. keyring fallback (canonical identity mapping in `specs/chess_bot_secrets_contract.md`)
4. dotenv fallback (`VAST_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.vast`, then `.env`)

Container guidance for this workspace:
- prefer dotenv provider path (`VAST_DOTENV_PATH` or `CHESSBOT_DOTENV_PATH`) over keyring.
- canonical token/key identity mapping is maintained in `specs/chess_bot_secrets_contract.md`.

## Validation
- `tests/test_vast_api_helpers.py`
- `tests/test_vast_cycle_scripts.py`
- `tests/test_cloud_connectivity_architecture.py` (provider interface/timeout framework)

## Sources
- Vast API reference and endpoint descriptions: https://docs.vast.ai/api-reference/instances
- Vast create instance from ask: https://docs.vast.ai/api-reference/instances/create-instance
- Vast instance list: https://docs.vast.ai/api-reference/instances/show-instances
- Vast update instance: https://docs.vast.ai/api-reference/instances/change-bid
- Vast delete instance: https://docs.vast.ai/api-reference/instances/destroy-instance
- Vast template precedence: https://docs.vast.ai/cli/commands
- Vast web API endpoint examples: https://docs.vast.ai/rest-api
