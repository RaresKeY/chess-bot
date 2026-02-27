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

## Token Resolution
Vast scripts resolve API key in this order:
1. explicit `--api-key`
2. env `VAST_API_KEY`
3. keyring (`service=vast`, `username=VAST_API_KEY`)
4. dotenv fallback (`VAST_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.vast`, then `.env`)

## Validation
- `tests/test_vast_api_helpers.py`
- `tests/test_vast_cycle_scripts.py`

## Sources
- Vast API reference and endpoint descriptions: https://docs.vast.ai/api-reference/instances
- Vast create instance from ask: https://docs.vast.ai/api-reference/instances/create-instance
- Vast instance list: https://docs.vast.ai/api-reference/instances/show-instances
- Vast update instance: https://docs.vast.ai/api-reference/instances/change-bid
- Vast delete instance: https://docs.vast.ai/api-reference/instances/destroy-instance
- Vast template precedence: https://docs.vast.ai/cli/commands
- Vast web API endpoint examples: https://docs.vast.ai/rest-api
