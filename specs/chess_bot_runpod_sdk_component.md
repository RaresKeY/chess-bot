# Chess Bot RunPod SDK Component

## Responsibility
Provide a **separate RunPod SDK-based component** for host-side RunPod operations, maintained side-by-side with (and not replacing) the raw REST/GraphQL component.

## Component Boundary (Explicit)
- SDK component:
  - `scripts/runpod_sdk_component.py`
  - `src/chessbot/runpod_sdk_component.py`
- Raw API component (existing, unchanged):
  - `scripts/runpod_provision.py`
  - `scripts/runpod_cycle_*.sh`
  - `scripts/runpod_cli_doctor.sh`

The SDK component is intentionally modular and independent. Changes here should not require changing the raw API scripts unless an explicit cross-component migration is planned.

## Current Capability Surface
- `gpu-search`
  - lists/ranks GPU types through SDK methods when available
  - supports cloud/memory/price filtering
- `template-list`
  - lists templates through SDK methods
  - supports optional template-name filter and pod-only filtering
- `provision`
  - chooses template by id/name
  - supports explicit `--gpu-type-id` or constraint-based auto-pick from SDK GPU list
  - supports env injection (`--env KEY=VALUE`)
  - supports interruptible mode and wait-until-ready polling
- `pod-status`
  - fetches current pod status
- `pod-stop`
  - requests pod stop
- `pod-terminate`
  - requests pod termination/deletion

## Authentication / Secret Resolution
- API key resolution order:
  1. `--api-key`
  2. env `RUNPOD_API_KEY`
  3. keyring (`service=runpod`, `username=RUNPOD_API_KEY`)
  4. dotenv fallback (`RUNPOD_SDK_DOTENV_PATH`/`RUNPOD_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.runpod`, then `.env`)
- Component sets SDK auth state via:
  - env `RUNPOD_API_KEY`
  - common SDK fields when present (`runpod.api_key`, `runpod.config.api_key`)

## Maintenance Contract
- Keep this component modular:
  - no required edits to raw API scripts for SDK component updates
  - SDK-specific compatibility logic stays in `src/chessbot/runpod_sdk_component.py`
- Keep output shape explicit:
  - top-level JSON output includes `"component": "runpod_sdk"` so operators can distinguish source path from raw component outputs.
- Backward compatibility:
  - adding SDK method probes is acceptable
  - do not remove existing raw component behavior from this SDK component spec

## Known Runtime Constraint
- This component requires the Python `runpod` package in the active environment.
- If missing, it exits with an actionable install message and does not mutate raw API workflows.

## Tests
- `tests/test_runpod_sdk_component.py`
  - verifies nested callable discovery, GPU ranking behavior, template selection, API key dotenv fallback, parser defaults, and provision-path behavior with a mocked SDK object.
