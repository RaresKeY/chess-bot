# Chess Bot RunPod SDK Component

## Responsibility
Provide a **separate RunPod SDK-based component** for host-side RunPod operations, maintained side-by-side with (and not replacing) the raw REST/GraphQL component.

## Component Boundary (Explicit)
- SDK component:
  - `scripts/runpod_sdk_component.py`
  - `scripts/runpod_sdk_cycle_start.sh`
  - `scripts/runpod_sdk_cycle_stop.sh`
  - `scripts/runpod_sdk_cycle_full_smoke.sh`
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
  - supports explicit `--image-name` for SDK versions that require `image_name` or `template_id` on create
  - when template-list API is unavailable, template fallback now applies only to the "template method missing" case (it no longer masks auth/runtime template-list errors)
  - fails fast with an explicit error when both resolved `template_id` and `image_name` are empty before calling SDK `create_pod`
  - supports explicit `--gpu-type-id` or constraint-based auto-pick from SDK GPU list
  - supports fallback template handling for SDK variants that lack template-list APIs (uses provided `--template-id`/`--template-name` directly in provision payload)
  - supports env injection (`--env KEY=VALUE`)
  - supports interruptible mode and wait-until-ready polling
- `pod-status`
  - fetches current pod status
- `pod-stop`
  - requests pod stop
- `pod-terminate`
  - requests pod termination/deletion
- SDK smoke flow wrappers:
  - `scripts/runpod_sdk_cycle_start.sh` provisions via SDK and records compatible `provision.json`
    - passes `--image-name` only when non-empty (avoids empty-arg ambiguity)
    - during SSH readiness wait, refreshes pod status via SDK `pod-status` and updates local `provision.json` so delayed public IP / SSH port mappings are picked up before timeout
  - `scripts/runpod_sdk_cycle_stop.sh` stops pod via SDK (`pod-stop`)
  - `scripts/runpod_sdk_cycle_full_smoke.sh` runs:
    - sdk-start -> dataset push -> train -> collect -> local-validate -> sdk-stop
  - these wrappers are intentionally separate from raw `runpod_cycle_start.sh`/`runpod_cycle_stop.sh`

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
  - SDK calls suppress third-party stdout noise (for example raw response debug prints) so CLI output remains valid JSON for wrapper consumers.
- Backward compatibility:
  - adding SDK method probes is acceptable
  - do not remove existing raw component behavior from this SDK component spec

## Known Runtime Constraint
- This component requires the Python `runpod` package in the active environment.
- If missing, it exits with an actionable install message and does not mutate raw API workflows.

## Tests
- `tests/test_runpod_sdk_component.py`
  - verifies nested callable discovery, GPU ranking behavior, template selection, API key dotenv fallback, parser defaults, provision-path behavior with a mocked SDK object, template-fallback behavior when SDK template-list methods are unavailable, non-fallback behavior for runtime template-list errors, and fail-fast guard when neither template id nor image name is available.
- `tests/test_runpod_cycle_scripts.py`
  - verifies SDK smoke wrappers exist, SDK full smoke flow calls SDK start/stop plus shared train/collect validation steps, SDK CLI wrapper help runs from repo-root path, SDK start wrapper only passes `--image-name` when non-empty, SDK start readiness logic refreshes endpoint fields via `pod-status`, and SSH-dependent wrappers fail fast when host `ssh` binary is missing.
- `tests/test_runpod_sdk_guardrails.py`
  - SDK/direct isolation guards: SDK start/stop/full-smoke wrappers must remain bound to `runpod_sdk_component.py` and must not drift to direct `runpod_provision.py` flows.
- `tests/test_runpod_direct_api_guardrails.py`
  - direct API isolation guards run separately to ensure SDK changes do not alter direct start/stop/full-smoke wiring.
