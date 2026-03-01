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
    - during SSH readiness wait, refreshes pod status via SDK `pod-status`, updates local `provision.json`, and recomputes SSH host/port each poll so delayed `runtime.ports` mappings are picked up before timeout
  - `scripts/runpod_sdk_cycle_stop.sh` stops pod via SDK (`pod-stop`)
  - `scripts/runpod_sdk_cycle_full_smoke.sh` runs:
    - sdk-start -> dataset push -> train -> collect -> local-validate -> sdk-stop
  - these wrappers are intentionally separate from raw `runpod_cycle_start.sh`/`runpod_cycle_stop.sh`

## Authentication / Secret Resolution
- API key resolution order:
  1. `--api-key`
  2. env `RUNPOD_API_KEY`
  3. keyring fallback (canonical identity mapping in `specs/chess_bot_secrets_contract.md`)
  4. dotenv fallback (`RUNPOD_SDK_DOTENV_PATH`/`RUNPOD_DOTENV_PATH`/`CHESSBOT_DOTENV_PATH`, then `.env.runpod`, then `.env`)
- Canonical token/key identity mapping is maintained in `specs/chess_bot_secrets_contract.md`.
- Container guidance in this workspace:
  - prefer dotenv provider path (`RUNPOD_SDK_DOTENV_PATH` or `RUNPOD_DOTENV_PATH`) over keyring.
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

## Known Bug (Availability Reporting)
- Observed on 2026-03-01:
  - `scripts/runpod_sdk_component.py gpu-search` returned large GPU catalogs for both `COMMUNITY` and `SECURE`, but every row reported `max_gpu_count=0` and `price_per_hr=0.0`.
  - At the same time, the direct non-SDK path (`scripts/runpod_provision.py gpu-search`) reported normal non-zero capacity and pricing (for example `NVIDIA RTX A5000` with `max_gpu_count=10`, spot price populated).
- Investigation details (2026-03-01):
  - API key resolution in this environment was successful for SDK component (`_resolve_api_key` returned a non-empty key).
  - Separate container keyring probe showed `keyring` module unavailable (`ModuleNotFoundError`) in both `.venv` and system `python3`.
  - SDK probe resolved RunPod auth from dotenv source in-container (`resolved_api_key_source=dotenv`) and still reproduced the all-zero `gpu-search` output.
  - Invalid explicit API key (`--api-key INVALID_TEST_KEY`) caused SDK `gpu-search` to fail unauthorized, confirming SDK path is not silently running unauthenticated.
  - Installed SDK surface on host exposed `runpod.get_gpus`/`runpod.get_gpu` (no `get_gpu_types` method).
  - `runpod.get_gpus()` returned rows with only `id`, `displayName`, `memoryInGb`.
  - `runpod.get_gpu(<id>)` returned richer fields (`communityPrice`, `securePrice`, `communitySpotPrice`, `secureSpotPrice`, `maxGpuCount`, `lowestPrice`).
  - Current component `gpu-search` uses `_sdk_gpu_types -> get_gpus()` and `_rank_gpu_rows`, which defaults missing price/count fields to `0.0`/`0`.
- Conclusion:
  - This bug is primarily a SDK method/data-shape mismatch in availability enrichment, not a container keyring availability problem.
- Current operational stance:
  - Treat SDK `gpu-search` availability fields as bugged/unreliable for real-time capacity decisions until fixed.
  - Use direct non-SDK discovery/provision (`scripts/runpod_provision.py`, `scripts/runpod_cycle_start.sh`, `scripts/runpod_cycle_full_smoke.sh`) as source of truth for availability and launch attempts.

## Tests
- `tests/test_runpod_sdk_component.py`
  - verifies nested callable discovery, GPU ranking behavior, template selection, API key dotenv fallback, parser defaults, provision-path behavior with a mocked SDK object, template-fallback behavior when SDK template-list methods are unavailable, non-fallback behavior for runtime template-list errors, and fail-fast guard when neither template id nor image name is available.
- `tests/test_runpod_cycle_scripts.py`
  - verifies SDK smoke wrappers exist, SDK full smoke flow calls SDK start/stop plus shared train/collect validation steps, SDK CLI wrapper help runs from repo-root path, SDK start wrapper only passes `--image-name` when non-empty, SDK start readiness logic refreshes endpoint fields via `pod-status`, and SSH-dependent wrappers fail fast when host `ssh` binary is missing.
- `tests/test_runpod_sdk_guardrails.py`
  - SDK/direct isolation guards: SDK start/stop/full-smoke wrappers must remain bound to `runpod_sdk_component.py` and must not drift to direct `runpod_provision.py` flows.
- `tests/test_runpod_direct_api_guardrails.py`
  - direct API isolation guards run separately to ensure SDK changes do not alter direct start/stop/full-smoke wiring.
