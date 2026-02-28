# Chess Bot Cloud Connectivity / Health / Telemetry Checks

## Responsibility
Define a repeatable, modular cloud-check architecture that validates provider connectivity, health tooling, and telemetry command paths with strict timeouts.

## Code Ownership
- Framework entrypoint:
  - `scripts/cloud_connectivity_health_checks.sh`
- Shared helpers:
  - `scripts/cloud_checks/common.sh`
- Provider modules:
  - `scripts/cloud_checks/providers/runpod.sh`
  - `scripts/cloud_checks/providers/vast.sh`
- Backward-compatible RunPod wrapper:
  - `scripts/runpod_connectivity_telemetry_checks.sh`
- Regression wrappers/tests:
  - `scripts/runpod_regression_checks.sh`
  - `tests/test_cloud_connectivity_architecture.py`
  - `tests/test_runpod_connectivity_checks.py`

## Architecture Contract
- One framework script handles lifecycle and timeout policy.
- Provider modules implement two functions:
  - `<provider>_provider_local_checks`
  - `<provider>_provider_live_checks`
- Framework inputs:
  - `CLOUD_CHECK_PROVIDER` (`runpod` default)
  - `CLOUD_CHECK_ENABLE_LIVE` (`0` default)
  - `CLOUD_CHECK_TIMEOUT_SECONDS` (`25` default)
  - `CLOUD_CHECK_RUN_ID` (auto-generated if unset)
- Every check step runs via `timeout`.

## Telemetry / Health Integration
- Framework emits start/complete events and running/done checkpoints via:
  - `scripts/telemetry_control.sh event ...`
  - `scripts/telemetry_control.sh checkpoint ...`
- Telemetry is written under:
  - `artifacts/runpod_cycles/<run_id>/telemetry/`

## Provider Expectations
1. Local checks are deterministic and runnable without live cloud API access.
2. Live checks are opt-in and safe to skip in CI/local.
3. Commands must be short-running and timeout-guarded.
4. Provider scripts should avoid side effects (prefer read/probe calls).

## RunPod Provider (current)
- Local:
  - telemetry status + emit/checkpoint path
  - `runpod_provision.py --help`
- Live:
  - `runpod_provision.py template-list --limit 3`
  - `runpod_provision.py gpu-search --limit 5`
  - `scripts/runpod_cli_doctor.sh`
- Backward compatibility:
  - `scripts/runpod_connectivity_telemetry_checks.sh` forwards to the framework using provider `runpod`.

## Vast Provider (current scaffold)
- Local:
  - `vast_provision.py --help`
- Live:
  - `vast_provision.py offer-search --limit 3`
  - `vast_provision.py instance-list`
- Goal:
  - keep API shape parallel with RunPod checks for cross-cloud operator parity.

## Regression and Operator Commands
- Generic:
  - `bash scripts/cloud_connectivity_health_checks.sh --provider runpod`
  - `bash scripts/cloud_connectivity_health_checks.sh --provider runpod --live`
  - `bash scripts/cloud_connectivity_health_checks.sh --provider vast`
- Existing RunPod compatibility:
  - `bash scripts/runpod_connectivity_telemetry_checks.sh`
