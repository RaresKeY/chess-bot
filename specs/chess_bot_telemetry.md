# Chess Bot Telemetry and Watchdog Controls

## Responsibility
Define the host-side telemetry event/checkpoint contract for RunPod cycle flows, the central status/watchdog controls, and container-level OpenTelemetry collector behavior.

## Code Ownership
- Telemetry components:
  - `scripts/telemetry/telemetry_common.sh`
  - `scripts/telemetry_emit_event.sh`
  - `scripts/telemetry_checkpoint.sh`
  - `scripts/telemetry_healthchecks_ping.sh`
  - `scripts/telemetry_status.sh`
  - `scripts/telemetry_watchdog.sh`
  - `scripts/telemetry_control.sh`
- Backward-compat alias:
  - `scripts/runpod_cycle_watchdog.sh`
- Flow integrations:
  - `scripts/runpod_full_train_easy.sh`
  - `scripts/runpod_cycle_full_train_hf.sh`
  - `scripts/runpod_cycle_benchmark_matrix.sh`
  - `scripts/runpod_cycle_full_smoke.sh`
  - `scripts/runpod_cycle_train.sh`
  - `scripts/runpod_file_transfer.sh`
- Container telemetry runtime:
  - `deploy/runpod_cloud_training/otel-collector-config.yaml`
  - `deploy/runpod_cloud_training/entrypoint.sh`
  - `deploy/runpod_cloud_training/healthchecks_ping.sh`
  - `deploy/runpod_cloud_training/Dockerfile`

## Host Telemetry Contract
- Telemetry files are written under:
  - `artifacts/runpod_cycles/<run_id>/telemetry/`
- Event stream:
  - `events.jsonl`
  - record fields: `ts_utc`, `ts_epoch_ms`, `run_id`, `event`, `status`, `message`, `extra`
- Checkpoint stream:
  - `checkpoints.jsonl`
  - record fields: `ts_utc`, `ts_epoch_ms`, `run_id`, `checkpoint`, `state`, `note`
- Healthchecks ping URL:
  - `RUNPOD_HEALTHCHECKS_URL` (fallback `HEALTHCHECKS_URL`)
  - kinds: `start`, `success`, `fail`, `log`

## Central Control Scripts
- Single command surface:
  - `bash scripts/telemetry_control.sh <status|event|checkpoint|watchdog|health> ...`
- Status snapshot:
  - `bash scripts/telemetry_status.sh [--run-id <id>] [--json]`
  - combines latest telemetry rows with `runpod_cycle_status.sh --no-write`
- Watchdog:
  - `bash scripts/telemetry_watchdog.sh`
  - stall actions: `none`, `collect`, `stop`, `terminate`, `collect-stop`, `collect-terminate`
- Legacy script:
  - `scripts/runpod_cycle_watchdog.sh` delegates to `telemetry_watchdog.sh`

## Flow Checkpointing
- Full HF flow and benchmark matrix emit stage checkpoints and events.
- Normal modular smoke/train flows also emit lifecycle checkpoints:
  - start, dataset push/fetch, train, collect, validate, stop, done/error
- File transfers emit start/complete telemetry events.

## Container OpenTelemetry Collector
- Image includes `otelcol-contrib` binary.
- Entrypoint defaults to enabling collector:
  - `START_OTEL_COLLECTOR=1`
- Config path:
  - `OTEL_CONFIG_PATH` override
  - default: `${RUNPOD_MODULE_DIR}/otel-collector-config.yaml`
- Local collector export path:
  - `OTEL_FILE_EXPORT_PATH` (default `artifacts/telemetry/otel/collector.jsonl` under `REPO_DIR`)
- Default config enables:
  - `hostmetrics` receiver (CPU/memory/fs/disk/network/process)
  - `otlp` receiver (grpc/http)
  - `batch` processor
  - `logging` and `file` exporters

## Container Healthchecks Utility
- Utility script:
  - `bash deploy/runpod_cloud_training/healthchecks_ping.sh <start|success|fail|log> [message]`
- Entrypoint sends best-effort startup/failure/shutdown pings when URL is configured.

## Checkpoint-Oriented Operations Guidance
1. Use `telemetry_control.sh status --json` as the canonical run snapshot for automation.
2. Treat checkpoint state transitions (`running` -> `done`/`error`) as operator checkpoints.
3. Prefer watchdog `collect-stop` for unattended runs to preserve artifacts before stop.
4. Keep telemetry additions append-only and JSONL compatible.

## Testing Expectations
- Add regression tests for:
  - telemetry script existence/command routing
  - watchdog alias contract
  - checkpoint/event hooks in modified cycle scripts
  - entrypoint OpenTelemetry/healthchecks controls
  - Dockerfile telemetry runtime dependencies
- Connectivity timeout checks category:
  - framework script: `scripts/cloud_connectivity_health_checks.sh`
  - RunPod compatibility wrapper: `scripts/runpod_connectivity_telemetry_checks.sh`
  - local telemetry path checks always run
  - live RunPod connectivity probes are opt-in (`RUNPOD_ENABLE_LIVE_CONNECTIVITY_CHECKS=1`)
  - every probe must be timeout-guarded (`RUNPOD_CONNECTIVITY_TIMEOUT_SECONDS`)
