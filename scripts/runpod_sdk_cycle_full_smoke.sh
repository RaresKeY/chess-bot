#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-runpod-sdk-cycle-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUNPOD_CYCLE_RUN_ID="${RUN_ID}"
FLOW_SUCCESS=0

telemetry_event() {
  local ev="$1"
  local st="$2"
  local msg="${3:-}"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/telemetry_emit_event.sh" \
    --event "${ev}" --status "${st}" --message "${msg}" >/dev/null 2>&1 || true
}

telemetry_checkpoint() {
  local name="$1"
  local state="$2"
  local note="${3:-}"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/telemetry_checkpoint.sh" \
    --name "${name}" --state "${state}" --note "${note}" >/dev/null 2>&1 || true
}

cleanup_on_error() {
  if [[ "${FLOW_SUCCESS}" == "1" ]]; then
    return 0
  fi
  telemetry_checkpoint "full_smoke_sdk_flow" "error" "runpod sdk full smoke flow failed"
  telemetry_event "full_smoke_sdk_flow_error" "error" "runpod sdk full smoke flow failed"
}
trap cleanup_on_error EXIT

echo "[runpod-sdk-cycle-full] repo=${REPO_ROOT}"
echo "[runpod-sdk-cycle-full] run_id=${RUN_ID}"
echo "[runpod-sdk-cycle-full] This will provision via SDK, upload dataset, run short training, collect artifacts, validate locally, and stop the pod via SDK."
telemetry_event "full_smoke_sdk_flow_start" "info" "runpod sdk full smoke flow started"
telemetry_checkpoint "full_smoke_sdk_flow" "running" "sdk full smoke flow started"

telemetry_checkpoint "full_smoke_sdk_start" "running" "starting pod via sdk"
bash "${REPO_ROOT}/scripts/runpod_sdk_cycle_start.sh"
telemetry_checkpoint "full_smoke_sdk_start" "done" "pod started via sdk"
telemetry_checkpoint "full_smoke_sdk_push_dataset" "running" "pushing dataset"
bash "${REPO_ROOT}/scripts/runpod_cycle_push_dataset.sh"
telemetry_checkpoint "full_smoke_sdk_push_dataset" "done" "dataset pushed"
telemetry_checkpoint "full_smoke_sdk_train" "running" "running training"
bash "${REPO_ROOT}/scripts/runpod_cycle_train.sh"
telemetry_checkpoint "full_smoke_sdk_train" "done" "training completed"
telemetry_checkpoint "full_smoke_sdk_collect" "running" "collecting artifacts"
bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh"
telemetry_checkpoint "full_smoke_sdk_collect" "done" "artifacts collected"
telemetry_checkpoint "full_smoke_sdk_local_validate" "running" "validating locally"
bash "${REPO_ROOT}/scripts/runpod_cycle_local_validate.sh"
telemetry_checkpoint "full_smoke_sdk_local_validate" "done" "local validation completed"
telemetry_checkpoint "full_smoke_sdk_stop" "running" "stopping pod via sdk"
bash "${REPO_ROOT}/scripts/runpod_sdk_cycle_stop.sh"
telemetry_checkpoint "full_smoke_sdk_stop" "done" "pod stopped via sdk"
FLOW_SUCCESS=1
telemetry_checkpoint "full_smoke_sdk_flow" "done" "runpod sdk full smoke flow completed"
telemetry_event "full_smoke_sdk_flow_complete" "ok" "runpod sdk full smoke flow completed"

echo "[runpod-sdk-cycle-full] completed run_id=${RUN_ID}"
echo "[runpod-sdk-cycle-full] report=${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/reports/observations.md"
