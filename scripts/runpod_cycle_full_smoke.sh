#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-runpod-cycle-$(date -u +%Y%m%dT%H%M%SZ)}"
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
  telemetry_checkpoint "full_smoke_flow" "error" "full smoke flow failed"
  telemetry_event "full_smoke_flow_error" "error" "runpod full smoke flow failed"
}
trap cleanup_on_error EXIT

echo "[runpod-cycle-full] repo=${REPO_ROOT}"
echo "[runpod-cycle-full] run_id=${RUN_ID}"
echo "[runpod-cycle-full] This will provision a RunPod pod, upload dataset, run a short training, collect artifacts, validate locally, and stop the pod."
telemetry_event "full_smoke_flow_start" "info" "runpod full smoke flow started"
telemetry_checkpoint "full_smoke_flow" "running" "full smoke flow started"

telemetry_checkpoint "full_smoke_start" "running" "starting pod"
bash "${REPO_ROOT}/scripts/runpod_cycle_start.sh"
telemetry_checkpoint "full_smoke_start" "done" "pod started"
telemetry_checkpoint "full_smoke_push_dataset" "running" "pushing dataset"
bash "${REPO_ROOT}/scripts/runpod_cycle_push_dataset.sh"
telemetry_checkpoint "full_smoke_push_dataset" "done" "dataset pushed"
telemetry_checkpoint "full_smoke_train" "running" "running training"
bash "${REPO_ROOT}/scripts/runpod_cycle_train.sh"
telemetry_checkpoint "full_smoke_train" "done" "training completed"
telemetry_checkpoint "full_smoke_collect" "running" "collecting artifacts"
bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh"
telemetry_checkpoint "full_smoke_collect" "done" "artifacts collected"
telemetry_checkpoint "full_smoke_local_validate" "running" "validating locally"
bash "${REPO_ROOT}/scripts/runpod_cycle_local_validate.sh"
telemetry_checkpoint "full_smoke_local_validate" "done" "local validation completed"
telemetry_checkpoint "full_smoke_stop" "running" "stopping pod"
bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh"
telemetry_checkpoint "full_smoke_stop" "done" "pod stopped"
FLOW_SUCCESS=1
telemetry_checkpoint "full_smoke_flow" "done" "runpod full smoke flow completed"
telemetry_event "full_smoke_flow_complete" "ok" "runpod full smoke flow completed"

echo "[runpod-cycle-full] completed run_id=${RUN_ID}"
echo "[runpod-cycle-full] report=${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/reports/observations.md"
