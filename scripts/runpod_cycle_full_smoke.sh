#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-runpod-cycle-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUNPOD_CYCLE_RUN_ID="${RUN_ID}"

echo "[runpod-cycle-full] repo=${REPO_ROOT}"
echo "[runpod-cycle-full] run_id=${RUN_ID}"
echo "[runpod-cycle-full] This will provision a RunPod pod, upload dataset, run a short training, collect artifacts, validate locally, and stop the pod."

bash "${REPO_ROOT}/scripts/runpod_cycle_start.sh"
bash "${REPO_ROOT}/scripts/runpod_cycle_push_dataset.sh"
bash "${REPO_ROOT}/scripts/runpod_cycle_train.sh"
bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh"
bash "${REPO_ROOT}/scripts/runpod_cycle_local_validate.sh"
bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh"

echo "[runpod-cycle-full] completed run_id=${RUN_ID}"
echo "[runpod-cycle-full] report=${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/reports/observations.md"
