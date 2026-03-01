#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

runpod_cycle_require_cmd jq

POD_ID="${RUNPOD_POD_ID:-$(runpod_cycle_pod_id "${PROVISION_JSON}")}"
[[ -n "${POD_ID}" ]] || { echo "[runpod-sdk-cycle-stop] missing pod id (set RUNPOD_POD_ID or provide provision json)" >&2; exit 1; }

mkdir -p "${CYCLE_DIR}"
OUT_JSON="${CYCLE_DIR}/stop_response.json"
POD_NAME="${RUNPOD_POD_NAME:-}"
if [[ -z "${POD_NAME}" && -f "${PROVISION_JSON}" ]]; then
  POD_NAME="$(runpod_cycle_pod_name "${PROVISION_JSON}" || true)"
fi

"${PY_BIN}" "${REPO_ROOT}/scripts/runpod_sdk_component.py" \
  --keyring-service runpod \
  --keyring-username RUNPOD_API_KEY \
  pod-stop \
  --pod-id "${POD_ID}" | tee "${OUT_JSON}"

STOP_STATE="STOP_REQUESTED"
STOP_NOTE="Requested pod stop via SDK component; pod storage may still incur charges until terminated"
if jq -e '.stop_response' "${OUT_JSON}" >/dev/null 2>&1; then
  STOP_STATE="STOPPED"
  STOP_NOTE="SDK pod-stop returned response; pod storage may still incur charges until terminated"
fi

runpod_cycle_registry_record \
  "${REPO_ROOT}" \
  "runpod_sdk_cycle_stop.sh" \
  "stop" \
  "${STOP_STATE}" \
  "${POD_ID}" \
  "${RUN_ID}" \
  "${POD_NAME}" \
  "" \
  "" \
  "" \
  "${STOP_NOTE}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Pod Stop (SDK Component)" \
  "- Pod ID: \`${POD_ID}\`" \
  "- Stop response: \`${OUT_JSON}\`" \
  "- Tracked pods registry: \`$(runpod_cycle_registry_file "${REPO_ROOT}")\`" \
  "- Note: RunPod stop halts compute but can continue storage charges; use tracked terminate script to delete pods when done." \
  ""

echo "[runpod-sdk-cycle-stop] pod_id=${POD_ID}"
