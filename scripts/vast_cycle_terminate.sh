#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vast_cycle_common.sh"

REPO_ROOT="$(vast_cycle_repo_root)"
PY_BIN="$(vast_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="$(vast_cycle_run_id)"
CYCLE_DIR="$(vast_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(vast_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(vast_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

vast_cycle_require_cmd jq

INSTANCE_ID="${VAST_INSTANCE_ID:-$(vast_cycle_instance_id "${PROVISION_JSON}")}"
[[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] || { echo "[vast-cycle-terminate] invalid instance id: ${INSTANCE_ID}" >&2; exit 1; }

mkdir -p "${CYCLE_DIR}"
OUT_JSON="${CYCLE_DIR}/terminate_response.json"
INSTANCE_LABEL="${VAST_INSTANCE_LABEL:-}"
if [[ -z "${INSTANCE_LABEL}" && -f "${PROVISION_JSON}" ]]; then
  INSTANCE_LABEL="$(vast_cycle_instance_label "${PROVISION_JSON}" || true)"
fi

if "${PY_BIN}" "${REPO_ROOT}/scripts/vast_provision.py" destroy-instance --instance-id "${INSTANCE_ID}" | tee "${OUT_JSON}"; then
  TERM_STATE="TERMINATED"
  TERM_NOTE="Deleted via DELETE /instances/{id}/"
else
  TERM_STATE="TERMINATE_ERROR"
  TERM_NOTE="DELETE /instances/{id}/ failed"
fi

vast_cycle_registry_record \
  "${REPO_ROOT}" \
  "vast_cycle_terminate.sh" \
  "terminate" \
  "${TERM_STATE}" \
  "${INSTANCE_ID}" \
  "${RUN_ID}" \
  "${INSTANCE_LABEL}" \
  "" \
  "" \
  "${TERM_NOTE}"

vast_cycle_append_report "${REPORT_MD}" \
  "## Instance Termination" \
  "- Instance ID: \`${INSTANCE_ID}\`" \
  "- Terminate response: \`${OUT_JSON}\`" \
  "- Tracked instances registry: \`$(vast_cycle_registry_file "${REPO_ROOT}")\`" \
  ""

echo "[vast-cycle-terminate] instance_id=${INSTANCE_ID} state=${TERM_STATE}"
if [[ "${TERM_STATE}" != "TERMINATED" ]]; then
  exit 1
fi
