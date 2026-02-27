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
[[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] || { echo "[vast-cycle-stop] invalid instance id: ${INSTANCE_ID}" >&2; exit 1; }

mkdir -p "${CYCLE_DIR}"
OUT_JSON="${CYCLE_DIR}/stop_response.json"
INSTANCE_LABEL="${VAST_INSTANCE_LABEL:-}"
if [[ -z "${INSTANCE_LABEL}" && -f "${PROVISION_JSON}" ]]; then
  INSTANCE_LABEL="$(vast_cycle_instance_label "${PROVISION_JSON}" || true)"
fi

"${PY_BIN}" "${REPO_ROOT}/scripts/vast_provision.py" manage-instance --instance-id "${INSTANCE_ID}" --state stopped | tee "${OUT_JSON}"

vast_cycle_registry_record \
  "${REPO_ROOT}" \
  "vast_cycle_stop.sh" \
  "stop" \
  "STOPPED" \
  "${INSTANCE_ID}" \
  "${RUN_ID}" \
  "${INSTANCE_LABEL}" \
  "" \
  "" \
  "Requested stop via PUT /instances/{id}/ state=stopped"

vast_cycle_append_report "${REPORT_MD}" \
  "## Instance Stop" \
  "- Instance ID: \`${INSTANCE_ID}\`" \
  "- Stop response: \`${OUT_JSON}\`" \
  "- Tracked instances registry: \`$(vast_cycle_registry_file "${REPO_ROOT}")\`" \
  ""

echo "[vast-cycle-stop] instance_id=${INSTANCE_ID}"
