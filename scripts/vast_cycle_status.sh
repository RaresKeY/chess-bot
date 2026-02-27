#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vast_cycle_common.sh"

REPO_ROOT="$(vast_cycle_repo_root)"
PY_BIN="$(vast_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="${VAST_CYCLE_RUN_ID:-$(vast_cycle_run_id)}"
CYCLE_DIR="$(vast_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(vast_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(vast_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
REPORT_DIR="$(dirname "${REPORT_MD}")"
LOGS_DIR="$(vast_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${REPORT_DIR}" "${LOGS_DIR}" "${CYCLE_DIR}"

if [[ ! -f "${PROVISION_JSON}" ]]; then
  echo "[vast-cycle-status] missing provision file: ${PROVISION_JSON}" >&2
  exit 1
fi

INSTANCE_ID="${VAST_INSTANCE_ID:-$(vast_cycle_instance_id "${PROVISION_JSON}")}"
[[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] || { echo "[vast-cycle-status] invalid instance id: ${INSTANCE_ID}" >&2; exit 1; }

OUT_JSON="${CYCLE_DIR}/status_response.json"
"${PY_BIN}" "${REPO_ROOT}/scripts/vast_provision.py" manage-instance --instance-id "${INSTANCE_ID}" --label "$(vast_cycle_instance_label "${PROVISION_JSON}" || true)" >/dev/null 2>&1 || true
"${PY_BIN}" "${REPO_ROOT}/scripts/vast_provision.py" instance-list | tee "${OUT_JSON}"

echo "[vast-cycle-status] instance_id=${INSTANCE_ID}"
echo "[vast-cycle-status] status_json=${OUT_JSON}"
