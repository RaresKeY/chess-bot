#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

telemetry_event() {
  local run_id="$1"
  local ev="$2"
  local st="$3"
  local msg="${4:-}"
  RUNPOD_CYCLE_RUN_ID="${run_id}" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry_emit_event.sh" \
    --event "${ev}" --status "${st}" --message "${msg}" >/dev/null 2>&1 || true
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/runpod_file_transfer.sh pull <remote_path> <local_path>
  bash scripts/runpod_file_transfer.sh push <local_path> <remote_path>
  bash scripts/runpod_file_transfer.sh sync <local_dir> <remote_dir>

Notes:
- Requires RUNPOD_CYCLE_RUN_ID (or defaults to generated run id).
- Uses provision metadata from artifacts/runpod_cycles/<run_id>/provision.json.
- Retries are controlled by RUNPOD_TRANSFER_RETRIES (default 3).

Optional env:
  RUNPOD_TRANSFER_RETRIES=3
  RUNPOD_TRANSFER_RETRY_SLEEP_SECONDS=5
  RUNPOD_TRANSFER_CHECKSUM=0
  RUNPOD_TRANSFER_COMPRESS=1
  RUNPOD_TRANSFER_PARTIAL=1
  RUNPOD_TRANSFER_BWLIMIT_KBPS=<int>
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MODE="${1:-}"
SRC_PATH="${2:-}"
DST_PATH="${3:-}"
[[ -n "${MODE}" && -n "${SRC_PATH}" && -n "${DST_PATH}" ]] || { usage >&2; exit 1; }

case "${MODE}" in
  pull|push|sync) ;;
  *)
    echo "[runpod-file-transfer] unsupported mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${LOGS_DIR}"
LOG_FILE="${LOGS_DIR}/file_transfer_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"

runpod_cycle_require_cmd rsync
runpod_cycle_require_cmd jq
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

[[ -f "${PROVISION_JSON}" ]] || { echo "[runpod-file-transfer] missing provision file: ${PROVISION_JSON}" >&2; exit 1; }

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"

RETRIES="${RUNPOD_TRANSFER_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RUNPOD_TRANSFER_RETRY_SLEEP_SECONDS:-5}"
USE_CHECKSUM="${RUNPOD_TRANSFER_CHECKSUM:-0}"
USE_COMPRESS="${RUNPOD_TRANSFER_COMPRESS:-1}"
USE_PARTIAL="${RUNPOD_TRANSFER_PARTIAL:-1}"
BW_LIMIT_KBPS="${RUNPOD_TRANSFER_BWLIMIT_KBPS:-}"

rsync_opts=( -a --info=stats2,progress2 --human-readable )
if [[ "${USE_COMPRESS}" == "1" ]]; then
  rsync_opts+=( -z )
fi
if [[ "${USE_CHECKSUM}" == "1" ]]; then
  rsync_opts+=( --checksum )
fi
if [[ "${USE_PARTIAL}" == "1" ]]; then
  rsync_opts+=( --partial --append-verify )
fi
if [[ -n "${BW_LIMIT_KBPS}" ]]; then
  rsync_opts+=( --bwlimit "${BW_LIMIT_KBPS}" )
fi
if [[ "${MODE}" == "sync" ]]; then
  rsync_opts+=( --delete )
fi

printf -v RSYNC_SSH 'ssh -i %q -p %q -o BatchMode=yes -o ConnectTimeout=%q -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o StrictHostKeyChecking=%q -o UserKnownHostsFile=%q' \
  "${SSH_KEY}" "${SSH_PORT}" "${SSH_CONNECT_TIMEOUT}" "${SSH_HOST_KEY_CHECKING}" "${SSH_KNOWN_HOSTS_FILE}"

REMOTE_PREFIX="${SSH_USER}@${SSH_HOST}:"
RSYNC_SRC=""
RSYNC_DST=""

case "${MODE}" in
  pull)
    mkdir -p "$(dirname "${DST_PATH}")"
    RSYNC_SRC="${REMOTE_PREFIX}${SRC_PATH}"
    RSYNC_DST="${DST_PATH}"
    ;;
  push)
    [[ -e "${SRC_PATH}" ]] || { echo "[runpod-file-transfer] local source missing: ${SRC_PATH}" >&2; exit 1; }
    RSYNC_SRC="${SRC_PATH}"
    RSYNC_DST="${REMOTE_PREFIX}${DST_PATH}"
    ;;
  sync)
    [[ -d "${SRC_PATH}" ]] || { echo "[runpod-file-transfer] sync source must be local directory: ${SRC_PATH}" >&2; exit 1; }
    RSYNC_SRC="${SRC_PATH%/}/"
    RSYNC_DST="${REMOTE_PREFIX}${DST_PATH%/}/"
    ;;
esac

{
  echo "[runpod-file-transfer] run_id=${RUN_ID}"
  echo "[runpod-file-transfer] mode=${MODE}"
  echo "[runpod-file-transfer] src=${SRC_PATH}"
  echo "[runpod-file-transfer] dst=${DST_PATH}"
  echo "[runpod-file-transfer] retries=${RETRIES} retry_sleep_seconds=${RETRY_SLEEP_SECONDS}"
  echo "[runpod-file-transfer] checksum=${USE_CHECKSUM} compress=${USE_COMPRESS} partial=${USE_PARTIAL}"
  echo "[runpod-file-transfer] bwlimit_kbps=${BW_LIMIT_KBPS:-<unset>}"
} | tee "${LOG_FILE}"
telemetry_event "${RUN_ID}" "file_transfer_start" "info" "mode=${MODE} src=${SRC_PATH} dst=${DST_PATH}"

attempt=1
while true; do
  {
    echo "[runpod-file-transfer] attempt=${attempt}/${RETRIES}"
    printf '[runpod-file-transfer] exec: rsync'
    for arg in "${rsync_opts[@]}"; do printf ' %q' "${arg}"; done
    printf ' -e %q %q %q\n' "${RSYNC_SSH}" "${RSYNC_SRC}" "${RSYNC_DST}"
  } | tee -a "${LOG_FILE}"

  if rsync "${rsync_opts[@]}" -e "${RSYNC_SSH}" "${RSYNC_SRC}" "${RSYNC_DST}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[runpod-file-transfer] success" | tee -a "${LOG_FILE}"
    telemetry_event "${RUN_ID}" "file_transfer_complete" "ok" "mode=${MODE} src=${SRC_PATH} dst=${DST_PATH}"
    break
  fi

  if (( attempt >= RETRIES )); then
    echo "[runpod-file-transfer] failed after ${RETRIES} attempts" | tee -a "${LOG_FILE}" >&2
    telemetry_event "${RUN_ID}" "file_transfer_complete" "error" "mode=${MODE} src=${SRC_PATH} dst=${DST_PATH}"
    exit 1
  fi

  echo "[runpod-file-transfer] retrying in ${RETRY_SLEEP_SECONDS}s" | tee -a "${LOG_FILE}" >&2
  sleep "${RETRY_SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done

echo "[runpod-file-transfer] log=${LOG_FILE}"
