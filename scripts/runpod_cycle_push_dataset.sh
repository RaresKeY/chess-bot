#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${LOGS_DIR}"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd rsync

LOCAL_DATASET_DIR="${RUNPOD_LOCAL_DATASET_DIR:-${REPO_ROOT}/data/dataset/_smoke_runpod}"
REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_DATASET_NAME="${RUNPOD_REMOTE_DATASET_NAME:-_smoke_runpod_${RUN_ID}}"
REMOTE_DATASET_DIR="${REMOTE_REPO_DIR}/data/dataset/${REMOTE_DATASET_NAME}"

[[ -f "${LOCAL_DATASET_DIR}/train.jsonl" ]] || { echo "[runpod-cycle-push] missing ${LOCAL_DATASET_DIR}/train.jsonl" >&2; exit 1; }
[[ -f "${LOCAL_DATASET_DIR}/val.jsonl" ]] || { echo "[runpod-cycle-push] missing ${LOCAL_DATASET_DIR}/val.jsonl" >&2; exit 1; }

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_OPTS=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/runpod_known_hosts)

REMOTE_READY_TIMEOUT_SECONDS="${RUNPOD_REMOTE_READY_TIMEOUT_SECONDS:-300}"
REMOTE_READY_POLL_SECONDS="${RUNPOD_REMOTE_READY_POLL_SECONDS:-5}"
remote_ready_deadline=$(( $(date +%s) + REMOTE_READY_TIMEOUT_SECONDS ))
READY_CHECK_LOG="${LOGS_DIR}/push_dataset_ready_check.log"
RSYNC_LOG="${LOGS_DIR}/push_dataset_rsync.log"

while true; do
  if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
    "test -d '${REMOTE_REPO_DIR}' && test -w '${REMOTE_REPO_DIR}' && mkdir -p '${REMOTE_DATASET_DIR}'" >/dev/null 2>&1; then
    break
  fi
  if (( $(date +%s) >= remote_ready_deadline )); then
    echo "[runpod-cycle-push] remote repo not writable yet after ${REMOTE_READY_TIMEOUT_SECONDS}s: ${REMOTE_REPO_DIR}" >&2
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
      "ls -ld '${REMOTE_REPO_DIR}' 2>/dev/null || true; id; ps aux | grep '[e]ntrypoint.sh' || true" \
      2>&1 | tee -a "${READY_CHECK_LOG}" || true
    exit 1
  fi
  echo "[runpod-cycle-push] waiting for remote repo readiness: ${REMOTE_REPO_DIR}" >&2
  sleep "${REMOTE_READY_POLL_SECONDS}"
done

{
  printf '[runpod-cycle-push] local=%s\n' "${LOCAL_DATASET_DIR}"
  printf '[runpod-cycle-push] remote=%s\n' "${REMOTE_DATASET_DIR}"
  printf '[runpod-cycle-push] ssh=%s@%s:%s\n' "${SSH_USER}" "${SSH_HOST}" "${SSH_PORT}"
} > "${RSYNC_LOG}"

rsync -az --info=stats1 --progress \
  -e "ssh -i ${SSH_KEY} -p ${SSH_PORT} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/runpod_known_hosts" \
  "${LOCAL_DATASET_DIR}/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DATASET_DIR}/" \
  2>&1 | tee -a "${RSYNC_LOG}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Dataset Push" \
  "- Local dataset dir: \`${LOCAL_DATASET_DIR}\`" \
  "- Remote dataset dir: \`${REMOTE_DATASET_DIR}\`" \
  "- SSH endpoint used: \`${SSH_USER}@${SSH_HOST}:${SSH_PORT}\`" \
  "- Ready-check log: \`${READY_CHECK_LOG}\`" \
  "- Rsync log: \`${RSYNC_LOG}\`" \
  ""

echo "[runpod-cycle-push] remote_dataset_dir=${REMOTE_DATASET_DIR}"
