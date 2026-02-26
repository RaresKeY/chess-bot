#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

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

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "mkdir -p '${REMOTE_DATASET_DIR}'"
rsync -az --info=stats1 --progress \
  -e "ssh -i ${SSH_KEY} -p ${SSH_PORT} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/runpod_known_hosts" \
  "${LOCAL_DATASET_DIR}/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DATASET_DIR}/"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Dataset Push" \
  "- Local dataset dir: \`${LOCAL_DATASET_DIR}\`" \
  "- Remote dataset dir: \`${REMOTE_DATASET_DIR}\`" \
  "- SSH endpoint used: \`${SSH_USER}@${SSH_HOST}:${SSH_PORT}\`" \
  ""

echo "[runpod-cycle-push] remote_dataset_dir=${REMOTE_DATASET_DIR}"
