#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${LOGS_DIR}"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd rsync
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"

REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_TIMING_LOG="${REMOTE_REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl"
LOCAL_COLLECT_DIR="${RUNPOD_LOCAL_COLLECT_DIR:-${CYCLE_DIR}/collected}"
RSYNC_ARTIFACTS_LOG="${LOGS_DIR}/collect_rsync_run_artifacts.log"
RSYNC_TIMING_LOG="${LOGS_DIR}/collect_rsync_timing.log"

mkdir -p "${LOCAL_COLLECT_DIR}"

printf -v RSYNC_SSH 'ssh -i %q -p %q -o BatchMode=yes -o ConnectTimeout=%q -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o StrictHostKeyChecking=%q -o UserKnownHostsFile=%q' \
  "${SSH_KEY}" "${SSH_PORT}" "${SSH_CONNECT_TIMEOUT}" "${SSH_HOST_KEY_CHECKING}" "${SSH_KNOWN_HOSTS_FILE}"

{
  printf '[runpod-cycle-collect] ssh=%s@%s:%s\n' "${SSH_USER}" "${SSH_HOST}" "${SSH_PORT}"
  printf '[runpod-cycle-collect] remote_run_dir=%s\n' "${REMOTE_RUN_DIR}"
  printf '[runpod-cycle-collect] local_collect_dir=%s/run_artifacts\n' "${LOCAL_COLLECT_DIR}"
} > "${RSYNC_ARTIFACTS_LOG}"

rsync -az --info=stats1 --progress -e "${RSYNC_SSH}" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_RUN_DIR}/" "${LOCAL_COLLECT_DIR}/run_artifacts/" \
  2>&1 | tee -a "${RSYNC_ARTIFACTS_LOG}"

rsync -az --info=stats1 -e "${RSYNC_SSH}" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_TIMING_LOG}" "${LOCAL_COLLECT_DIR}/runpod_phase_times.jsonl" \
  2>&1 | tee "${RSYNC_TIMING_LOG}" || true

runpod_cycle_append_report "${REPORT_MD}" \
  "## Artifact Collection" \
  "- Remote run dir: \`${REMOTE_RUN_DIR}\`" \
  "- Local collect dir: \`${LOCAL_COLLECT_DIR}\`" \
  "- SSH endpoint used: \`${SSH_USER}@${SSH_HOST}:${SSH_PORT}\`" \
  "- Timing log (best effort): \`${LOCAL_COLLECT_DIR}/runpod_phase_times.jsonl\`" \
  "- Rsync artifact log: \`${RSYNC_ARTIFACTS_LOG}\`" \
  "- Rsync timing-log transfer log: \`${RSYNC_TIMING_LOG}\`" \
  ""

echo "[runpod-cycle-collect] local_collect_dir=${LOCAL_COLLECT_DIR}"
