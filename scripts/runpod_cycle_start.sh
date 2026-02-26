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
mkdir -p "${CYCLE_DIR}" "$(dirname "${REPORT_MD}")"

POD_NAME="${RUNPOD_POD_NAME:-chess-bot-cycle-${RUN_ID}}"
TEMPLATE_NAME="${RUNPOD_TEMPLATE_NAME:-chess-bot-training}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 3090}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-40}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-15}"

cmd=(
  "${PY_BIN}" "${REPO_ROOT}/scripts/runpod_provision.py"
  --keyring-service runpod
  --keyring-username RUNPOD_API_KEY
  provision
  --name "${POD_NAME}"
  --cloud-type "${CLOUD_TYPE}"
  --gpu-count "${GPU_COUNT}"
  --gpu-type-id "${GPU_TYPE_ID}"
  --template-name "${TEMPLATE_NAME}"
  --volume-in-gb "${VOLUME_GB}"
  --container-disk-in-gb "${CONTAINER_DISK_GB}"
  --wait-ready
)

if [[ "${RUNPOD_USE_PRESET_ENV:-0}" == "1" ]]; then
  cmd+=( --use-runpod-training-preset-env )
else
  cmd+=( --no-use-runpod-training-preset-env )
fi

for extra_env in ${RUNPOD_START_ENVS:-}; do
  cmd+=( --env "${extra_env}" )
done

printf '[runpod-cycle-start] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${PROVISION_JSON}"

IP="$(runpod_cycle_public_ip "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_USER="$(runpod_cycle_ssh_user)"
POD_ID="$(runpod_cycle_pod_id "${PROVISION_JSON}")"

runpod_cycle_append_report "${REPORT_MD}" \
  "# RunPod Cycle Report (${RUN_ID})" \
  "" \
  "- Date (UTC): $(date -u +%F)" \
  "- Pod name: \`${POD_NAME}\`" \
  "- Pod ID: \`${POD_ID}\`" \
  "- Cloud type: \`${CLOUD_TYPE}\`" \
  "- Requested GPU type: \`${GPU_TYPE_ID}\`" \
  "- Public IP: \`${IP}\`" \
  "- SSH host (effective): \`${SSH_HOST}\`" \
  "- SSH user (effective): \`${SSH_USER}\`" \
  "- SSH port: \`${SSH_PORT}\`" \
  "- Provision record: \`${PROVISION_JSON}\`" \
  ""

echo "[runpod-cycle-start] run_id=${RUN_ID}"
echo "[runpod-cycle-start] pod_id=${POD_ID}"
echo "[runpod-cycle-start] public_ip=${IP}"
echo "[runpod-cycle-start] ssh_host=${SSH_HOST}"
echo "[runpod-cycle-start] ssh_port=${SSH_PORT}"
