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
DEFAULT_REMOTE_REPO_DIR="${RUNPOD_DEFAULT_REMOTE_REPO_DIR:-/workspace/chess-bot-${RUN_ID}}"

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

if [[ "${RUNPOD_INJECT_LOCAL_SSH_KEY_ENV:-1}" == "1" ]]; then
  SSH_PUBKEY_PATH="${RUNPOD_SSH_PUBKEY_PATH:-$HOME/.ssh/id_ed25519.pub}"
  if [[ -f "${SSH_PUBKEY_PATH}" ]]; then
    SSH_PUBKEY_VALUE="$(<"${SSH_PUBKEY_PATH}")"
    if [[ -n "${SSH_PUBKEY_VALUE}" ]]; then
      cmd+=( --env "AUTHORIZED_KEYS=${SSH_PUBKEY_VALUE}" )
      cmd+=( --env "PUBLIC_KEY=${SSH_PUBKEY_VALUE}" )
    fi
  else
    echo "[runpod-cycle-start] warning: local ssh pubkey not found at ${SSH_PUBKEY_PATH}; continuing without AUTHORIZED_KEYS/PUBLIC_KEY override" >&2
  fi
fi

if [[ "${RUNPOD_SET_UNIQUE_REPO_DIR:-1}" == "1" ]]; then
  cmd+=( --env "REPO_DIR=${DEFAULT_REMOTE_REPO_DIR}" )
fi

if [[ "${RUNPOD_SET_SMOKE_SERVICE_ENVS:-1}" == "1" ]]; then
  cmd+=( --env "START_SSHD=1" )
  cmd+=( --env "START_JUPYTER=0" )
  cmd+=( --env "START_INFERENCE_API=0" )
  cmd+=( --env "START_HF_WATCH=0" )
  cmd+=( --env "START_IDLE_WATCHDOG=0" )
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
POD_NAME_RECORDED="$(runpod_cycle_pod_name "${PROVISION_JSON}")"

runpod_cycle_registry_record \
  "${REPO_ROOT}" \
  "runpod_cycle_start.sh" \
  "start" \
  "RUNNING" \
  "${POD_ID}" \
  "${RUN_ID}" \
  "${POD_NAME_RECORDED:-$POD_NAME}" \
  "${IP}" \
  "${SSH_HOST}" \
  "${SSH_PORT}" \
  "Provisioned via runpod_cycle_start.sh (wait-ready enabled)"

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
  "- Tracked pods registry: \`$(runpod_cycle_registry_file "${REPO_ROOT}")\`" \
  ""

echo "[runpod-cycle-start] run_id=${RUN_ID}"
echo "[runpod-cycle-start] pod_id=${POD_ID}"
echo "[runpod-cycle-start] public_ip=${IP}"
echo "[runpod-cycle-start] ssh_host=${SSH_HOST}"
echo "[runpod-cycle-start] ssh_port=${SSH_PORT}"
