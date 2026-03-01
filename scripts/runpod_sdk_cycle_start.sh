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
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"
mkdir -p "${CYCLE_DIR}" "$(dirname "${REPORT_MD}")"

POD_NAME="${RUNPOD_POD_NAME:-chess-bot-cycle-${RUN_ID}}"
TEMPLATE_NAME="${RUNPOD_SDK_TEMPLATE_NAME:-${RUNPOD_TEMPLATE_NAME:-chess-bot-training}}"
IMAGE_NAME="${RUNPOD_SDK_IMAGE_NAME:-}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 3090}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-40}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-15}"
DEFAULT_REMOTE_REPO_DIR="${RUNPOD_DEFAULT_REMOTE_REPO_DIR:-/workspace/chess-bot-${RUN_ID}}"
INTERRUPTIBLE="${RUNPOD_INTERRUPTIBLE:-0}"
SSH_READY_REQUIRED="${RUNPOD_REQUIRE_SSH_READY:-1}"

if [[ "${SSH_READY_REQUIRED}" == "1" ]]; then
  runpod_cycle_require_cmd ssh
fi

cmd=(
  "${PY_BIN}" "${REPO_ROOT}/scripts/runpod_sdk_component.py"
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
if [[ -n "${IMAGE_NAME}" ]]; then
  cmd+=( --image-name "${IMAGE_NAME}" )
fi
if [[ "${INTERRUPTIBLE}" == "1" ]]; then
  cmd+=( --interruptible )
else
  cmd+=( --no-interruptible )
fi

INJECT_MANAGED_SSH_KEY_ENV="${RUNPOD_INJECT_MANAGED_SSH_KEY_ENV:-1}"
if [[ "${INJECT_MANAGED_SSH_KEY_ENV}" == "1" ]]; then
  SSH_PUBKEY_PATH="$(runpod_cycle_ssh_pubkey_path)"
  if [[ -f "${SSH_PUBKEY_PATH}" ]]; then
    SSH_PUBKEY_VALUE="$(<"${SSH_PUBKEY_PATH}")"
    if [[ -n "${SSH_PUBKEY_VALUE}" ]]; then
      cmd+=( --env "AUTHORIZED_KEYS=${SSH_PUBKEY_VALUE}" )
      cmd+=( --env "PUBLIC_KEY=${SSH_PUBKEY_VALUE}" )
    fi
  else
    echo "[runpod-sdk-cycle-start] warning: managed ssh pubkey not found at ${SSH_PUBKEY_PATH}; continuing without AUTHORIZED_KEYS/PUBLIC_KEY override" >&2
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
  cmd+=( --env "START_OTEL_COLLECTOR=0" )
fi

for extra_env in ${RUNPOD_START_ENVS:-}; do
  cmd+=( --env "${extra_env}" )
done

printf '[runpod-sdk-cycle-start] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${PROVISION_JSON}"

# Compatibility normalization for downstream runpod_cycle_* scripts.
TMP_JSON="${PROVISION_JSON}.tmp"
jq '
  .pod_id = (.pod_id // .create_response.id // .create_response.podId // .create_response.pod_id // "") |
  .pod_status = (
    .pod_status
    // .create_response.pod_status
    // .create_response.podStatus
    // .create_response
    // {}
  )
' "${PROVISION_JSON}" > "${TMP_JSON}"
mv "${TMP_JSON}" "${PROVISION_JSON}"

IP="$(runpod_cycle_public_ip "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_USER="$(runpod_cycle_ssh_user)"
POD_ID="$(runpod_cycle_pod_id "${PROVISION_JSON}")"
POD_NAME_RECORDED="$(runpod_cycle_pod_name "${PROVISION_JSON}")"

if [[ "${SSH_READY_REQUIRED}" == "1" ]]; then
  SSH_KEY="$(runpod_cycle_ssh_key)"
  SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
  SSH_READY_TIMEOUT_SECONDS="${RUNPOD_SSH_READY_TIMEOUT_SECONDS:-360}"
  SSH_READY_POLL_SECONDS="${RUNPOD_SSH_READY_POLL_SECONDS:-8}"
  SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
  SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"
  SSH_READY_DEADLINE=$(( $(date +%s) + SSH_READY_TIMEOUT_SECONDS ))

  sdk_refresh_status_json() {
    local status_json tmp_merge
    status_json="${PROVISION_JSON}.status.json"
    tmp_merge="${PROVISION_JSON}.status.merge.tmp"
    if "${PY_BIN}" "${REPO_ROOT}/scripts/runpod_sdk_component.py" \
      --keyring-service runpod \
      --keyring-username RUNPOD_API_KEY \
      pod-status --pod-id "${POD_ID}" > "${status_json}" 2>/dev/null; then
      jq --slurpfile s "${status_json}" '
        .pod_status = ($s[0].pod_status // .pod_status // {})
      ' "${PROVISION_JSON}" > "${tmp_merge}" && mv "${tmp_merge}" "${PROVISION_JSON}"
    fi
    rm -f "${status_json}" "${tmp_merge}" 2>/dev/null || true
  }

  sdk_mapped_ssh_port() {
    jq -r '((.pod_status.portMappings["22"] // .create_response.portMappings["22"] // "") | tostring)' "${PROVISION_JSON}" 2>/dev/null || true
  }

  while true; do
    sdk_refresh_status_json
    # Recompute SSH endpoint each poll so late runtime port mappings are applied.
    SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
    SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
    mapped_ssh_port="$(sdk_mapped_ssh_port)"
    if [[ -n "${mapped_ssh_port}" && "${mapped_ssh_port}" != "null" ]]; then
      SSH_PORT="${mapped_ssh_port}"
    fi
    if [[ -n "${SSH_HOST}" && "${SSH_HOST}" != "null" ]] && ssh \
      -i "${SSH_KEY}" \
      -p "${SSH_PORT}" \
      -o BatchMode=yes \
      -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
      -o IdentitiesOnly=yes \
      -o AddKeysToAgent=no \
      -o IdentityAgent=none \
      -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" \
      -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
      "${SSH_USER}@${SSH_HOST}" "echo ready" >/dev/null 2>&1; then
      break
    fi
    if (( $(date +%s) >= SSH_READY_DEADLINE )); then
      echo "[runpod-sdk-cycle-start] ssh readiness timed out for ${SSH_USER}@${SSH_HOST}:${SSH_PORT}" >&2
      if [[ "${RUNPOD_TERMINATE_ON_SSH_NOT_READY:-1}" == "1" ]]; then
        echo "[runpod-sdk-cycle-start] terminating pod after ssh readiness timeout: ${POD_ID}" >&2
        RUNPOD_CYCLE_RUN_ID="${RUN_ID}" RUNPOD_POD_ID="${POD_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh" >/dev/null 2>&1 || true
      fi
      exit 1
    fi
    sleep "${SSH_READY_POLL_SECONDS}"
  done
fi

runpod_cycle_registry_record \
  "${REPO_ROOT}" \
  "runpod_sdk_cycle_start.sh" \
  "start" \
  "RUNNING" \
  "${POD_ID}" \
  "${RUN_ID}" \
  "${POD_NAME_RECORDED:-$POD_NAME}" \
  "${IP}" \
  "${SSH_HOST}" \
  "${SSH_PORT}" \
  "Provisioned via runpod_sdk_cycle_start.sh (SDK component, wait-ready enabled)"

runpod_cycle_append_report "${REPORT_MD}" \
  "# RunPod SDK Cycle Report (${RUN_ID})" \
  "" \
  "- Date (UTC): $(date -u +%F)" \
  "- Component path: \`runpod_sdk_component.py\`" \
  "- Pod name: \`${POD_NAME}\`" \
  "- Pod ID: \`${POD_ID}\`" \
  "- Cloud type: \`${CLOUD_TYPE}\`" \
  "- Interruptible (spot): \`${INTERRUPTIBLE}\`" \
  "- Requested GPU type: \`${GPU_TYPE_ID}\`" \
  "- Public IP: \`${IP}\`" \
  "- SSH host (effective): \`${SSH_HOST}\`" \
  "- SSH user (effective): \`${SSH_USER}\`" \
  "- SSH port: \`${SSH_PORT}\`" \
  "- Provision record: \`${PROVISION_JSON}\`" \
  "- Tracked pods registry: \`$(runpod_cycle_registry_file "${REPO_ROOT}")\`" \
  ""

echo "[runpod-sdk-cycle-start] run_id=${RUN_ID}"
echo "[runpod-sdk-cycle-start] pod_id=${POD_ID}"
echo "[runpod-sdk-cycle-start] public_ip=${IP}"
echo "[runpod-sdk-cycle-start] ssh_host=${SSH_HOST}"
echo "[runpod-sdk-cycle-start] ssh_port=${SSH_PORT}"
