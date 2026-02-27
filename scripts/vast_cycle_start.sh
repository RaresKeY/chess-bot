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
mkdir -p "${CYCLE_DIR}" "$(dirname "${REPORT_MD}")"

INSTANCE_LABEL="${VAST_INSTANCE_LABEL:-chess-bot-vast-${RUN_ID}}"
GPU_COUNT="${VAST_GPU_COUNT:-1}"
GPU_NAME="${VAST_GPU_NAME:-}"
MAX_DPH_TOTAL="${VAST_MAX_DPH_TOTAL:-0}"
MIN_RELIABILITY="${VAST_MIN_RELIABILITY:-0}"
MIN_GPU_RAM_GB="${VAST_MIN_GPU_RAM_GB:-0}"
IMAGE="${VAST_IMAGE:-ghcr.io/rareskey/chess-bot-runpod:latest}"
DISK_GB="${VAST_DISK_GB:-40}"
RUNTYPE="${VAST_RUNTYPE:-ssh}"
TARGET_STATE="${VAST_TARGET_STATE:-running}"

cmd=(
  "${PY_BIN}" "${REPO_ROOT}/scripts/vast_provision.py"
  --keyring-service vast
  --keyring-username VAST_API_KEY
  provision
  --label "${INSTANCE_LABEL}"
  --gpu-count "${GPU_COUNT}"
  --max-dph-total "${MAX_DPH_TOTAL}"
  --min-reliability "${MIN_RELIABILITY}"
  --min-gpu-ram-gb "${MIN_GPU_RAM_GB}"
  --image "${IMAGE}"
  --disk "${DISK_GB}"
  --runtype "${RUNTYPE}"
  --target-state "${TARGET_STATE}"
  --wait-ready
)

if [[ -n "${GPU_NAME}" ]]; then
  cmd+=( --gpu-name "${GPU_NAME}" )
fi
if [[ -n "${VAST_OFFER_ID:-}" ]]; then
  cmd+=( --offer-id "${VAST_OFFER_ID}" )
fi
for extra_env in ${VAST_START_ENVS:-}; do
  cmd+=( --env "${extra_env}" )
done

printf '[vast-cycle-start] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${PROVISION_JSON}"

INSTANCE_ID="$(vast_cycle_instance_id "${PROVISION_JSON}")"
SSH_HOST="$(vast_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(vast_cycle_ssh_port "${PROVISION_JSON}")"
SSH_USER="$(vast_cycle_ssh_user)"

vast_cycle_registry_record \
  "${REPO_ROOT}" \
  "vast_cycle_start.sh" \
  "start" \
  "RUNNING" \
  "${INSTANCE_ID}" \
  "${RUN_ID}" \
  "${INSTANCE_LABEL}" \
  "${SSH_HOST}" \
  "${SSH_PORT}" \
  "Provisioned via vast_cycle_start.sh"

vast_cycle_append_report "${REPORT_MD}" \
  "# Vast Cycle Report (${RUN_ID})" \
  "" \
  "- Date (UTC): $(date -u +%F)" \
  "- Instance label: \`${INSTANCE_LABEL}\`" \
  "- Instance ID: \`${INSTANCE_ID}\`" \
  "- GPU count: \`${GPU_COUNT}\`" \
  "- GPU name filter: \`${GPU_NAME:-<none>}\`" \
  "- Max dph_total: \`${MAX_DPH_TOTAL}\`" \
  "- SSH host: \`${SSH_HOST}\`" \
  "- SSH user: \`${SSH_USER}\`" \
  "- SSH port: \`${SSH_PORT}\`" \
  "- Provision record: \`${PROVISION_JSON}\`" \
  "- Tracked instances registry: \`$(vast_cycle_registry_file "${REPO_ROOT}")\`" \
  ""

echo "[vast-cycle-start] run_id=${RUN_ID}"
echo "[vast-cycle-start] instance_id=${INSTANCE_ID}"
echo "[vast-cycle-start] ssh_host=${SSH_HOST}"
echo "[vast-cycle-start] ssh_port=${SSH_PORT}"
