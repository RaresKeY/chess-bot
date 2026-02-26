#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

POD_NAME="${RUNPOD_POD_NAME:-chess-bot-train}"
TEMPLATE_NAME="${RUNPOD_TEMPLATE_NAME:-chess-bot-training}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
MIN_MEMORY_GB="${RUNPOD_MIN_MEMORY_GB:-24}"
MAX_HOURLY_PRICE="${RUNPOD_MAX_HOURLY_PRICE:-0}"
GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-40}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-15}"
WAIT_READY="${RUNPOD_WAIT_READY:-1}"

cmd=(
  "${PY_BIN}" "${REPO_ROOT}/scripts/runpod_provision.py"
  provision
  --name "${POD_NAME}"
  --cloud-type "${CLOUD_TYPE}"
  --gpu-count "${GPU_COUNT}"
  --min-memory-gb "${MIN_MEMORY_GB}"
  --template-name "${TEMPLATE_NAME}"
  --volume-in-gb "${VOLUME_GB}"
  --container-disk-in-gb "${CONTAINER_DISK_GB}"
)

if [[ "${MAX_HOURLY_PRICE}" != "0" ]]; then
  cmd+=( --max-hourly-price "${MAX_HOURLY_PRICE}" )
fi
if [[ -n "${GPU_TYPE_ID}" ]]; then
  cmd+=( --gpu-type-id "${GPU_TYPE_ID}" )
fi
if [[ "${WAIT_READY}" == "1" ]]; then
  cmd+=( --wait-ready )
else
  cmd+=( --no-wait-ready )
fi

printf '[runpod-quick-launch] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
