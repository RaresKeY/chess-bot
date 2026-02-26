#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Opinionated defaults for "just run it" usage. Override via env only when needed.
export RUNPOD_HF_DATASET_REPO_ID="${RUNPOD_HF_DATASET_REPO_ID:-LogicLark-QuantumQuill/chess-bot-datasets}"
export RUNPOD_HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
export RUNPOD_FULL_TRAIN_EPOCHS="${RUNPOD_FULL_TRAIN_EPOCHS:-100}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
export RUNPOD_GPU_MIN_MEMORY_GB="${RUNPOD_GPU_MIN_MEMORY_GB:-24}"
export RUNPOD_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA RTX 6000 Ada Generation}"
export RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"

TEMP_KEY_BASE="${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}"
if [[ ! -f "${TEMP_KEY_BASE}" || ! -f "${TEMP_KEY_BASE}.pub" ]]; then
  rm -f "${TEMP_KEY_BASE}" "${TEMP_KEY_BASE}.pub"
  ssh-keygen -t ed25519 -N "" -f "${TEMP_KEY_BASE}" -C "codex-runpod-temp" >/dev/null
fi
chmod 600 "${TEMP_KEY_BASE}" 2>/dev/null || true
export RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-${TEMP_KEY_BASE}}"
export RUNPOD_SSH_PUBKEY_PATH="${RUNPOD_SSH_PUBKEY_PATH:-${TEMP_KEY_BASE}.pub}"

cat <<EOF
[runpod-full-train-easy] starting full RunPod flow
[runpod-full-train-easy] hf_repo=${RUNPOD_HF_DATASET_REPO_ID}
[runpod-full-train-easy] epochs=${RUNPOD_FULL_TRAIN_EPOCHS}
[runpod-full-train-easy] gpu=${RUNPOD_GPU_TYPE_ID}
[runpod-full-train-easy] temp_ssh_key=${RUNPOD_SSH_KEY}
[runpod-full-train-easy] batch_size_override=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>}
[runpod-full-train-easy] num_workers_override=${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-<unset>}
EOF

exec bash "${REPO_ROOT}/scripts/runpod_cycle_full_train_hf.sh"
