#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Opinionated defaults for "just run it" usage. Override via env only when needed.
export RUNPOD_HF_DATASET_REPO_ID="${RUNPOD_HF_DATASET_REPO_ID:-LogicLark-QuantumQuill/chess-bot-datasets}"
export RUNPOD_HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
export RUNPOD_HF_DATASET_SCHEMA_FILTER="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-auto}"
export RUNPOD_FULL_TRAIN_EPOCHS="${RUNPOD_FULL_TRAIN_EPOCHS:-100}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
export RUNPOD_GPU_MIN_MEMORY_GB="${RUNPOD_GPU_MIN_MEMORY_GB:-24}"
export RUNPOD_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 5090}"
export RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"
export RUNPOD_FULL_TRAIN_NPROC_PER_NODE="${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${RUNPOD_GPU_COUNT}}"
export RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

cat <<EOF
[runpod-full-train-easy] starting full RunPod flow
[runpod-full-train-easy] hf_repo=${RUNPOD_HF_DATASET_REPO_ID}
[runpod-full-train-easy] hf_prefix=${RUNPOD_HF_DATASET_PATH_PREFIX}
[runpod-full-train-easy] hf_schema_filter=${RUNPOD_HF_DATASET_SCHEMA_FILTER}
[runpod-full-train-easy] epochs=${RUNPOD_FULL_TRAIN_EPOCHS}
[runpod-full-train-easy] gpu=${RUNPOD_GPU_TYPE_ID}
[runpod-full-train-easy] gpu_count=${RUNPOD_GPU_COUNT}
[runpod-full-train-easy] train_nproc_per_node=${RUNPOD_FULL_TRAIN_NPROC_PER_NODE}
[runpod-full-train-easy] temp_ssh_key=$(runpod_cycle_ssh_key)
[runpod-full-train-easy] batch_size_override=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>}
[runpod-full-train-easy] num_workers_override=${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-<unset>}
[runpod-full-train-easy] max_total_rows=${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS:-<unset>}
[runpod-full-train-easy] runtime_max_samples_per_game=${RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME:-<unset>}
[runpod-full-train-easy] final_report=artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md
EOF

exec bash "${REPO_ROOT}/scripts/runpod_cycle_full_train_hf.sh"
