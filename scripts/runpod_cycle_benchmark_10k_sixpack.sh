#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-runpod-bench-10k6-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUNPOD_CYCLE_RUN_ID="${RUN_ID}"

export RUNPOD_BENCH_EPOCHS="${RUNPOD_BENCH_EPOCHS:-10}"
export RUNPOD_BENCH_MAX_TOTAL_ROWS="${RUNPOD_BENCH_MAX_TOTAL_ROWS:-100000}"
export RUNPOD_BENCH_TRIALS="${RUNPOD_BENCH_TRIALS:-fp32,fp16,bf16,fp32_sparse,fp16_sparse,bf16_sparse}"
export RUNPOD_BENCH_SPARSITY_L1_LAMBDAS="${RUNPOD_BENCH_SPARSITY_L1_LAMBDAS:-1e-5,5e-5,1e-4}"
export RUNPOD_BENCH_RUNTIME_MAX_SAMPLES_PER_GAME="${RUNPOD_BENCH_RUNTIME_MAX_SAMPLES_PER_GAME:-0}"
export RUNPOD_HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
export RUNPOD_BENCH_HF_DATASET_NAME="${RUNPOD_BENCH_HF_DATASET_NAME:-elite_2025-11_game}"
export RUNPOD_BENCH_STOP_POD="${RUNPOD_BENCH_STOP_POD:-0}"
export RUNPOD_BENCH_TERMINATE_POD="${RUNPOD_BENCH_TERMINATE_POD:-1}"
export RUNPOD_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA A40}"
export RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"
export RUNPOD_FULL_TRAIN_NPROC_PER_NODE="${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${RUNPOD_GPU_COUNT}}"
export RUNPOD_BENCH_BATCH_SIZE="${RUNPOD_BENCH_BATCH_SIZE:-auto}"
export RUNPOD_BENCH_NUM_WORKERS="${RUNPOD_BENCH_NUM_WORKERS:-4}"
export RUNPOD_BENCH_TRANSFER_TOOL="${RUNPOD_BENCH_TRANSFER_TOOL:-rclone}"
export RUNPOD_BENCH_TRANSFER_RETRIES="${RUNPOD_BENCH_TRANSFER_RETRIES:-3}"
export RUNPOD_BENCH_TRANSFER_TIMEOUT_SECONDS="${RUNPOD_BENCH_TRANSFER_TIMEOUT_SECONDS:-1200}"
export RUNPOD_BENCH_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS="${RUNPOD_BENCH_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS:-0}"
export RUNPOD_BENCH_COLLECT_INCLUDE_EPOCH_CHECKPOINTS="${RUNPOD_BENCH_COLLECT_INCLUDE_EPOCH_CHECKPOINTS:-0}"
export RUNPOD_BENCH_SKIP_FINAL_COLLECT="${RUNPOD_BENCH_SKIP_FINAL_COLLECT:-0}"

echo "[runpod-bench-10k6] run_id=${RUN_ID}"
echo "[runpod-bench-10k6] gpu_type_id=${RUNPOD_GPU_TYPE_ID} gpu_count=${RUNPOD_GPU_COUNT} nproc=${RUNPOD_FULL_TRAIN_NPROC_PER_NODE}"
echo "[runpod-bench-10k6] trials=${RUNPOD_BENCH_TRIALS}"
echo "[runpod-bench-10k6] sparsity_l1_lambdas=${RUNPOD_BENCH_SPARSITY_L1_LAMBDAS}"
echo "[runpod-bench-10k6] epochs=${RUNPOD_BENCH_EPOCHS} max_total_rows=${RUNPOD_BENCH_MAX_TOTAL_ROWS} batch_size=${RUNPOD_BENCH_BATCH_SIZE} num_workers=${RUNPOD_BENCH_NUM_WORKERS}"
echo "[runpod-bench-10k6] transfer_tool=${RUNPOD_BENCH_TRANSFER_TOOL} transfer_retries=${RUNPOD_BENCH_TRANSFER_RETRIES} transfer_timeout_s=${RUNPOD_BENCH_TRANSFER_TIMEOUT_SECONDS} include_epoch_ckpt=${RUNPOD_BENCH_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS}"
echo "[runpod-bench-10k6] hf_path_prefix=${RUNPOD_HF_DATASET_PATH_PREFIX} hf_dataset_name=${RUNPOD_BENCH_HF_DATASET_NAME} terminate_pod=${RUNPOD_BENCH_TERMINATE_POD}"

bash "${REPO_ROOT}/scripts/runpod_cycle_benchmark_matrix.sh"

SUMMARY_MD="${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/benchmarks/trial_summary.md"
SUMMARY_JSONL="${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/benchmarks/trial_summary.jsonl"
TELEMETRY_STATUS="${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/reports/telemetry_status_end.json"
mkdir -p "$(dirname "${TELEMETRY_STATUS}")"
RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/telemetry_status.sh" --json > "${TELEMETRY_STATUS}" || true

echo "[runpod-bench-10k6] summary_md=${SUMMARY_MD}"
echo "[runpod-bench-10k6] summary_jsonl=${SUMMARY_JSONL}"
echo "[runpod-bench-10k6] telemetry_status=${TELEMETRY_STATUS}"
