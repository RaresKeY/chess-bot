#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-easy-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUNPOD_CYCLE_RUN_ID="${RUN_ID}"

# Cheap/fast defaults for end-to-end smoke validation; caller can override any env.
export RUNPOD_FULL_TRAIN_EPOCHS="${RUNPOD_FULL_TRAIN_EPOCHS:-1}"
export RUNPOD_HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
export RUNPOD_HF_DATASET_NAME="${RUNPOD_HF_DATASET_NAME:-elite_2025-11_game}"
export RUNPOD_HF_DATASET_SCHEMA_FILTER="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-game_jsonl_runtime_splice_v1}"
export RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME="${RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME:-1}"
export RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT="${RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT:-8}"
export RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET="${RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET:-1}"
export RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE="${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-128}"
export RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE="${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-2}"
export RUNPOD_GPU_SAMPLE_SECONDS="${RUNPOD_GPU_SAMPLE_SECONDS:-2}"
export RUNPOD_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 3090}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"

echo "[runpod-full-train-easy-smoke] run_id=${RUN_ID}"
echo "[runpod-full-train-easy-smoke] hf_prefix=${RUNPOD_HF_DATASET_PATH_PREFIX}"
echo "[runpod-full-train-easy-smoke] hf_dataset_name=${RUNPOD_HF_DATASET_NAME}"
echo "[runpod-full-train-easy-smoke] hf_schema_filter=${RUNPOD_HF_DATASET_SCHEMA_FILTER}"
echo "[runpod-full-train-easy-smoke] epochs=${RUNPOD_FULL_TRAIN_EPOCHS}"
echo "[runpod-full-train-easy-smoke] gpu=${RUNPOD_GPU_TYPE_ID}"
echo "[runpod-full-train-easy-smoke] runtime_max_samples_per_game=${RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME}"

bash "${REPO_ROOT}/scripts/runpod_full_train_easy.sh"

PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

"${PY_BIN}" "${REPO_ROOT}/scripts/runpod_cycle_verify_full_hf_run.py" \
  --run-id "${RUN_ID}" \
  --output-json "${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/full_hf_verify_after_stop.json"

bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh"

"${PY_BIN}" "${REPO_ROOT}/scripts/runpod_cycle_verify_full_hf_run.py" \
  --run-id "${RUN_ID}" \
  --require-terminated \
  --output-json "${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/full_hf_verify_after_terminate.json"

echo "[runpod-full-train-easy-smoke] completed run_id=${RUN_ID}"
