#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${VAST_SMOKE_RUN_ID:-vast-local-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_DIR="${VAST_SMOKE_DATASET_DIR:-${REPO_ROOT}/data/dataset/_smoke_fast_game}"
SMOKE_EPOCHS="${VAST_SMOKE_EPOCHS:-1}"
SMOKE_MAX_TOTAL_ROWS="${VAST_SMOKE_MAX_TOTAL_ROWS:-512}"
VENV_DIR="${VAST_SMOKE_VENV_DIR:-${REPO_ROOT}/.venv}"

TRAIN_PATH="${DATASET_DIR}/train.jsonl"
VAL_PATH="${DATASET_DIR}/val.jsonl"
OUT_DIR="${REPO_ROOT}/artifacts/vast_cycles/${RUN_ID}/local_smoke"
MODEL_OUT="${OUT_DIR}/model_${RUN_ID}.pt"
METRICS_OUT="${OUT_DIR}/metrics_${RUN_ID}.json"
PROGRESS_OUT="${OUT_DIR}/progress_${RUN_ID}.jsonl"
PRESET_SCRIPT="${REPO_ROOT}/deploy/vast_cloud_training/train_baseline_preset.sh"

[[ -f "${PRESET_SCRIPT}" ]] || { echo "[vast-local-smoke] missing preset script: ${PRESET_SCRIPT}" >&2; exit 1; }
[[ -f "${TRAIN_PATH}" ]] || { echo "[vast-local-smoke] missing train dataset: ${TRAIN_PATH}" >&2; exit 1; }
[[ -f "${VAL_PATH}" ]] || { echo "[vast-local-smoke] missing val dataset: ${VAL_PATH}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"

TRAIN_EXTRA_ARGS="--epochs ${SMOKE_EPOCHS} --batch-size 64 --num-workers 0 --max-total-rows ${SMOKE_MAX_TOTAL_ROWS} --no-progress --progress-jsonl-out ${PROGRESS_OUT}"

echo "[vast-local-smoke] repo=${REPO_ROOT}"
echo "[vast-local-smoke] run_id=${RUN_ID}"
echo "[vast-local-smoke] dataset_dir=${DATASET_DIR}"
echo "[vast-local-smoke] train=${TRAIN_PATH}"
echo "[vast-local-smoke] val=${VAL_PATH}"
echo "[vast-local-smoke] out_dir=${OUT_DIR}"
echo "[vast-local-smoke] venv_dir=${VENV_DIR}"
echo "[vast-local-smoke] epochs=${SMOKE_EPOCHS}"
echo "[vast-local-smoke] max_total_rows=${SMOKE_MAX_TOTAL_ROWS}"

REPO_DIR="${REPO_ROOT}" \
VENV_DIR="${VENV_DIR}" \
RUN_ID="${RUN_ID}" \
TRAIN_OUT_DIR="${OUT_DIR}" \
TRAIN_PATH="${TRAIN_PATH}" \
VAL_PATH="${VAL_PATH}" \
TRAIN_MODEL_OUT="${MODEL_OUT}" \
TRAIN_METRICS_OUT="${METRICS_OUT}" \
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS}" \
bash "${PRESET_SCRIPT}"

echo "[vast-local-smoke] model=${MODEL_OUT}"
echo "[vast-local-smoke] metrics=${METRICS_OUT}"
echo "[vast-local-smoke] progress=${PROGRESS_OUT}"
