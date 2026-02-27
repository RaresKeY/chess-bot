#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/chess-bot}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
RUN_ID="${RUN_ID:-vast-train-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${TRAIN_OUT_DIR:-${REPO_DIR}/artifacts/vast_cycles/${RUN_ID}}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -U pip >/dev/null
"${VENV_DIR}/bin/python" -m pip install -r "${REPO_DIR}/requirements.txt" >/dev/null

TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-${REPO_DIR}/data/dataset/_smoke_baseline_game}"
TRAIN_PATH="${TRAIN_PATH:-${TRAIN_DATASET_DIR}/train.jsonl}"
VAL_PATH="${VAL_PATH:-${TRAIN_DATASET_DIR}/val.jsonl}"
METRICS_OUT="${TRAIN_METRICS_OUT:-${OUT_DIR}/metrics_${RUN_ID}.json}"
MODEL_OUT="${TRAIN_MODEL_OUT:-${OUT_DIR}/model_${RUN_ID}.pt}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:---epochs 1 --batch-size 64 --num-workers 2}"

cmd=(
  "${VENV_DIR}/bin/python" "${REPO_DIR}/scripts/train_baseline.py"
  --train "${TRAIN_PATH}"
  --val "${VAL_PATH}"
  --out "${MODEL_OUT}"
  --metrics-out "${METRICS_OUT}"
)

if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${TRAIN_EXTRA_ARGS} )
  cmd+=( "${extra[@]}" )
fi

printf '[vast-train-preset] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "[vast-train-preset] model=${MODEL_OUT}"
echo "[vast-train-preset] metrics=${METRICS_OUT}"
