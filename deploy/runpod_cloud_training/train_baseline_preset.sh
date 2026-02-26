#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/chess-bot}"
VENV_DIR="${VENV_DIR:-/opt/venvs/chessbot}"
PY_BIN="${VENV_DIR}/bin/python"
TRAIN_SCRIPT="${REPO_DIR}/scripts/train_baseline.py"

TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-}"
TRAIN_PATH="${TRAIN_PATH:-}"
VAL_PATH="${VAL_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
METRICS_OUT="${METRICS_OUT:-}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-6}"
TRAIN_PHASE_WEIGHT_ENDGAME="${TRAIN_PHASE_WEIGHT_ENDGAME:-2.0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
RUNPOD_PHASE_TIMING_ENABLED="${RUNPOD_PHASE_TIMING_ENABLED:-1}"
RUNPOD_PHASE_TIMING_LOG="${RUNPOD_PHASE_TIMING_LOG:-${REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl}"
RUNPOD_PHASE_TIMING_RUN_ID="${RUNPOD_PHASE_TIMING_RUN_ID:-runpod-train-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUNPOD_PHASE_TIMING_SOURCE="${RUNPOD_PHASE_TIMING_SOURCE:-runpod_train_preset}"

now_epoch_ms() {
  date +%s%3N
}

log_phase_timing() {
  local phase="$1"
  local status="$2"
  local elapsed_ms="$3"
  local extra="${4:-}"
  if [[ "${RUNPOD_PHASE_TIMING_ENABLED}" != "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${RUNPOD_PHASE_TIMING_LOG}")" 2>/dev/null || true
  printf '{"ts_epoch_ms":%s,"source":"%s","run_id":"%s","phase":"%s","status":"%s","elapsed_ms":%s%s}\n' \
    "$(now_epoch_ms)" "${RUNPOD_PHASE_TIMING_SOURCE}" "${RUNPOD_PHASE_TIMING_RUN_ID}" "${phase}" "${status}" "${elapsed_ms}" "${extra}" \
    >> "${RUNPOD_PHASE_TIMING_LOG}" 2>/dev/null || true
}

find_latest_dataset_dir() {
  local base="${REPO_DIR}/data/dataset"
  if [[ ! -d "${base}" ]]; then
    return 1
  fi
  local best=""
  while IFS= read -r dir; do
    if [[ -f "${dir}/train.jsonl" && -f "${dir}/val.jsonl" ]]; then
      best="${dir}"
    fi
  done < <(find "${base}" -maxdepth 1 -mindepth 1 -type d | sort)
  [[ -n "${best}" ]] || return 1
  printf '%s\n' "${best}"
}

if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

t_detect0="$(now_epoch_ms)"
if [[ -z "${TRAIN_DATASET_DIR}" && ( -z "${TRAIN_PATH}" || -z "${VAL_PATH}" ) ]]; then
  TRAIN_DATASET_DIR="$(find_latest_dataset_dir || true)"
fi

if [[ -z "${TRAIN_PATH}" && -n "${TRAIN_DATASET_DIR}" ]]; then
  TRAIN_PATH="${TRAIN_DATASET_DIR}/train.jsonl"
fi
if [[ -z "${VAL_PATH}" && -n "${TRAIN_DATASET_DIR}" ]]; then
  VAL_PATH="${TRAIN_DATASET_DIR}/val.jsonl"
fi

if [[ -z "${TRAIN_PATH}" || -z "${VAL_PATH}" ]]; then
  t_detect1="$(now_epoch_ms)"
  log_phase_timing "resolve_dataset_paths" "error" "$((t_detect1 - t_detect0))"
  echo "[runpod-train] Missing TRAIN_PATH/VAL_PATH and no auto-detected dataset dir with train.jsonl+val.jsonl" >&2
  exit 1
fi
t_detect1="$(now_epoch_ms)"
log_phase_timing "resolve_dataset_paths" "ok" "$((t_detect1 - t_detect0))"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
dataset_name="$(basename "$(dirname "${TRAIN_PATH}")")"
if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${REPO_DIR}/artifacts/${dataset_name}_lstm_phase_side_${ts}.pt"
fi
if [[ -z "${METRICS_OUT}" ]]; then
  METRICS_OUT="${REPO_DIR}/artifacts/${dataset_name}_lstm_phase_side_${ts}.json"
fi

echo "[runpod-train] repo=${REPO_DIR}"
echo "[runpod-train] train=${TRAIN_PATH}"
echo "[runpod-train] val=${VAL_PATH}"
echo "[runpod-train] output=${OUTPUT_PATH}"
echo "[runpod-train] metrics=${METRICS_OUT}"
echo "[runpod-train] preset=current_lstm_phase_side_v1 (embed=256 hidden=512 layers=2 dropout=0.15 epochs=40 lr=2e-4 plateau+early-stop)"

cmd=(
  "${PY_BIN}" "${TRAIN_SCRIPT}"
  --train "${TRAIN_PATH}"
  --val "${VAL_PATH}"
  --output "${OUTPUT_PATH}"
  --metrics-out "${METRICS_OUT}"
  --epochs 40
  --lr 0.0002
  --embed-dim 256
  --hidden-dim 512
  --num-layers 2
  --dropout 0.15
  --batch-size "${TRAIN_BATCH_SIZE}"
  --num-workers "${TRAIN_NUM_WORKERS}"
  --amp
  --phase-feature
  --side-to-move-feature
  --phase-weight-endgame "${TRAIN_PHASE_WEIGHT_ENDGAME}"
  --lr-scheduler plateau
  --lr-scheduler-metric val_loss
  --lr-plateau-factor 0.5
  --lr-plateau-patience 3
  --lr-plateau-threshold 0.0001
  --early-stopping-patience 8
  --early-stopping-metric val_loss
  --early-stopping-min-delta 0.002
)

if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${TRAIN_EXTRA_ARGS} )
  cmd+=( "${extra[@]}" )
fi

printf '[runpod-train] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

t_train0="$(now_epoch_ms)"
if "${cmd[@]}"; then
  t_train1="$(now_epoch_ms)"
  log_phase_timing "train_baseline" "ok" "$((t_train1 - t_train0))"
  exit 0
fi
rc=$?
t_train1="$(now_epoch_ms)"
log_phase_timing "train_baseline" "error" "$((t_train1 - t_train0))" ",\"exit_code\":${rc}"
exit "${rc}"
