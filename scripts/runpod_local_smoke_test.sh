#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-chess-bot-runpod:latest}"
DATASET_SOURCE_DIR="${DATASET_SOURCE_DIR:-${REPO_ROOT}/data/dataset/elite_2025-11_cap4}"
SMOKE_DIR="${SMOKE_DIR:-${REPO_ROOT}/data/dataset/_smoke_runpod}"
SMOKE_TRAIN_ROWS="${SMOKE_TRAIN_ROWS:-256}"
SMOKE_VAL_ROWS="${SMOKE_VAL_ROWS:-64}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-64}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_PROGRESS_JSONL_OUT="${SMOKE_PROGRESS_JSONL_OUT:-}"
SYNC_REQUIREMENTS_ON_START="${SYNC_REQUIREMENTS_ON_START:-1}"
RUNPOD_PHASE_TIMING_LOG="${RUNPOD_PHASE_TIMING_LOG:-${REPO_ROOT}/artifacts/timings/runpod_phase_times.jsonl}"
RUNPOD_PHASE_TIMING_RUN_ID="${RUNPOD_PHASE_TIMING_RUN_ID:-local-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUNPOD_PHASE_TIMING_ENABLED="${RUNPOD_PHASE_TIMING_ENABLED:-1}"

now_epoch_ms() { date +%s%3N; }

log_phase_timing() {
  local phase="$1"
  local status="$2"
  local elapsed_ms="$3"
  local extra="${4:-}"
  if [[ "${RUNPOD_PHASE_TIMING_ENABLED}" != "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${RUNPOD_PHASE_TIMING_LOG}")" 2>/dev/null || true
  printf '{"ts_epoch_ms":%s,"source":"local_runpod_smoke","run_id":"%s","phase":"%s","status":"%s","elapsed_ms":%s%s}\n' \
    "$(now_epoch_ms)" "${RUNPOD_PHASE_TIMING_RUN_ID}" "${phase}" "${status}" "${elapsed_ms}" "${extra}" \
    >> "${RUNPOD_PHASE_TIMING_LOG}"
}

run_timed() {
  local phase="$1"
  shift
  local t0 t1
  t0="$(now_epoch_ms)"
  if "$@"; then
    t1="$(now_epoch_ms)"
    log_phase_timing "${phase}" "ok" "$((t1 - t0))"
    return 0
  fi
  local rc=$?
  t1="$(now_epoch_ms)"
  log_phase_timing "${phase}" "error" "$((t1 - t0))" ",\"exit_code\":${rc}"
  return "${rc}"
}

prep_smoke_dataset() {
  mkdir -p "${SMOKE_DIR}"
  head -n "${SMOKE_TRAIN_ROWS}" "${DATASET_SOURCE_DIR}/train.jsonl" > "${SMOKE_DIR}/train.jsonl"
  head -n "${SMOKE_VAL_ROWS}" "${DATASET_SOURCE_DIR}/val.jsonl" > "${SMOKE_DIR}/val.jsonl"
}

prepare_timing_log_file() {
  mkdir -p "$(dirname "${RUNPOD_PHASE_TIMING_LOG}")"
  touch "${RUNPOD_PHASE_TIMING_LOG}"
  # Local bind mounts + rootless Docker can prevent container-side appends unless the
  # host-created file is world-writable. This only affects a local smoke timing log.
  chmod 666 "${RUNPOD_PHASE_TIMING_LOG}" || true
}

run_container_smoke() {
  docker run --rm --name chess-bot-runpod-smoke \
    -v "${REPO_ROOT}:/workspace/chess-bot" \
    -e CLONE_REPO_ON_START=0 \
    -e GIT_AUTO_PULL=0 \
    -e SYNC_REQUIREMENTS_ON_START="${SYNC_REQUIREMENTS_ON_START}" \
    -e START_SSHD=0 \
    -e START_JUPYTER=0 \
    -e START_INFERENCE_API=0 \
    -e START_HF_WATCH=0 \
    -e START_IDLE_WATCHDOG=0 \
    -e RUNPOD_PHASE_TIMING_ENABLED="${RUNPOD_PHASE_TIMING_ENABLED}" \
    -e RUNPOD_PHASE_TIMING_LOG=/workspace/chess-bot/artifacts/timings/runpod_phase_times.jsonl \
    -e RUNPOD_PHASE_TIMING_RUN_ID="${RUNPOD_PHASE_TIMING_RUN_ID}" \
    -e TRAIN_PATH=/workspace/chess-bot/data/dataset/_smoke_runpod/train.jsonl \
    -e VAL_PATH=/workspace/chess-bot/data/dataset/_smoke_runpod/val.jsonl \
    -e OUTPUT_PATH=/workspace/chess-bot/artifacts/smoke_runpod_model.pt \
    -e METRICS_OUT=/workspace/chess-bot/artifacts/smoke_runpod_metrics.json \
    -e TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
    -e TRAIN_NUM_WORKERS="${SMOKE_NUM_WORKERS}" \
    -e TRAIN_PROGRESS_JSONL_OUT="${SMOKE_PROGRESS_JSONL_OUT}" \
    -e TRAIN_EXTRA_ARGS="--epochs ${SMOKE_EPOCHS} --no-progress" \
    "${IMAGE_NAME}" /bin/bash -lc 'if [[ -f /workspace/chess-bot/deploy/runpod_cloud_training/train_baseline_preset.sh ]]; then bash /workspace/chess-bot/deploy/runpod_cloud_training/train_baseline_preset.sh; else bash /opt/runpod_cloud_training/train_baseline_preset.sh; fi'
}

echo "[local-smoke] image=${IMAGE_NAME}"
echo "[local-smoke] source_dataset=${DATASET_SOURCE_DIR}"
echo "[local-smoke] smoke_dir=${SMOKE_DIR} train_rows=${SMOKE_TRAIN_ROWS} val_rows=${SMOKE_VAL_ROWS}"
if [[ -n "${SMOKE_PROGRESS_JSONL_OUT}" ]]; then
  echo "[local-smoke] progress_jsonl=${SMOKE_PROGRESS_JSONL_OUT}"
fi
echo "[local-smoke] timing_log=${RUNPOD_PHASE_TIMING_LOG}"
echo "[local-smoke] run_id=${RUNPOD_PHASE_TIMING_RUN_ID}"

run_timed "prepare_smoke_dataset" prep_smoke_dataset
run_timed "prepare_timing_log_file" prepare_timing_log_file
run_timed "docker_run_smoke_training" run_container_smoke

echo "[local-smoke] done"
