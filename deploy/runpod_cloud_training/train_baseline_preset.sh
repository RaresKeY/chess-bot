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
HF_FETCH_LATEST_ALL_DATASETS="${HF_FETCH_LATEST_ALL_DATASETS:-0}"
HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID:-${HF_REPO_ID:-}}"
HF_DATASET_PATH_PREFIX="${HF_DATASET_PATH_PREFIX:-validated_datasets}"
HF_DATASET_CACHE_DIR="${HF_DATASET_CACHE_DIR:-${REPO_DIR}/data/hf_datasets}"
HF_DATASET_FETCH_MANIFEST="${HF_DATASET_FETCH_MANIFEST:-${REPO_DIR}/artifacts/hf_dataset_fetch_manifest.json}"
HF_USE_EXISTING_FETCH_MANIFEST="${HF_USE_EXISTING_FETCH_MANIFEST:-0}"
HF_DATASET_SCHEMA_FILTER="${HF_DATASET_SCHEMA_FILTER:-auto}"
TRAIN_RUNTIME_MIN_CONTEXT="${TRAIN_RUNTIME_MIN_CONTEXT:-8}"
TRAIN_RUNTIME_MIN_TARGET="${TRAIN_RUNTIME_MIN_TARGET:-1}"
TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME="${TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME:-0}"
TRAIN_ROLLOUT_HORIZON="${TRAIN_ROLLOUT_HORIZON:-8}"
TRAIN_CLOSENESS_HORIZON="${TRAIN_CLOSENESS_HORIZON:-${TRAIN_ROLLOUT_HORIZON}}"
TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE="${TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE:-0}"
TRAIN_MAX_TRAIN_ROWS="${TRAIN_MAX_TRAIN_ROWS:-0}"
TRAIN_MAX_VAL_ROWS="${TRAIN_MAX_VAL_ROWS:-0}"
TRAIN_MAX_TOTAL_ROWS="${TRAIN_MAX_TOTAL_ROWS:-0}"
TRAIN_BEST_CHECKPOINT_OUT="${TRAIN_BEST_CHECKPOINT_OUT:-}"
TRAIN_EPOCH_CHECKPOINT_DIR="${TRAIN_EPOCH_CHECKPOINT_DIR:-}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-6}"
TRAIN_PHASE_WEIGHT_ENDGAME="${TRAIN_PHASE_WEIGHT_ENDGAME:-2.0}"
TRAIN_PROGRESS_JSONL_OUT="${TRAIN_PROGRESS_JSONL_OUT:-}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-1}"
TRAIN_NCCL_HANG_CHECK_ENABLED="${TRAIN_NCCL_HANG_CHECK_ENABLED:-1}"
TRAIN_NCCL_HANG_TIMEOUT_SECONDS="${TRAIN_NCCL_HANG_TIMEOUT_SECONDS:-900}"
TRAIN_NCCL_HANG_POLL_SECONDS="${TRAIN_NCCL_HANG_POLL_SECONDS:-30}"
TRAIN_NCCL_HANG_EXIT_CODE="${TRAIN_NCCL_HANG_EXIT_CODE:-124}"
TRAIN_NCCL_HANG_LOG_PATH="${TRAIN_NCCL_HANG_LOG_PATH:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
RUNPOD_PHASE_TIMING_ENABLED="${RUNPOD_PHASE_TIMING_ENABLED:-1}"
RUNPOD_PHASE_TIMING_LOG="${RUNPOD_PHASE_TIMING_LOG:-${REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl}"
RUNPOD_PHASE_TIMING_RUN_ID="${RUNPOD_PHASE_TIMING_RUN_ID:-runpod-train-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUNPOD_PHASE_TIMING_SOURCE="${RUNPOD_PHASE_TIMING_SOURCE:-runpod_train_preset}"
HF_ALL_TRAIN_PATHS=()
HF_ALL_VAL_PATHS=()
DETECTED_DATASET_SCHEMA="${DETECTED_DATASET_SCHEMA:-}"
DETECTED_DATASET_FORMAT="${DETECTED_DATASET_FORMAT:-}"

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

file_mtime_epoch() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    printf '%s\n' "0"
    return 0
  fi
  stat -c %Y "${p}" 2>/dev/null || printf '%s\n' "0"
}

write_nccl_hang_diagnostics() {
  local log_path="$1"
  local train_pid="$2"
  local progress_path="$3"
  local idle_seconds="$4"
  local timeout_seconds="$5"
  mkdir -p "$(dirname "${log_path}")" 2>/dev/null || true
  {
    echo "=== NCCL Hang Watchdog Diagnostic ==="
    echo "ts_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_dir=${REPO_DIR}"
    echo "train_pid=${train_pid}"
    echo "train_nproc_per_node=${TRAIN_NPROC_PER_NODE}"
    echo "progress_path=${progress_path}"
    echo "idle_seconds=${idle_seconds}"
    echo "timeout_seconds=${timeout_seconds}"
    echo "train_output=${OUTPUT_PATH}"
    echo "train_metrics=${METRICS_OUT}"
    echo "nccl_env: NCCL_DEBUG=${NCCL_DEBUG:-} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-}"
    echo "torch_nccl_env: TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-} TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-} TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-} TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-} TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-} TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-}"
    if [[ -f "${progress_path}" ]]; then
      echo "progress_exists=1 progress_lines=$(wc -l < "${progress_path}" 2>/dev/null || echo 0) progress_mtime=$(file_mtime_epoch "${progress_path}")"
      echo "--- progress_tail ---"
      tail -n 50 "${progress_path}" || true
    else
      echo "progress_exists=0"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "--- nvidia_smi_L ---"
      nvidia-smi -L || true
      echo "--- nvidia_smi_summary ---"
      nvidia-smi || true
      echo "--- nvidia_smi_pmon ---"
      nvidia-smi pmon -c 1 || true
    else
      echo "nvidia-smi not found"
    fi
    echo "--- process_snapshot ---"
    ps -eo pid,ppid,pgid,stat,etimes,%cpu,%mem,cmd | head -n 200 || true
    if command -v pgrep >/dev/null 2>&1; then
      echo "--- process_match (torchrun|train_baseline.py|python) ---"
      pgrep -af "torchrun|train_baseline.py|python" | head -n 200 || true
    fi
  } >> "${log_path}" 2>&1 || true
}

run_nccl_hang_watchdog() {
  local train_pid="$1"
  local progress_path="$2"
  local timeout_seconds="$3"
  local poll_seconds="$4"
  local log_path="$5"
  local marker_path="$6"
  local now_seconds progress_mtime last_progress_seconds idle_seconds
  now_seconds="$(date +%s)"
  progress_mtime="$(file_mtime_epoch "${progress_path}")"
  if [[ "${progress_mtime}" =~ ^[0-9]+$ ]] && (( progress_mtime > 0 )); then
    last_progress_seconds="${progress_mtime}"
  else
    last_progress_seconds="${now_seconds}"
  fi
  while kill -0 "${train_pid}" >/dev/null 2>&1; do
    sleep "${poll_seconds}" || true
    now_seconds="$(date +%s)"
    progress_mtime="$(file_mtime_epoch "${progress_path}")"
    if [[ "${progress_mtime}" =~ ^[0-9]+$ ]] && (( progress_mtime > last_progress_seconds )); then
      last_progress_seconds="${progress_mtime}"
    fi
    idle_seconds=$(( now_seconds - last_progress_seconds ))
    if (( idle_seconds >= timeout_seconds )); then
      echo "[runpod-train] nccl_hang_watchdog detected stalled multi-gpu progress (idle=${idle_seconds}s timeout=${timeout_seconds}s); collecting diagnostics: ${log_path}" >&2
      write_nccl_hang_diagnostics "${log_path}" "${train_pid}" "${progress_path}" "${idle_seconds}" "${timeout_seconds}"
      : > "${marker_path}"
      kill -TERM "${train_pid}" >/dev/null 2>&1 || true
      pkill -TERM -P "${train_pid}" >/dev/null 2>&1 || true
      sleep 8 || true
      kill -KILL "${train_pid}" >/dev/null 2>&1 || true
      pkill -KILL -P "${train_pid}" >/dev/null 2>&1 || true
      return 0
    fi
  done
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

detect_dataset_schema() {
  local train_path="$1"
  local dataset_dir="${2:-}"
  local schema=""
  local fmt=""
  if [[ -n "${dataset_dir}" && -f "${dataset_dir}/stats.json" ]]; then
    fmt="$("${PY_BIN}" - "${dataset_dir}/stats.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
print(str(d.get("dataset_format", "")))
PY
)"
  fi
  schema="$("${PY_BIN}" - "${train_path}" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "context" in row:
                print("spliced")
            elif "moves" in row or "moves_uci" in row:
                print("game")
            else:
                print("unknown")
            break
        else:
            print("empty")
except Exception:
    print("unknown")
PY
)"
  DETECTED_DATASET_SCHEMA="${schema}"
  DETECTED_DATASET_FORMAT="${fmt}"
}

if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

fetch_latest_hf_datasets() {
  if [[ -z "${HF_DATASET_REPO_ID}" ]]; then
    echo "[runpod-train] HF_FETCH_LATEST_ALL_DATASETS=1 but HF_DATASET_REPO_ID/HF_REPO_ID is not set" >&2
    return 1
  fi
  mkdir -p "${HF_DATASET_CACHE_DIR}" "$(dirname "${HF_DATASET_FETCH_MANIFEST}")"
  echo "[runpod-train] fetching latest datasets from HF repo=${HF_DATASET_REPO_ID} prefix=${HF_DATASET_PATH_PREFIX}"
  "${PY_BIN}" "${REPO_DIR}/scripts/hf_dataset_fetch.py" \
    --repo-id "${HF_DATASET_REPO_ID}" \
    --repo-path-prefix "${HF_DATASET_PATH_PREFIX}" \
    --dest-dir "${HF_DATASET_CACHE_DIR}" \
    --all-latest \
    --output-manifest "${HF_DATASET_FETCH_MANIFEST}"
}

load_hf_manifest_paths() {
  [[ -f "${HF_DATASET_FETCH_MANIFEST}" ]] || return 1
  mapfile -t _hf_train_paths < <("${PY_BIN}" - "${HF_DATASET_FETCH_MANIFEST}" "${HF_DATASET_SCHEMA_FILTER}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
schema_filter = (sys.argv[2] or "auto").strip()
agg = data.get("aggregate", {})
agg_by_format = data.get("aggregate_by_format", {}) or {}
def chosen():
    if schema_filter and schema_filter not in {"", "auto"}:
        return schema_filter
    # prefer compact game dataset if present, else spliced, else fallback to aggregate
    for cand in ("game_jsonl_runtime_splice_v1", "splice_rows_legacy"):
        if cand in agg_by_format:
            return cand
    return ""
fmt = chosen()
paths = agg_by_format.get(fmt, {}).get("train_paths", []) if fmt else agg.get("train_paths", [])
for p in paths:
    if p:
        print(p)
PY
)
  mapfile -t _hf_val_paths < <("${PY_BIN}" - "${HF_DATASET_FETCH_MANIFEST}" "${HF_DATASET_SCHEMA_FILTER}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
schema_filter = (sys.argv[2] or "auto").strip()
agg = data.get("aggregate", {})
agg_by_format = data.get("aggregate_by_format", {}) or {}
def chosen():
    if schema_filter and schema_filter not in {"", "auto"}:
        return schema_filter
    for cand in ("game_jsonl_runtime_splice_v1", "splice_rows_legacy"):
        if cand in agg_by_format:
            return cand
    return ""
fmt = chosen()
paths = agg_by_format.get(fmt, {}).get("val_paths", []) if fmt else agg.get("val_paths", [])
for p in paths:
    if p:
        print(p)
PY
)
  export HF_SELECTED_SCHEMA="$("${PY_BIN}" - "${HF_DATASET_FETCH_MANIFEST}" "${HF_DATASET_SCHEMA_FILTER}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
schema_filter = (sys.argv[2] or "auto").strip()
agg_by_format = data.get("aggregate_by_format", {}) or {}
if schema_filter and schema_filter not in {"", "auto"}:
    print(schema_filter)
elif "game_jsonl_runtime_splice_v1" in agg_by_format:
    print("game_jsonl_runtime_splice_v1")
elif "splice_rows_legacy" in agg_by_format:
    print("splice_rows_legacy")
else:
    print("")
PY
)"
  if (( ${#_hf_train_paths[@]} > 0 )); then
    TRAIN_PATH="${_hf_train_paths[0]}"
  fi
  if (( ${#_hf_val_paths[@]} > 0 )); then
    VAL_PATH="${_hf_val_paths[0]}"
  fi
  HF_ALL_TRAIN_PATHS=("${_hf_train_paths[@]}")
  HF_ALL_VAL_PATHS=("${_hf_val_paths[@]}")
}

t_detect0="$(now_epoch_ms)"
if [[ "${HF_FETCH_LATEST_ALL_DATASETS}" == "1" && -z "${TRAIN_DATASET_DIR}" && -z "${TRAIN_PATH}" && -z "${VAL_PATH}" ]]; then
  if [[ "${HF_USE_EXISTING_FETCH_MANIFEST}" == "1" && -f "${HF_DATASET_FETCH_MANIFEST}" ]]; then
    echo "[runpod-train] reusing existing HF fetch manifest: ${HF_DATASET_FETCH_MANIFEST}"
  elif ! fetch_latest_hf_datasets; then
    t_detect1="$(now_epoch_ms)"
    log_phase_timing "resolve_dataset_paths" "error" "$((t_detect1 - t_detect0))"
    exit 1
  fi
  load_hf_manifest_paths || true
fi

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
if [[ "${HF_FETCH_LATEST_ALL_DATASETS}" == "1" && ${#HF_ALL_TRAIN_PATHS[@]} -gt 0 ]]; then
  dataset_name="hf_latest_all_${#HF_ALL_TRAIN_PATHS[@]}dsets"
fi
if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${REPO_DIR}/artifacts/${dataset_name}_lstm_phase_side_${ts}.pt"
fi
if [[ -z "${METRICS_OUT}" ]]; then
  METRICS_OUT="${REPO_DIR}/artifacts/${dataset_name}_lstm_phase_side_${ts}.json"
fi
if ! [[ "${TRAIN_NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( TRAIN_NPROC_PER_NODE < 1 )); then
  TRAIN_NPROC_PER_NODE=1
fi
if ! [[ "${TRAIN_NCCL_HANG_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || (( TRAIN_NCCL_HANG_TIMEOUT_SECONDS < 1 )); then
  TRAIN_NCCL_HANG_TIMEOUT_SECONDS=900
fi
if ! [[ "${TRAIN_NCCL_HANG_POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( TRAIN_NCCL_HANG_POLL_SECONDS < 1 )); then
  TRAIN_NCCL_HANG_POLL_SECONDS=30
fi
if ! [[ "${TRAIN_NCCL_HANG_EXIT_CODE}" =~ ^[0-9]+$ ]] || (( TRAIN_NCCL_HANG_EXIT_CODE < 1 || TRAIN_NCCL_HANG_EXIT_CODE > 255 )); then
  TRAIN_NCCL_HANG_EXIT_CODE=124
fi
if (( TRAIN_NCCL_HANG_POLL_SECONDS > TRAIN_NCCL_HANG_TIMEOUT_SECONDS )); then
  TRAIN_NCCL_HANG_POLL_SECONDS="${TRAIN_NCCL_HANG_TIMEOUT_SECONDS}"
fi
if (( TRAIN_NPROC_PER_NODE > 1 )) && [[ "${TRAIN_NCCL_HANG_CHECK_ENABLED}" == "1" ]] && [[ -z "${TRAIN_PROGRESS_JSONL_OUT}" ]]; then
  TRAIN_PROGRESS_JSONL_OUT="$(dirname "${METRICS_OUT}")/train_progress_${ts}.jsonl"
fi
if [[ -z "${TRAIN_NCCL_HANG_LOG_PATH}" ]]; then
  TRAIN_NCCL_HANG_LOG_PATH="$(dirname "${METRICS_OUT}")/nccl_hang_watchdog_${ts}.log"
fi

echo "[runpod-train] repo=${REPO_DIR}"
echo "[runpod-train] train=${TRAIN_PATH}"
echo "[runpod-train] val=${VAL_PATH}"
if [[ -f "${HF_DATASET_FETCH_MANIFEST}" ]]; then
  echo "[runpod-train] hf_dataset_fetch_manifest=${HF_DATASET_FETCH_MANIFEST}"
  echo "[runpod-train] hf_dataset_schema_filter=${HF_DATASET_SCHEMA_FILTER}"
  echo "[runpod-train] hf_selected_schema=${HF_SELECTED_SCHEMA:-}"
fi
echo "[runpod-train] output=${OUTPUT_PATH}"
echo "[runpod-train] metrics=${METRICS_OUT}"
echo "[runpod-train] rollout_horizon=${TRAIN_ROLLOUT_HORIZON} closeness_horizon=${TRAIN_CLOSENESS_HORIZON}"
echo "[runpod-train] subset_caps max_total_rows=${TRAIN_MAX_TOTAL_ROWS} max_train_rows=${TRAIN_MAX_TRAIN_ROWS} max_val_rows=${TRAIN_MAX_VAL_ROWS}"
if [[ -n "${TRAIN_BEST_CHECKPOINT_OUT}" ]]; then
  echo "[runpod-train] best_checkpoint_out=${TRAIN_BEST_CHECKPOINT_OUT}"
fi
if [[ -n "${TRAIN_EPOCH_CHECKPOINT_DIR}" ]]; then
  echo "[runpod-train] epoch_checkpoint_dir=${TRAIN_EPOCH_CHECKPOINT_DIR}"
fi
if [[ -n "${TRAIN_PROGRESS_JSONL_OUT}" ]]; then
  echo "[runpod-train] progress_jsonl=${TRAIN_PROGRESS_JSONL_OUT}"
fi
echo "[runpod-train] train_nproc_per_node=${TRAIN_NPROC_PER_NODE}"
if (( TRAIN_NPROC_PER_NODE > 1 )); then
  echo "[runpod-train] nccl_hang_watchdog enabled=${TRAIN_NCCL_HANG_CHECK_ENABLED} timeout_seconds=${TRAIN_NCCL_HANG_TIMEOUT_SECONDS} poll_seconds=${TRAIN_NCCL_HANG_POLL_SECONDS} exit_code=${TRAIN_NCCL_HANG_EXIT_CODE} log_path=${TRAIN_NCCL_HANG_LOG_PATH}"
fi
echo "[runpod-train] preset=current_lstm_phase_side_v1 (embed=256 hidden=512 layers=2 dropout=0.15 epochs=40 lr=2e-4 plateau+early-stop)"
detect_dataset_schema "${TRAIN_PATH}" "${TRAIN_DATASET_DIR:-$(dirname "${TRAIN_PATH}")}"
echo "[runpod-train] detected_dataset_schema=${DETECTED_DATASET_SCHEMA}"
if [[ -n "${DETECTED_DATASET_FORMAT}" ]]; then
  echo "[runpod-train] detected_dataset_format=${DETECTED_DATASET_FORMAT}"
fi

train_args=(
  "${TRAIN_SCRIPT}"
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
  --rollout-horizon "${TRAIN_ROLLOUT_HORIZON}"
  --closeness-horizon "${TRAIN_CLOSENESS_HORIZON}"
  --lr-scheduler plateau
  --lr-scheduler-metric val_loss
  --lr-plateau-factor 0.5
  --lr-plateau-patience 3
  --lr-plateau-threshold 0.0001
  --early-stopping-patience 8
  --early-stopping-metric val_loss
  --early-stopping-min-delta 0.002
)

if [[ "${DETECTED_DATASET_SCHEMA}" == "game" ]]; then
  train_args+=(
    --runtime-min-context "${TRAIN_RUNTIME_MIN_CONTEXT}"
    --runtime-min-target "${TRAIN_RUNTIME_MIN_TARGET}"
    --runtime-max-samples-per-game "${TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME}"
  )
fi

if [[ "${TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE}" == "1" ]]; then
  train_args+=( --require-runtime-splice-cache )
fi

if [[ "${TRAIN_MAX_TRAIN_ROWS}" != "0" ]]; then
  train_args+=( --max-train-rows "${TRAIN_MAX_TRAIN_ROWS}" )
fi
if [[ "${TRAIN_MAX_VAL_ROWS}" != "0" ]]; then
  train_args+=( --max-val-rows "${TRAIN_MAX_VAL_ROWS}" )
fi
if [[ "${TRAIN_MAX_TOTAL_ROWS}" != "0" ]]; then
  train_args+=( --max-total-rows "${TRAIN_MAX_TOTAL_ROWS}" )
fi
if [[ -n "${TRAIN_BEST_CHECKPOINT_OUT}" ]]; then
  train_args+=( --best-checkpoint-out "${TRAIN_BEST_CHECKPOINT_OUT}" )
fi
if [[ -n "${TRAIN_EPOCH_CHECKPOINT_DIR}" ]]; then
  train_args+=( --epoch-checkpoint-dir "${TRAIN_EPOCH_CHECKPOINT_DIR}" )
fi

if [[ -n "${TRAIN_PROGRESS_JSONL_OUT}" ]]; then
  train_args+=( --progress-jsonl-out "${TRAIN_PROGRESS_JSONL_OUT}" )
fi

if [[ "${HF_FETCH_LATEST_ALL_DATASETS}" == "1" && ${#HF_ALL_TRAIN_PATHS[@]} -gt 0 && ${#HF_ALL_VAL_PATHS[@]} -gt 0 ]]; then
  for p in "${HF_ALL_TRAIN_PATHS[@]}"; do
    train_args+=( --train "${p}" )
  done
  for p in "${HF_ALL_VAL_PATHS[@]}"; do
    train_args+=( --val "${p}" )
  done
  echo "[runpod-train] using_hf_latest_all_datasets=1 train_files=${#HF_ALL_TRAIN_PATHS[@]} val_files=${#HF_ALL_VAL_PATHS[@]}"
  if [[ -n "${HF_SELECTED_SCHEMA:-}" ]]; then
    echo "[runpod-train] using_hf_schema=${HF_SELECTED_SCHEMA}"
  fi
else
  train_args+=( --train "${TRAIN_PATH}" --val "${VAL_PATH}" )
fi

if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${TRAIN_EXTRA_ARGS} )
  train_args+=( "${extra[@]}" )
fi

cmd=()
if (( TRAIN_NPROC_PER_NODE > 1 )); then
  cmd=( "${VENV_DIR}/bin/torchrun" --standalone --nnodes=1 --nproc-per-node "${TRAIN_NPROC_PER_NODE}" "${train_args[@]}" )
else
  cmd=( "${PY_BIN}" "${train_args[@]}" )
fi

printf '[runpod-train] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

t_train0="$(now_epoch_ms)"
if (( TRAIN_NPROC_PER_NODE > 1 )) && [[ "${TRAIN_NCCL_HANG_CHECK_ENABLED}" == "1" ]]; then
  NCCL_HANG_TRIGGER_FILE="${TRAIN_NCCL_HANG_LOG_PATH}.triggered"
  rm -f "${NCCL_HANG_TRIGGER_FILE}" >/dev/null 2>&1 || true
  "${cmd[@]}" &
  TRAIN_CMD_PID=$!
  run_nccl_hang_watchdog \
    "${TRAIN_CMD_PID}" \
    "${TRAIN_PROGRESS_JSONL_OUT}" \
    "${TRAIN_NCCL_HANG_TIMEOUT_SECONDS}" \
    "${TRAIN_NCCL_HANG_POLL_SECONDS}" \
    "${TRAIN_NCCL_HANG_LOG_PATH}" \
    "${NCCL_HANG_TRIGGER_FILE}" &
  NCCL_WATCHDOG_PID=$!
  set +e
  wait "${TRAIN_CMD_PID}"
  rc=$?
  set -e
  if kill -0 "${NCCL_WATCHDOG_PID}" >/dev/null 2>&1; then
    kill "${NCCL_WATCHDOG_PID}" >/dev/null 2>&1 || true
  fi
  wait "${NCCL_WATCHDOG_PID}" >/dev/null 2>&1 || true
  if [[ -f "${NCCL_HANG_TRIGGER_FILE}" ]]; then
    echo "[runpod-train] nccl_hang_watchdog_triggered=1 diagnostics=${TRAIN_NCCL_HANG_LOG_PATH}" >&2
    rc="${TRAIN_NCCL_HANG_EXIT_CODE}"
  fi
  rm -f "${NCCL_HANG_TRIGGER_FILE}" >/dev/null 2>&1 || true
elif "${cmd[@]}"; then
  t_train1="$(now_epoch_ms)"
  log_phase_timing "train_baseline" "ok" "$((t_train1 - t_train0))"
  exit 0
else
  rc=$?
fi
t_train1="$(now_epoch_ms)"
if [[ "${rc}" == "0" ]]; then
  log_phase_timing "train_baseline" "ok" "$((t_train1 - t_train0))"
  exit 0
fi
log_phase_timing "train_baseline" "error" "$((t_train1 - t_train0))" ",\"exit_code\":${rc}"
exit "${rc}"
