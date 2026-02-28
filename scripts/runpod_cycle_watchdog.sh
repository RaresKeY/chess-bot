#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/runpod_cycle_watchdog.sh [--stall-seconds N] [--poll-seconds N] [--max-watch-seconds N] [--on-stall ACTION]

Actions on stall:
  none | collect | stop | terminate | collect-stop | collect-terminate

Behavior:
- Polls run status via scripts/runpod_cycle_status.sh --no-write
- Treats progress/log line growth or state transition as activity
- On prolonged inactivity during training, triggers configured action
USAGE
}

STALL_SECONDS="${RUNPOD_WATCHDOG_STALL_SECONDS:-900}"
POLL_SECONDS="${RUNPOD_WATCHDOG_POLL_SECONDS:-30}"
MAX_WATCH_SECONDS="${RUNPOD_WATCHDOG_MAX_WATCH_SECONDS:-0}"
ON_STALL="${RUNPOD_WATCHDOG_ON_STALL:-collect-stop}"
AUTO_COLLECT_ON_FINISH="${RUNPOD_WATCHDOG_AUTO_COLLECT_ON_FINISH:-1}"

while (($#)); do
  case "$1" in
    --stall-seconds) STALL_SECONDS="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    --max-watch-seconds) MAX_WATCH_SECONDS="${2:-}"; shift 2 ;;
    --on-stall) ON_STALL="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[runpod-watchdog] unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "${ON_STALL}" in
  none|collect|stop|terminate|collect-stop|collect-terminate) ;;
  *) echo "[runpod-watchdog] unsupported --on-stall: ${ON_STALL}" >&2; exit 1 ;;
esac

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
runpod_cycle_require_cmd jq

status_cmd=( bash "${REPO_ROOT}/scripts/runpod_cycle_status.sh" --no-write )

start_ts="$(date +%s)"
last_activity_ts="${start_ts}"
prev_state=""
prev_progress_lines="-1"
prev_train_log_lines="-1"

run_action() {
  local action="$1"
  case "${action}" in
    none) return 0 ;;
    collect) RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh" || true ;;
    stop) RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh" || true ;;
    terminate) RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh" || true ;;
    collect-stop)
      RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh" || true
      RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh" || true
      ;;
    collect-terminate)
      RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh" || true
      RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh" || true
      ;;
  esac
}

echo "[runpod-watchdog] run_id=${RUN_ID} stall_seconds=${STALL_SECONDS} poll_seconds=${POLL_SECONDS} max_watch_seconds=${MAX_WATCH_SECONDS} on_stall=${ON_STALL}"

while true; do
  now_ts="$(date +%s)"
  summary="$(RUNPOD_CYCLE_RUN_ID="${RUN_ID}" "${status_cmd[@]}" 2>/dev/null || echo '{"remote":{"remote_state":"status_error"}}')"

  state="$(printf '%s\n' "${summary}" | jq -r '.remote.remote_state // "unknown"')"
  progress_lines="$(printf '%s\n' "${summary}" | jq -r '.remote.progress_lines // 0')"
  train_log_lines="$(printf '%s\n' "${summary}" | jq -r '.remote.train_log_lines // 0')"
  exit_code="$(printf '%s\n' "${summary}" | jq -r '.remote.train_exit_code // empty')"

  activity=0
  if [[ "${state}" != "${prev_state}" ]]; then
    activity=1
  fi
  if [[ "${progress_lines}" =~ ^[0-9]+$ && "${progress_lines}" != "${prev_progress_lines}" ]]; then
    activity=1
  fi
  if [[ "${train_log_lines}" =~ ^[0-9]+$ && "${train_log_lines}" != "${prev_train_log_lines}" ]]; then
    activity=1
  fi

  if [[ "${activity}" == "1" ]]; then
    last_activity_ts="${now_ts}"
  fi

  idle_for=$((now_ts - last_activity_ts))
  watched_for=$((now_ts - start_ts))

  echo "[runpod-watchdog] state=${state} progress_lines=${progress_lines} train_log_lines=${train_log_lines} idle_for=${idle_for}s watched_for=${watched_for}s"

  if [[ "${state}" == "training_finished" || "${state}" == "manual_training_finished" ]]; then
    if [[ "${AUTO_COLLECT_ON_FINISH}" == "1" ]]; then
      RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh" || true
    fi
    echo "[runpod-watchdog] training finished (exit_code=${exit_code:-<unknown>}); exiting"
    exit 0
  fi

  if [[ "${state}" == "training_running" || "${state}" == "manual_training_or_artifacts_present" ]]; then
    if (( idle_for >= STALL_SECONDS )); then
      echo "[runpod-watchdog] stall detected (idle_for=${idle_for}s >= ${STALL_SECONDS}s); action=${ON_STALL}" >&2
      run_action "${ON_STALL}"
      exit 2
    fi
  fi

  if (( MAX_WATCH_SECONDS > 0 && watched_for >= MAX_WATCH_SECONDS )); then
    echo "[runpod-watchdog] max watch time reached; exiting"
    exit 0
  fi

  prev_state="${state}"
  prev_progress_lines="${progress_lines}"
  prev_train_log_lines="${train_log_lines}"

  sleep "${POLL_SECONDS}"
done
