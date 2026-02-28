#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry/telemetry_common.sh"

STALL_SECONDS="${RUNPOD_TELEMETRY_STALL_SECONDS:-900}"
POLL_SECONDS="${RUNPOD_TELEMETRY_POLL_SECONDS:-30}"
ON_STALL="${RUNPOD_TELEMETRY_ON_STALL:-collect-stop}"
MAX_WATCH_SECONDS="${RUNPOD_TELEMETRY_MAX_WATCH_SECONDS:-0}"

while (($#)); do
  case "$1" in
    --stall-seconds) STALL_SECONDS="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    --on-stall) ON_STALL="${2:-}"; shift 2 ;;
    --max-watch-seconds) MAX_WATCH_SECONDS="${2:-}"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/telemetry_watchdog.sh [--stall-seconds N] [--poll-seconds N] [--on-stall ACTION] [--max-watch-seconds N]"
      exit 0
      ;;
    *) echo "[telemetry-watchdog] unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "${ON_STALL}" in
  none|collect|stop|terminate|collect-stop|collect-terminate) ;;
  *) echo "[telemetry-watchdog] unsupported --on-stall: ${ON_STALL}" >&2; exit 1 ;;
esac

REPO_ROOT="$(telemetry_repo_root)"
RUN_ID="$(telemetry_run_id)"

telemetry_emit_event "watchdog_start" "info" "telemetry watchdog started" "{\"stall_seconds\":${STALL_SECONDS},\"poll_seconds\":${POLL_SECONDS},\"on_stall\":\"${ON_STALL}\"}"
telemetry_emit_checkpoint "watchdog" "running" "telemetry watchdog active"
telemetry_healthchecks_ping start "run_id=${RUN_ID} watchdog_start"

start_ts="$(date +%s)"
last_activity_ts="${start_ts}"
prev_progress="-1"
prev_logs="-1"
prev_state=""

do_action() {
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

while true; do
  now_ts="$(date +%s)"
  status_json="$(RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_status.sh" --no-write 2>/dev/null || echo '{}')"
  remote_state="$(printf '%s\n' "${status_json}" | jq -r '.remote.remote_state // "unknown"')"
  progress_lines="$(printf '%s\n' "${status_json}" | jq -r '.remote.progress_lines // 0')"
  train_log_lines="$(printf '%s\n' "${status_json}" | jq -r '.remote.train_log_lines // 0')"
  train_exit_code="$(printf '%s\n' "${status_json}" | jq -r '.remote.train_exit_code // empty')"

  activity=0
  [[ "${remote_state}" != "${prev_state}" ]] && activity=1
  [[ "${progress_lines}" =~ ^[0-9]+$ && "${progress_lines}" != "${prev_progress}" ]] && activity=1
  [[ "${train_log_lines}" =~ ^[0-9]+$ && "${train_log_lines}" != "${prev_logs}" ]] && activity=1
  if [[ "${activity}" == "1" ]]; then
    last_activity_ts="${now_ts}"
  fi

  idle_for=$((now_ts - last_activity_ts))
  watched_for=$((now_ts - start_ts))

  telemetry_emit_event "watchdog_poll" "info" "watchdog poll" "{\"remote_state\":\"${remote_state}\",\"progress_lines\":${progress_lines},\"train_log_lines\":${train_log_lines},\"idle_for\":${idle_for}}"

  if [[ "${remote_state}" == "training_finished" || "${remote_state}" == "manual_training_finished" ]]; then
    telemetry_emit_checkpoint "training" "done" "watchdog observed training completion"
    telemetry_emit_event "watchdog_finish" "ok" "training finished" "{\"exit_code\":${train_exit_code:-0}}"
    telemetry_healthchecks_ping success "run_id=${RUN_ID} training_finished exit_code=${train_exit_code:-0}"
    exit 0
  fi

  if [[ "${remote_state}" == "training_running" || "${remote_state}" == "manual_training_or_artifacts_present" ]]; then
    if (( idle_for >= STALL_SECONDS )); then
      telemetry_emit_event "watchdog_stall" "error" "stall detected" "{\"idle_for\":${idle_for},\"on_stall\":\"${ON_STALL}\"}"
      telemetry_emit_checkpoint "watchdog" "error" "stall detected; action=${ON_STALL}"
      telemetry_healthchecks_ping fail "run_id=${RUN_ID} stall_detected idle_for=${idle_for}"
      do_action "${ON_STALL}"
      exit 2
    fi
  fi

  if (( MAX_WATCH_SECONDS > 0 && watched_for >= MAX_WATCH_SECONDS )); then
    telemetry_emit_event "watchdog_max_watch_reached" "warn" "max watch window reached" "{\"max_watch_seconds\":${MAX_WATCH_SECONDS}}"
    telemetry_emit_checkpoint "watchdog" "done" "max watch window reached"
    telemetry_healthchecks_ping log "run_id=${RUN_ID} watchdog_max_watch_reached"
    exit 0
  fi

  prev_state="${remote_state}"
  prev_progress="${progress_lines}"
  prev_logs="${train_log_lines}"
  sleep "${POLL_SECONDS}"
done
