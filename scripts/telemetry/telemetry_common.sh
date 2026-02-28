#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runpod_cycle_common.sh"

telemetry_repo_root() {
  runpod_cycle_repo_root
}

telemetry_run_id() {
  runpod_cycle_run_id
}

telemetry_dir() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${repo_root}/artifacts/runpod_cycles/${run_id}/telemetry"
}

telemetry_now_epoch_ms() {
  date +%s%3N
}

telemetry_emit_event() {
  local event_name="$1"
  local status="$2"
  local message="${3:-}"
  local extra_json="${4:-}"
  if [[ -z "${extra_json}" ]]; then
    extra_json='{}'
  fi
  local repo_root run_id out_dir out_file
  repo_root="$(telemetry_repo_root)"
  run_id="$(telemetry_run_id)"
  out_dir="$(telemetry_dir "${repo_root}" "${run_id}")"
  out_file="${out_dir}/events.jsonl"
  mkdir -p "${out_dir}"
  jq -cn \
    --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson ts_epoch_ms "$(telemetry_now_epoch_ms)" \
    --arg run_id "${run_id}" \
    --arg event "${event_name}" \
    --arg status "${status}" \
    --arg message "${message}" \
    --argjson extra "${extra_json}" \
    '{ts_utc:$ts_utc,ts_epoch_ms:$ts_epoch_ms,run_id:$run_id,event:$event,status:$status,message:$message,extra:$extra}' \
    >> "${out_file}"
}

telemetry_emit_checkpoint() {
  local checkpoint="$1"
  local state="$2"
  local note="${3:-}"
  local repo_root run_id out_dir out_file
  repo_root="$(telemetry_repo_root)"
  run_id="$(telemetry_run_id)"
  out_dir="$(telemetry_dir "${repo_root}" "${run_id}")"
  out_file="${out_dir}/checkpoints.jsonl"
  mkdir -p "${out_dir}"
  jq -cn \
    --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson ts_epoch_ms "$(telemetry_now_epoch_ms)" \
    --arg run_id "${run_id}" \
    --arg checkpoint "${checkpoint}" \
    --arg state "${state}" \
    --arg note "${note}" \
    '{ts_utc:$ts_utc,ts_epoch_ms:$ts_epoch_ms,run_id:$run_id,checkpoint:$checkpoint,state:$state,note:$note}' \
    >> "${out_file}"
}

telemetry_healthchecks_ping() {
  local ping_kind="$1"
  local msg="${2:-}"
  local base_url="${RUNPOD_HEALTHCHECKS_URL:-${HEALTHCHECKS_URL:-}}"
  [[ -n "${base_url}" ]] || return 0
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi
  local url="${base_url}"
  case "${ping_kind}" in
    start) url="${base_url%/}/start" ;;
    success) url="${base_url%/}" ;;
    fail) url="${base_url%/}/fail" ;;
    log) url="${base_url%/}/log" ;;
    *) url="${base_url%/}" ;;
  esac
  curl -fsS -m 10 -X POST --data-raw "${msg}" "${url}" >/dev/null 2>&1 || true
}
