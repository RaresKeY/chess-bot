#!/usr/bin/env bash
set -Eeuo pipefail

cloud_check_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

cloud_check_py_bin() {
  local repo_root="$1"
  local py_bin="${repo_root}/.venv/bin/python"
  if [[ -x "${py_bin}" ]]; then
    printf '%s\n' "${py_bin}"
    return 0
  fi
  printf '%s\n' "python3"
}

cloud_check_timeout_step() {
  local timeout_seconds="$1"
  local name="$2"
  shift 2
  echo
  echo "[cloud-check] >>> ${name}"
  if ! timeout "${timeout_seconds}s" "$@"; then
    local rc=$?
    if [[ "${rc}" == "124" ]]; then
      echo "[cloud-check] step timed out: ${name} (${timeout_seconds}s)" >&2
    fi
    return "${rc}"
  fi
}

cloud_check_emit_event() {
  local run_id="$1"
  local event_name="$2"
  local status="$3"
  local message="${4:-}"
  local extra_json="${5:-}"
  if [[ -z "${extra_json}" ]]; then
    extra_json='{}'
  fi
  local repo_root="$6"
  RUNPOD_CYCLE_RUN_ID="${run_id}" bash "${repo_root}/scripts/telemetry_control.sh" event \
    --event "${event_name}" \
    --status "${status}" \
    --message "${message}" \
    --extra-json "${extra_json}" >/dev/null 2>&1 || true
}

cloud_check_emit_checkpoint() {
  local run_id="$1"
  local name="$2"
  local state="$3"
  local note="${4:-}"
  local repo_root="$5"
  RUNPOD_CYCLE_RUN_ID="${run_id}" bash "${repo_root}/scripts/telemetry_control.sh" checkpoint \
    --name "${name}" \
    --state "${state}" \
    --note "${note}" >/dev/null 2>&1 || true
}
