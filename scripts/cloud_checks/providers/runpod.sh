#!/usr/bin/env bash
set -Eeuo pipefail

runpod_provider_local_checks() {
  local repo_root="$1"
  local py_bin="$2"
  local timeout_seconds="$3"

  cloud_check_timeout_step "${timeout_seconds}" "runpod script contract check" \
    test -x "${repo_root}/scripts/runpod_provision.py"
  cloud_check_timeout_step "${timeout_seconds}" "runpod telemetry status json" \
    bash "${repo_root}/scripts/telemetry_control.sh" status --json
  cloud_check_timeout_step "${timeout_seconds}" "runpod telemetry event+checkpoint" \
    /bin/bash -lc "RUNPOD_CYCLE_RUN_ID='${CLOUD_CHECK_RUN_ID}' bash '${repo_root}/scripts/telemetry_control.sh' event --event runpod_connectivity_local --status info --message local_ok && RUNPOD_CYCLE_RUN_ID='${CLOUD_CHECK_RUN_ID}' bash '${repo_root}/scripts/telemetry_control.sh' checkpoint --name runpod_connectivity_local --state running --note local_ok"
  cloud_check_timeout_step "${timeout_seconds}" "runpod cli help" \
    "${py_bin}" "${repo_root}/scripts/runpod_provision.py" --help >/dev/null
}

runpod_provider_live_checks() {
  local repo_root="$1"
  local py_bin="$2"
  local timeout_seconds="$3"

  cloud_check_timeout_step "${timeout_seconds}" "runpod template-list probe" \
    "${py_bin}" "${repo_root}/scripts/runpod_provision.py" template-list --limit 3
  cloud_check_timeout_step "${timeout_seconds}" "runpod gpu-search probe" \
    "${py_bin}" "${repo_root}/scripts/runpod_provision.py" gpu-search --limit 5
  cloud_check_timeout_step "${timeout_seconds}" "runpod cli doctor" \
    bash "${repo_root}/scripts/runpod_cli_doctor.sh"
}
