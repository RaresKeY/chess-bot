#!/usr/bin/env bash
set -Eeuo pipefail

vast_provider_local_checks() {
  local repo_root="$1"
  local py_bin="$2"
  local timeout_seconds="$3"

  cloud_check_timeout_step "${timeout_seconds}" "vast script contract check" \
    test -x "${repo_root}/scripts/vast_provision.py"
  cloud_check_timeout_step "${timeout_seconds}" "vast cli help" \
    "${py_bin}" "${repo_root}/scripts/vast_provision.py" --help >/dev/null
}

vast_provider_live_checks() {
  local repo_root="$1"
  local py_bin="$2"
  local timeout_seconds="$3"

  cloud_check_timeout_step "${timeout_seconds}" "vast offer-search probe" \
    "${py_bin}" "${repo_root}/scripts/vast_provision.py" offer-search --limit 3
  cloud_check_timeout_step "${timeout_seconds}" "vast instance-list probe" \
    "${py_bin}" "${repo_root}/scripts/vast_provision.py" instance-list
}
