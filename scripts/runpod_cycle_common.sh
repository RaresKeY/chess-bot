#!/usr/bin/env bash
set -Eeuo pipefail

runpod_cycle_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

runpod_cycle_py_bin() {
  local repo_root="$1"
  if [[ -x "${repo_root}/.venv/bin/python" ]]; then
    printf '%s\n' "${repo_root}/.venv/bin/python"
  else
    printf '%s\n' "python3"
  fi
}

runpod_cycle_require_cmd() {
  local c="$1"
  command -v "${c}" >/dev/null 2>&1 || {
    echo "[runpod-cycle] missing required command: ${c}" >&2
    exit 1
  }
}

runpod_cycle_run_id() {
  printf '%s\n' "${RUNPOD_CYCLE_RUN_ID:-runpod-cycle-$(date -u +%Y%m%dT%H%M%SZ)}"
}

runpod_cycle_dir() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${repo_root}/artifacts/runpod_cycles/${run_id}"
}

runpod_cycle_provision_json() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${RUNPOD_POD_JSON:-$(runpod_cycle_dir "${repo_root}" "${run_id}")/provision.json}"
}

runpod_cycle_report_md() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${RUNPOD_CYCLE_REPORT_MD:-$(runpod_cycle_dir "${repo_root}" "${run_id}")/reports/observations.md}"
}

runpod_cycle_keyring_token() {
  local py_bin="$1"
  "${py_bin}" - <<'PY'
import keyring
print((keyring.get_password("runpod", "RUNPOD_API_KEY") or "").strip())
PY
}

runpod_cycle_pod_field() {
  local pod_json="$1"
  local jq_expr="$2"
  jq -r "${jq_expr}" "${pod_json}"
}

runpod_cycle_public_ip() {
  local pod_json="$1"
  runpod_cycle_pod_field "${pod_json}" '(.pod_status.publicIp // .create_response.publicIp // "")'
}

runpod_cycle_ssh_port() {
  local pod_json="$1"
  if [[ -n "${RUNPOD_SSH_PORT:-}" ]]; then
    printf '%s\n' "${RUNPOD_SSH_PORT}"
    return 0
  fi
  runpod_cycle_pod_field "${pod_json}" '((.pod_status.portMappings["22"] // .create_response.portMappings["22"] // "22") | tostring)'
}

runpod_cycle_pod_id() {
  local pod_json="$1"
  runpod_cycle_pod_field "${pod_json}" '(.pod_id // .create_response.id // .create_response.podId // "")'
}

runpod_cycle_remote_repo_dir() {
  local pod_json="$1"
  runpod_cycle_pod_field "${pod_json}" '(.pod_status.env.REPO_DIR // .create_response.env.REPO_DIR // "/workspace/chess-bot")'
}

runpod_cycle_ssh_user() {
  printf '%s\n' "${RUNPOD_SSH_USER:-runner}"
}

runpod_cycle_ssh_host() {
  local pod_json="$1"
  if [[ -n "${RUNPOD_SSH_HOST:-}" ]]; then
    printf '%s\n' "${RUNPOD_SSH_HOST}"
    return 0
  fi
  runpod_cycle_public_ip "${pod_json}"
}

runpod_cycle_ssh_key() {
  printf '%s\n' "${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
}

runpod_cycle_ssh_base_args() {
  local pod_json="$1"
  local host port key user
  host="$(runpod_cycle_ssh_host "${pod_json}")"
  port="$(runpod_cycle_ssh_port "${pod_json}")"
  user="$(runpod_cycle_ssh_user)"
  key="$(runpod_cycle_ssh_key)"
  printf '%q ' \
    ssh \
    -i "${key}" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/tmp/runpod_known_hosts \
    -p "${port}" \
    "${user}@${host}"
}

runpod_cycle_append_report() {
  local report_path="$1"
  shift
  mkdir -p "$(dirname "${report_path}")"
  {
    for line in "$@"; do
      printf '%s\n' "$line"
    done
  } >> "${report_path}"
}
