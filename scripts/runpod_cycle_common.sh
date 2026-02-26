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

runpod_cycle_logs_dir() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${RUNPOD_CYCLE_LOGS_DIR:-$(runpod_cycle_dir "${repo_root}" "${run_id}")/logs}"
}

runpod_cycle_registry_file() {
  local repo_root="$1"
  printf '%s\n' "${RUNPOD_TRACKED_PODS_FILE:-${repo_root}/config/runpod_tracked_pods.jsonl}"
}

runpod_cycle_keyring_token() {
  local py_bin="$1"
  "${py_bin}" - <<'PY'
import keyring
print((keyring.get_password("runpod", "RUNPOD_API_KEY") or "").strip())
PY
}

runpod_cycle_api_token() {
  local py_bin="$1"
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    printf '%s\n' "${RUNPOD_API_KEY}"
    return 0
  fi
  runpod_cycle_keyring_token "${py_bin}"
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

runpod_cycle_pod_name() {
  local pod_json="$1"
  runpod_cycle_pod_field "${pod_json}" '(.pod_status.name // .create_response.name // "")'
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

runpod_cycle_ssh_host_key_checking() {
  local mode="${RUNPOD_SSH_HOST_KEY_CHECKING:-accept-new}"
  case "${mode}" in
    yes|no|accept-new) ;;
    *)
      echo "[runpod-cycle] invalid RUNPOD_SSH_HOST_KEY_CHECKING='${mode}' (expected yes|no|accept-new)" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "${mode}"
}

runpod_cycle_ssh_known_hosts_file() {
  local repo_root="$1"
  printf '%s\n' "${RUNPOD_SSH_KNOWN_HOSTS_FILE:-${repo_root}/config/runpod_known_hosts}"
}

runpod_cycle_prepare_ssh_client_files() {
  local repo_root="$1"
  local known_hosts
  known_hosts="$(runpod_cycle_ssh_known_hosts_file "${repo_root}")"
  mkdir -p "$(dirname "${known_hosts}")"
  touch "${known_hosts}"
  chmod 600 "${known_hosts}" 2>/dev/null || true
}

runpod_cycle_ssh_base_args() {
  local repo_root="$1"
  local pod_json="$2"
  local host port key user
  host="$(runpod_cycle_ssh_host "${pod_json}")"
  port="$(runpod_cycle_ssh_port "${pod_json}")"
  user="$(runpod_cycle_ssh_user)"
  key="$(runpod_cycle_ssh_key)"
  local host_key_checking known_hosts
  host_key_checking="$(runpod_cycle_ssh_host_key_checking)"
  known_hosts="$(runpod_cycle_ssh_known_hosts_file "${repo_root}")"
  printf '%q ' \
    ssh \
    -i "${key}" \
    -o IdentitiesOnly=yes \
    -o "StrictHostKeyChecking=${host_key_checking}" \
    -o "UserKnownHostsFile=${known_hosts}" \
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

runpod_cycle_find_model_artifact() {
  local artifacts_dir="$1"
  local run_id="${2:-}"
  local exact=""
  if [[ -n "${run_id}" && -f "${artifacts_dir}/model_${run_id}.pt" ]]; then
    exact="${artifacts_dir}/model_${run_id}.pt"
  fi
  if [[ -n "${exact}" ]]; then
    printf '%s\n' "${exact}"
    return 0
  fi
  find "${artifacts_dir}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | tail -n 1
}

runpod_cycle_registry_record() {
  local repo_root="$1"
  local source_script="$2"
  local action="$3"
  local state="$4"
  local pod_id="$5"
  local run_id="${6:-}"
  local pod_name="${7:-}"
  local public_ip="${8:-}"
  local ssh_host="${9:-}"
  local ssh_port="${10:-}"
  local note="${11:-}"

  local registry_file
  registry_file="$(runpod_cycle_registry_file "${repo_root}")"
  mkdir -p "$(dirname "${registry_file}")"

  jq -nc \
    --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg source_script "${source_script}" \
    --arg action "${action}" \
    --arg state "${state}" \
    --arg pod_id "${pod_id}" \
    --arg run_id "${run_id}" \
    --arg pod_name "${pod_name}" \
    --arg public_ip "${public_ip}" \
    --arg ssh_host "${ssh_host}" \
    --arg ssh_port "${ssh_port}" \
    --arg note "${note}" \
    '{
      ts_utc: $ts_utc,
      source_script: $source_script,
      action: $action,
      state: $state,
      pod_id: $pod_id
    }
    + (if $run_id != "" then {run_id: $run_id} else {} end)
    + (if $pod_name != "" then {pod_name: $pod_name} else {} end)
    + (if $public_ip != "" then {public_ip: $public_ip} else {} end)
    + (if $ssh_host != "" then {ssh_host: $ssh_host} else {} end)
    + (if $ssh_port != "" then {ssh_port: $ssh_port} else {} end)
    + (if $note != "" then {note: $note} else {} end)' >> "${registry_file}"
}
