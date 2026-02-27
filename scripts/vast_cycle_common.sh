#!/usr/bin/env bash
set -Eeuo pipefail

vast_cycle_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

vast_cycle_py_bin() {
  local repo_root="$1"
  if [[ -x "${repo_root}/.venv/bin/python" ]]; then
    printf '%s\n' "${repo_root}/.venv/bin/python"
  else
    printf '%s\n' "python3"
  fi
}

vast_cycle_require_cmd() {
  local c="$1"
  command -v "${c}" >/dev/null 2>&1 || {
    echo "[vast-cycle] missing required command: ${c}" >&2
    exit 1
  }
}

vast_cycle_run_id() {
  printf '%s\n' "${VAST_CYCLE_RUN_ID:-vast-cycle-$(date -u +%Y%m%dT%H%M%SZ)}"
}

vast_cycle_dir() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${repo_root}/artifacts/vast_cycles/${run_id}"
}

vast_cycle_provision_json() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${VAST_INSTANCE_JSON:-$(vast_cycle_dir "${repo_root}" "${run_id}")/provision.json}"
}

vast_cycle_report_md() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${VAST_CYCLE_REPORT_MD:-$(vast_cycle_dir "${repo_root}" "${run_id}")/reports/observations.md}"
}

vast_cycle_logs_dir() {
  local repo_root="$1"
  local run_id="$2"
  printf '%s\n' "${VAST_CYCLE_LOGS_DIR:-$(vast_cycle_dir "${repo_root}" "${run_id}")/logs}"
}

vast_cycle_registry_file() {
  local repo_root="$1"
  printf '%s\n' "${VAST_TRACKED_INSTANCES_FILE:-${repo_root}/config/vast_tracked_instances.jsonl}"
}

vast_cycle_keyring_token() {
  local py_bin="$1"
  "${py_bin}" - <<'PY'
try:
    import keyring
except Exception:
    print("")
    raise SystemExit(0)
try:
    print((keyring.get_password("vast", "VAST_API_KEY") or "").strip())
except Exception:
    print("")
PY
}

vast_cycle_dotenv_token() {
  local py_bin="$1"
  local repo_root="$2"
  CHESSBOT_REPO_ROOT="${repo_root}" "${py_bin}" - <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(os.environ.get("CHESSBOT_REPO_ROOT", ".")).resolve()
sys.path.insert(0, str(repo_root))
from src.chessbot.secrets import default_dotenv_paths, lookup_dotenv_value

paths = default_dotenv_paths(
    repo_root=repo_root,
    override_var_names=("VAST_DOTENV_PATH", "CHESSBOT_DOTENV_PATH"),
    fallback_filenames=(".env.vast", ".env"),
)
print(lookup_dotenv_value(("VAST_API_KEY",), paths))
PY
}

vast_cycle_api_token() {
  local py_bin="$1"
  local repo_root="${2:-$(vast_cycle_repo_root)}"
  if [[ -n "${VAST_API_KEY:-}" ]]; then
    printf '%s\n' "${VAST_API_KEY}"
    return 0
  fi
  local keyring_token
  keyring_token="$(vast_cycle_keyring_token "${py_bin}")"
  if [[ -n "${keyring_token}" ]]; then
    printf '%s\n' "${keyring_token}"
    return 0
  fi
  vast_cycle_dotenv_token "${py_bin}" "${repo_root}"
}

vast_cycle_instance_field() {
  local provision_json="$1"
  local jq_expr="$2"
  jq -r "${jq_expr}" "${provision_json}"
}

vast_cycle_instance_id() {
  local provision_json="$1"
  vast_cycle_instance_field "${provision_json}" '(.instance_id // .create_response.new_contract // .create_response.instance_id // .show_response.instances.id // 0 | tonumber)'
}

vast_cycle_instance_label() {
  local provision_json="$1"
  vast_cycle_instance_field "${provision_json}" '(.show_response.instances.label // .create_response.label // "")'
}

vast_cycle_ssh_host() {
  local provision_json="$1"
  if [[ -n "${VAST_SSH_HOST:-}" ]]; then
    printf '%s\n' "${VAST_SSH_HOST}"
    return 0
  fi
  vast_cycle_instance_field "${provision_json}" '(.show_response.instances.ssh_host // .show_response.instances.public_ipaddr // .show_response.instances.public_ip // "")'
}

vast_cycle_ssh_port() {
  local provision_json="$1"
  if [[ -n "${VAST_SSH_PORT:-}" ]]; then
    printf '%s\n' "${VAST_SSH_PORT}"
    return 0
  fi
  vast_cycle_instance_field "${provision_json}" '(.show_response.instances.ssh_port // .show_response.instances.direct_port_start // 22 | tostring)'
}

vast_cycle_ssh_user() {
  printf '%s\n' "${VAST_SSH_USER:-root}"
}

vast_cycle_append_report() {
  local report_path="$1"
  shift
  mkdir -p "$(dirname "${report_path}")"
  {
    for line in "$@"; do
      printf '%s\n' "$line"
    done
  } >> "${report_path}"
}

vast_cycle_registry_record() {
  local repo_root="$1"
  local source_script="$2"
  local action="$3"
  local state="$4"
  local instance_id="$5"
  local run_id="$6"
  local instance_label="$7"
  local ssh_host="$8"
  local ssh_port="$9"
  local note="${10:-}"

  local registry
  registry="$(vast_cycle_registry_file "${repo_root}")"
  mkdir -p "$(dirname "${registry}")"

  local ts_utc
  ts_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  jq -nc \
    --arg ts_utc "${ts_utc}" \
    --arg source_script "${source_script}" \
    --arg action "${action}" \
    --arg state "${state}" \
    --argjson instance_id "${instance_id:-0}" \
    --arg run_id "${run_id}" \
    --arg instance_label "${instance_label}" \
    --arg ssh_host "${ssh_host}" \
    --arg ssh_port "${ssh_port}" \
    --arg note "${note}" \
    '{ts_utc:$ts_utc,source_script:$source_script,action:$action,state:$state,instance_id:$instance_id,run_id:$run_id,instance_label:$instance_label,ssh_host:$ssh_host,ssh_port:$ssh_port,note:$note}' \
    >> "${registry}"
}
