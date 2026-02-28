#!/usr/bin/env bash
set -Eeuo pipefail

RUNNER_USER="${RUNNER_USER:-runner}"
RUNNER_HOME="$(getent passwd "${RUNNER_USER}" | cut -d: -f6)"
REPO_URL="${REPO_URL:-https://github.com/RaresKeY/chess-bot.git}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-/workspace/chess-bot}"
VENV_DIR="${VENV_DIR:-/opt/venvs/chessbot}"
CLONE_REPO_ON_START="${CLONE_REPO_ON_START:-1}"
GIT_AUTO_PULL="${GIT_AUTO_PULL:-1}"
SYNC_REQUIREMENTS_ON_START="${SYNC_REQUIREMENTS_ON_START:-1}"
FORCE_PIP_SYNC="${FORCE_PIP_SYNC:-0}"
START_SSHD="${START_SSHD:-1}"
START_JUPYTER="${START_JUPYTER:-1}"
START_INFERENCE_API="${START_INFERENCE_API:-1}"
START_HF_WATCH="${START_HF_WATCH:-0}"
START_IDLE_WATCHDOG="${START_IDLE_WATCHDOG:-0}"
START_OTEL_COLLECTOR="${START_OTEL_COLLECTOR:-1}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
INFERENCE_API_HOST="${INFERENCE_API_HOST:-0.0.0.0}"
INFERENCE_API_PORT="${INFERENCE_API_PORT:-8000}"
INFERENCE_API_MODEL_PATH="${INFERENCE_API_MODEL_PATH:-latest}"
INFERENCE_API_DEVICE="${INFERENCE_API_DEVICE:-auto}"
INFERENCE_API_TOPK_DEFAULT="${INFERENCE_API_TOPK_DEFAULT:-10}"
INFERENCE_API_VERBOSE="${INFERENCE_API_VERBOSE:-1}"
HF_SYNC_SOURCE_DIR="${HF_SYNC_SOURCE_DIR:-${REPO_DIR}/artifacts}"
HF_SYNC_INTERVAL_SECONDS="${HF_SYNC_INTERVAL_SECONDS:-120}"
HF_SYNC_PATTERNS="${HF_SYNC_PATTERNS:-*.pt,*.json}"
HF_SYNC_VERBOSE="${HF_SYNC_VERBOSE:-1}"
IDLE_TIMEOUT_SECONDS="${IDLE_TIMEOUT_SECONDS:-3600}"
IDLE_CHECK_INTERVAL_SECONDS="${IDLE_CHECK_INTERVAL_SECONDS:-60}"
IDLE_GPU_UTIL_THRESHOLD="${IDLE_GPU_UTIL_THRESHOLD:-10}"
IDLE_GPU_MEM_MB_THRESHOLD="${IDLE_GPU_MEM_MB_THRESHOLD:-1024}"
IDLE_WATCHDOG_VERBOSE="${IDLE_WATCHDOG_VERBOSE:-1}"
RUNPOD_PHASE_TIMING_ENABLED="${RUNPOD_PHASE_TIMING_ENABLED:-1}"
RUNPOD_PHASE_TIMING_LOG="${RUNPOD_PHASE_TIMING_LOG:-${REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl}"
RUNPOD_PHASE_TIMING_RUN_ID="${RUNPOD_PHASE_TIMING_RUN_ID:-runpod-entrypoint-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUNPOD_MODULE_IMAGE_DIR="${RUNPOD_MODULE_IMAGE_DIR:-/opt/runpod_cloud_training}"
RUNPOD_MODULE_DIR="${RUNPOD_MODULE_DIR:-}"
RUNPOD_HEALTHCHECKS_URL="${RUNPOD_HEALTHCHECKS_URL:-${HEALTHCHECKS_URL:-}}"
OTEL_CONFIG_PATH="${OTEL_CONFIG_PATH:-}"
OTEL_FILE_EXPORT_PATH="${OTEL_FILE_EXPORT_PATH:-${REPO_DIR}/artifacts/telemetry/otel/collector.jsonl}"

PIDS=()

log() {
  printf '[entrypoint] %s\n' "$*"
}

healthchecks_ping() {
  local kind="$1"
  local msg="${2:-}"
  [[ -n "${RUNPOD_HEALTHCHECKS_URL}" ]] || return 0
  command -v curl >/dev/null 2>&1 || return 0
  local url="${RUNPOD_HEALTHCHECKS_URL%/}"
  case "${kind}" in
    start) url="${url}/start" ;;
    success) ;;
    fail) url="${url}/fail" ;;
    log) url="${url}/log" ;;
  esac
  curl -fsS -m 10 -X POST --data-raw "${msg}" "${url}" >/dev/null 2>&1 || true
}

_now_epoch_ms() {
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
  printf '{"ts_epoch_ms":%s,"source":"runpod_entrypoint","run_id":"%s","phase":"%s","status":"%s","elapsed_ms":%s%s}\n' \
    "$(_now_epoch_ms)" "${RUNPOD_PHASE_TIMING_RUN_ID}" "${phase}" "${status}" "${elapsed_ms}" "${extra}" \
    >> "${RUNPOD_PHASE_TIMING_LOG}" 2>/dev/null || true
}

run_timed_phase() {
  local phase="$1"
  shift
  local t0 t1
  t0="$(_now_epoch_ms)"
  if "$@"; then
    t1="$(_now_epoch_ms)"
    log_phase_timing "${phase}" "ok" "$((t1 - t0))"
    return 0
  fi
  local rc=$?
  t1="$(_now_epoch_ms)"
  log_phase_timing "${phase}" "error" "$((t1 - t0))" ",\"exit_code\":${rc}"
  return "${rc}"
}

run_as_runner() {
  su -s /bin/bash "${RUNNER_USER}" -c "$*"
}

resolve_module_dir() {
  local repo_module_dir="${REPO_DIR}/deploy/runpod_cloud_training"
  if [[ -n "${RUNPOD_MODULE_DIR}" ]]; then
    log "Using explicit RUNPOD_MODULE_DIR=${RUNPOD_MODULE_DIR}"
    return
  fi
  if [[ -d "${repo_module_dir}" ]]; then
    RUNPOD_MODULE_DIR="${repo_module_dir}"
    log "Using repo module directory (latest from git pull): ${RUNPOD_MODULE_DIR}"
    return
  fi
  RUNPOD_MODULE_DIR="${RUNPOD_MODULE_IMAGE_DIR}"
  log "Repo module directory missing; falling back to image module directory: ${RUNPOD_MODULE_DIR}"
}

ensure_ssh_keys() {
  local ssh_dir="${RUNNER_HOME}/.ssh"
  mkdir -p "${ssh_dir}"
  chmod 700 "${ssh_dir}"
  touch "${ssh_dir}/authorized_keys"
  chmod 600 "${ssh_dir}/authorized_keys"
  if [[ -n "${AUTHORIZED_KEYS:-}" ]]; then
    printf '%s\n' "${AUTHORIZED_KEYS}" > "${ssh_dir}/authorized_keys"
    log "Loaded AUTHORIZED_KEYS from environment"
  fi
  chown -R "${RUNNER_USER}:${RUNNER_USER}" "${ssh_dir}"
}

ensure_runner_ssh_account() {
  local shadow_entry shadow_hash
  shadow_entry="$(getent shadow "${RUNNER_USER}" 2>/dev/null || true)"
  shadow_hash="$(printf '%s' "${shadow_entry}" | cut -d: -f2)"
  if [[ -z "${shadow_hash}" ]]; then
    log "No shadow entry found for ${RUNNER_USER}; skipping account unlock check"
    return 0
  fi
  if [[ "${shadow_hash}" == '!'* || "${shadow_hash}" == '*'* ]]; then
    if passwd -d "${RUNNER_USER}" >/dev/null 2>&1; then
      log "Unlocked ${RUNNER_USER} account for SSH public-key auth"
    elif usermod -U "${RUNNER_USER}" >/dev/null 2>&1; then
      log "Unlocked ${RUNNER_USER} account for SSH public-key auth via usermod"
    else
      log "Warning: failed to unlock ${RUNNER_USER} account; direct SSH pubkey auth may be denied"
    fi
  fi
}

configure_sshd() {
  mkdir -p /var/run/sshd /etc/ssh/sshd_config.d
  cat >/etc/ssh/sshd_config.d/chessbot.conf <<EOF
Port 22
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
ChallengeResponseAuthentication no
UsePAM no
X11Forwarding no
AuthorizedKeysFile .ssh/authorized_keys
LogLevel VERBOSE
AllowUsers ${RUNNER_USER}
EOF
}

clone_or_update_repo() {
  if [[ "${CLONE_REPO_ON_START}" != "1" ]]; then
    log "Repo clone on start disabled (CLONE_REPO_ON_START=${CLONE_REPO_ON_START})"
    return
  fi

  mkdir -p "$(dirname "${REPO_DIR}")"
  if [[ -e "${REPO_DIR}" && ! -d "${REPO_DIR}" ]]; then
    log "REPO_DIR exists but is not a directory: ${REPO_DIR}"
    return 1
  fi
  if [[ -d "${REPO_DIR}" && ! -d "${REPO_DIR}/.git" ]]; then
    if [[ -z "$(ls -A "${REPO_DIR}" 2>/dev/null)" ]]; then
      log "Cloning repo ${REPO_URL} -> existing empty dir ${REPO_DIR}"
      git clone --branch "${REPO_REF}" --single-branch "${REPO_URL}" "${REPO_DIR}"
      chown -R "${RUNNER_USER}:${RUNNER_USER}" "${REPO_DIR}"
      return 0
    fi
    if [[ -f "${REPO_DIR}/requirements.txt" ]]; then
      log "Using existing non-git repo directory at ${REPO_DIR} (requirements.txt detected); skipping clone/pull"
      return 0
    fi
    log "REPO_DIR exists and is non-empty but not a git repo: ${REPO_DIR}; skipping clone/pull"
    log "Set REPO_DIR to an empty path or mount a valid repo checkout to enable sync/startup services"
    return 0
  fi
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning repo ${REPO_URL} -> ${REPO_DIR}"
    git clone --branch "${REPO_REF}" --single-branch "${REPO_URL}" "${REPO_DIR}"
    chown -R "${RUNNER_USER}:${RUNNER_USER}" "${REPO_DIR}"
    return
  fi

  if [[ "${GIT_AUTO_PULL}" == "1" ]]; then
    log "Updating repo in ${REPO_DIR} to ${REPO_REF}"
    git -C "${REPO_DIR}" fetch origin "${REPO_REF}" --depth 1 || git -C "${REPO_DIR}" fetch origin "${REPO_REF}"
    git -C "${REPO_DIR}" checkout "${REPO_REF}" || true
    git -C "${REPO_DIR}" pull --ff-only origin "${REPO_REF}" || log "git pull skipped (non-ff or local changes)"
  else
    log "Repo exists; skipping pull (GIT_AUTO_PULL=${GIT_AUTO_PULL})"
  fi
}

sync_repo_requirements() {
  local req_file="${REPO_DIR}/requirements.txt"
  local stamp_file="${VENV_DIR}/.repo_requirements.sha256"
  if [[ "${SYNC_REQUIREMENTS_ON_START}" != "1" ]]; then
    log "Requirement sync disabled"
    return
  fi
  if [[ ! -f "${req_file}" ]]; then
    log "No repo requirements.txt found at ${req_file}; skipping sync"
    return
  fi
  local current_hash
  current_hash="$(sha256sum "${req_file}" | awk '{print $1}')"
  local previous_hash=""
  if [[ -f "${stamp_file}" ]]; then
    previous_hash="$(cat "${stamp_file}")"
  fi
  if [[ "${FORCE_PIP_SYNC}" == "1" || "${current_hash}" != "${previous_hash}" ]]; then
    log "Installing repo requirements into ${VENV_DIR} (hash changed)"
    "${VENV_DIR}/bin/pip" install -r "${req_file}"
    printf '%s\n' "${current_hash}" > "${stamp_file}"
  else
    log "Repo requirements already synced"
  fi
}

python_module_available() {
  local module_name="$1"
  "${VENV_DIR}/bin/python" -c "import ${module_name}" >/dev/null 2>&1
}

start_bg() {
  local name="$1"
  shift
  log "Starting ${name}: $*"
  "$@" &
  PIDS+=("$!")
}

start_runner_bg() {
  local name="$1"
  shift
  local cmd="$*"
  log "Starting ${name} as ${RUNNER_USER}: ${cmd}"
  su -s /bin/bash "${RUNNER_USER}" -c "${cmd}" &
  PIDS+=("$!")
}

cleanup() {
  log "Shutting down services"
  healthchecks_ping "log" "runpod entrypoint shutting down"
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}

trap cleanup EXIT INT TERM
trap 'healthchecks_ping fail "runpod entrypoint error"' ERR

healthchecks_ping start "runpod entrypoint boot"

mkdir -p /workspace
chown -R "${RUNNER_USER}:${RUNNER_USER}" /workspace "${RUNNER_HOME}"

run_timed_phase "ensure_ssh_keys" ensure_ssh_keys
run_timed_phase "ensure_runner_ssh_account" ensure_runner_ssh_account
run_timed_phase "configure_sshd" configure_sshd
run_timed_phase "clone_or_update_repo" clone_or_update_repo
run_timed_phase "sync_repo_requirements" sync_repo_requirements
run_timed_phase "resolve_module_dir" resolve_module_dir

if [[ ! -d "${REPO_DIR}" ]]; then
  log "Repo dir ${REPO_DIR} is missing. Set CLONE_REPO_ON_START=1 or mount the repo."
  exit 1
fi

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ -z "${JUPYTER_TOKEN}" ]]; then
  JUPYTER_TOKEN="$(openssl rand -hex 16)"
  export JUPYTER_TOKEN
  log "Generated Jupyter token"
fi

if [[ "${START_SSHD}" == "1" ]]; then
  run_timed_phase "start_sshd" start_bg "sshd" /usr/sbin/sshd -D -e
fi

if [[ "${START_JUPYTER}" == "1" ]]; then
  run_timed_phase "start_jupyter" start_runner_bg "jupyterlab" \
    "cd '${REPO_DIR}' && '${VENV_DIR}/bin/jupyter' lab \
      --ServerApp.ip=0.0.0.0 \
      --ServerApp.port='${JUPYTER_PORT}' \
      --ServerApp.token='${JUPYTER_TOKEN}' \
      --ServerApp.password='' \
      --ServerApp.allow_remote_access=True \
      --ServerApp.root_dir='${REPO_DIR}' \
      --no-browser"
fi

if [[ "${START_INFERENCE_API}" == "1" ]]; then
  if ! python_module_available torch; then
    log "Skipping inference API start: torch is not installed in ${VENV_DIR} yet"
    START_INFERENCE_API="0"
  fi
fi

if [[ "${START_INFERENCE_API}" == "1" ]]; then
  inference_verbose_flag="--no-verbose"
  if [[ "${INFERENCE_API_VERBOSE}" == "1" ]]; then
    inference_verbose_flag="--verbose"
  fi
  run_timed_phase "start_inference_api" start_runner_bg "inference-api" \
    "cd '${REPO_DIR}' && '${VENV_DIR}/bin/python' '${RUNPOD_MODULE_DIR}/inference_api.py' \
      --repo-dir '${REPO_DIR}' \
      --model '${INFERENCE_API_MODEL_PATH}' \
      --host '${INFERENCE_API_HOST}' \
      --port '${INFERENCE_API_PORT}' \
      --device '${INFERENCE_API_DEVICE}' \
      --topk-default '${INFERENCE_API_TOPK_DEFAULT}' \
      ${inference_verbose_flag}"
fi

if [[ "${START_HF_WATCH}" == "1" ]]; then
  hf_watch_verbose_flag="--no-verbose"
  if [[ "${HF_SYNC_VERBOSE}" == "1" ]]; then
    hf_watch_verbose_flag="--verbose"
  fi
  run_timed_phase "start_hf_watch" start_runner_bg "hf-auto-sync" \
    "cd '${REPO_DIR}' && '${VENV_DIR}/bin/python' '${RUNPOD_MODULE_DIR}/hf_auto_sync_watch.py' \
      --source-dir '${HF_SYNC_SOURCE_DIR}' \
      --patterns '${HF_SYNC_PATTERNS}' \
      --interval-seconds '${HF_SYNC_INTERVAL_SECONDS}' \
      ${hf_watch_verbose_flag}"
fi

if [[ "${START_IDLE_WATCHDOG}" == "1" ]]; then
  idle_watchdog_verbose_flag="--no-verbose"
  if [[ "${IDLE_WATCHDOG_VERBOSE}" == "1" ]]; then
    idle_watchdog_verbose_flag="--verbose"
  fi
  run_timed_phase "start_idle_watchdog" start_bg "idle-watchdog" \
    "${VENV_DIR}/bin/python" "${RUNPOD_MODULE_DIR}/idle_watchdog.py" \
      --idle-seconds "${IDLE_TIMEOUT_SECONDS}" \
      --check-interval-seconds "${IDLE_CHECK_INTERVAL_SECONDS}" \
      --gpu-util-threshold "${IDLE_GPU_UTIL_THRESHOLD}" \
      --gpu-mem-mb-threshold "${IDLE_GPU_MEM_MB_THRESHOLD}" \
      --jupyter-port "${JUPYTER_PORT}" \
      --api-port "${INFERENCE_API_PORT}" \
      ${idle_watchdog_verbose_flag}
fi

if [[ "${START_OTEL_COLLECTOR}" == "1" ]]; then
  if ! command -v otelcol-contrib >/dev/null 2>&1; then
    log "Skipping OpenTelemetry Collector start: otelcol-contrib not found"
  else
    OTEL_CONFIG_RESOLVED="${OTEL_CONFIG_PATH:-${RUNPOD_MODULE_DIR}/otel-collector-config.yaml}"
    run_timed_phase "start_otel_collector" start_runner_bg "otel-collector" \
      "mkdir -p \"\$(dirname '${OTEL_FILE_EXPORT_PATH}')\" && OTEL_FILE_EXPORT_PATH='${OTEL_FILE_EXPORT_PATH}' otelcol-contrib --config '${OTEL_CONFIG_RESOLVED}'"
  fi
fi

log "Startup complete"
healthchecks_ping success "runpod entrypoint startup complete"
log "Repo: ${REPO_DIR} (${REPO_REF})"
log "SSH: port 22 (user=${RUNNER_USER})"
if [[ "${START_JUPYTER}" == "1" ]]; then
  log "Jupyter: port ${JUPYTER_PORT} token=${JUPYTER_TOKEN}"
fi
if [[ "${START_INFERENCE_API}" == "1" ]]; then
  log "Inference API: http://${INFERENCE_API_HOST}:${INFERENCE_API_PORT}"
fi
if [[ "${START_OTEL_COLLECTOR}" == "1" ]]; then
  log "OpenTelemetry Collector: enabled"
fi

if [[ "$#" -gt 0 ]]; then
  custom_t0="$(_now_epoch_ms)"
  start_bg "custom-cmd" "$@"
  custom_t1="$(_now_epoch_ms)"
  log_phase_timing "start_custom_cmd" "ok" "$((custom_t1 - custom_t0))"
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  log "No services started. Sleeping."
  tail -f /dev/null
fi

wait -n "${PIDS[@]}"
exit $?
