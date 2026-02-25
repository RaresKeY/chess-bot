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

PIDS=()

log() {
  printf '[entrypoint] %s\n' "$*"
}

run_as_runner() {
  su -s /bin/bash "${RUNNER_USER}" -c "$*"
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
AllowUsers ${RUNNER_USER}
EOF
}

clone_or_update_repo() {
  if [[ "${CLONE_REPO_ON_START}" != "1" ]]; then
    log "Repo clone on start disabled (CLONE_REPO_ON_START=${CLONE_REPO_ON_START})"
    return
  fi

  mkdir -p "$(dirname "${REPO_DIR}")"
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
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}

trap cleanup EXIT INT TERM

mkdir -p /workspace
chown -R "${RUNNER_USER}:${RUNNER_USER}" /workspace "${RUNNER_HOME}"

ensure_ssh_keys
configure_sshd
clone_or_update_repo
sync_repo_requirements

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
  start_bg "sshd" /usr/sbin/sshd -D -e
fi

if [[ "${START_JUPYTER}" == "1" ]]; then
  start_runner_bg "jupyterlab" \
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
  inference_verbose_flag="--no-verbose"
  if [[ "${INFERENCE_API_VERBOSE}" == "1" ]]; then
    inference_verbose_flag="--verbose"
  fi
  start_runner_bg "inference-api" \
    "cd '${REPO_DIR}' && '${VENV_DIR}/bin/python' /opt/runpod_cloud_training/inference_api.py \
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
  start_runner_bg "hf-auto-sync" \
    "cd '${REPO_DIR}' && '${VENV_DIR}/bin/python' /opt/runpod_cloud_training/hf_auto_sync_watch.py \
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
  start_bg "idle-watchdog" \
    "${VENV_DIR}/bin/python" /opt/runpod_cloud_training/idle_watchdog.py \
      --idle-seconds "${IDLE_TIMEOUT_SECONDS}" \
      --check-interval-seconds "${IDLE_CHECK_INTERVAL_SECONDS}" \
      --gpu-util-threshold "${IDLE_GPU_UTIL_THRESHOLD}" \
      --gpu-mem-mb-threshold "${IDLE_GPU_MEM_MB_THRESHOLD}" \
      --jupyter-port "${JUPYTER_PORT}" \
      --api-port "${INFERENCE_API_PORT}" \
      ${idle_watchdog_verbose_flag}
fi

log "Startup complete"
log "Repo: ${REPO_DIR} (${REPO_REF})"
log "SSH: port 22 (user=${RUNNER_USER})"
if [[ "${START_JUPYTER}" == "1" ]]; then
  log "Jupyter: port ${JUPYTER_PORT} token=${JUPYTER_TOKEN}"
fi
if [[ "${START_INFERENCE_API}" == "1" ]]; then
  log "Inference API: http://${INFERENCE_API_HOST}:${INFERENCE_API_PORT}"
fi

if [[ "$#" -gt 0 ]]; then
  start_bg "custom-cmd" "$@"
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  log "No services started. Sleeping."
  tail -f /dev/null
fi

wait -n "${PIDS[@]}"
exit $?
