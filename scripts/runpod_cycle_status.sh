#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-$(runpod_cycle_run_id)}"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
REPORT_DIR="$(dirname "${REPORT_MD}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${REPORT_DIR}" "${LOGS_DIR}"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd ssh
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

WATCH_MODE=0
POLL_SECONDS="${RUNPOD_STATUS_POLL_SECONDS:-5}"
WRITE_SNAPSHOT=1
for arg in "$@"; do
  case "${arg}" in
    --watch) WATCH_MODE=1 ;;
    --no-write) WRITE_SNAPSHOT=0 ;;
  esac
done

if [[ ! -f "${PROVISION_JSON}" ]]; then
  echo "[runpod-cycle-status] missing provision file: ${PROVISION_JSON}" >&2
  exit 1
fi

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"
SSH_OPTS=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}")

REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_HF_FETCH_MANIFEST="${REMOTE_RUN_DIR}/hf_dataset_fetch_manifest.json"
REMOTE_CONTEXT_JSON="${REMOTE_RUN_DIR}/context_probe_${RUN_ID}.json"
REMOTE_TRAIN_LOG="${REMOTE_RUN_DIR}/train_stdout_${RUN_ID}.log"
REMOTE_PROGRESS_JSONL="${REMOTE_RUN_DIR}/train_progress_${RUN_ID}.jsonl"
REMOTE_TRAIN_PID_FILE="${REMOTE_RUN_DIR}/train_pid.txt"
REMOTE_TRAIN_EXIT_CODE_FILE="${REMOTE_RUN_DIR}/train_exit_code.txt"
REMOTE_BEST_CHECKPOINT="${REMOTE_RUN_DIR}/model_best_${RUN_ID}.pt"
REMOTE_EPOCH_CHECKPOINT_DIR="${REMOTE_RUN_DIR}/epoch_checkpoints"

collect_once() {
  local ts_utc
  ts_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local pod_id pod_status
  pod_id="$(jq -r '.pod_id // empty' "${PROVISION_JSON}")"
  pod_status="$(jq -r '.pod_status.desiredStatus // empty' "${PROVISION_JSON}")"

  local local_watch_running=0
  pgrep -f "scripts/runpod_cycle_watch_progress.sh" >/dev/null 2>&1 && local_watch_running=1
  local local_full_flow_running=0
  pgrep -f "scripts/runpod_cycle_full_train_hf.sh" >/dev/null 2>&1 && local_full_flow_running=1

  local remote_json
  if ! remote_json="$(
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
      "RUN_ID='${RUN_ID}' REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' REMOTE_RUN_DIR='${REMOTE_RUN_DIR}' REMOTE_HF_FETCH_MANIFEST='${REMOTE_HF_FETCH_MANIFEST}' REMOTE_CONTEXT_JSON='${REMOTE_CONTEXT_JSON}' REMOTE_TRAIN_LOG='${REMOTE_TRAIN_LOG}' REMOTE_PROGRESS_JSONL='${REMOTE_PROGRESS_JSONL}' REMOTE_TRAIN_PID_FILE='${REMOTE_TRAIN_PID_FILE}' REMOTE_TRAIN_EXIT_CODE_FILE='${REMOTE_TRAIN_EXIT_CODE_FILE}' REMOTE_BEST_CHECKPOINT='${REMOTE_BEST_CHECKPOINT}' REMOTE_EPOCH_CHECKPOINT_DIR='${REMOTE_EPOCH_CHECKPOINT_DIR}' /bin/bash -s" <<'EOF'
set -Eeuo pipefail
python3 - <<'PY'
import json
import os
import pathlib
import subprocess

def line_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("rb") as f:
        for _ in f:
            c += 1
    return c

run_dir = pathlib.Path(os.environ["REMOTE_RUN_DIR"])
repo_dir = pathlib.Path(os.environ["REMOTE_REPO_DIR"])
pid_file = pathlib.Path(os.environ["REMOTE_TRAIN_PID_FILE"])
exit_file = pathlib.Path(os.environ["REMOTE_TRAIN_EXIT_CODE_FILE"])
progress_file = pathlib.Path(os.environ["REMOTE_PROGRESS_JSONL"])
train_log = pathlib.Path(os.environ["REMOTE_TRAIN_LOG"])
epoch_dir = pathlib.Path(os.environ["REMOTE_EPOCH_CHECKPOINT_DIR"])
best_ckpt = pathlib.Path(os.environ["REMOTE_BEST_CHECKPOINT"])
manifest = pathlib.Path(os.environ["REMOTE_HF_FETCH_MANIFEST"])
context = pathlib.Path(os.environ["REMOTE_CONTEXT_JSON"])

pid = None
train_pid_alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        pid = None
if pid is not None:
    try:
        train_pid_alive = pathlib.Path(f"/proc/{pid}").exists()
    except Exception:
        train_pid_alive = False

exit_code = None
if exit_file.exists():
    try:
        exit_code = int(exit_file.read_text(encoding="utf-8").strip())
    except Exception:
        exit_code = None

gpu_line = ""
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=3.0,
    )
    gpu_line = out.strip().splitlines()[0] if out.strip() else ""
except Exception:
    gpu_line = ""

epoch_ckpt_count = 0
if epoch_dir.exists():
    epoch_ckpt_count = len(list(epoch_dir.glob("model_epoch_*.pt")))

state = "unknown"
if not repo_dir.exists():
    state = "repo_not_ready"
elif not run_dir.exists():
    state = "run_dir_not_created"
elif train_pid_alive:
    state = "training_running"
elif exit_code is not None:
    state = "training_finished"
elif manifest.exists() and context.exists():
    state = "pretrain_ready_or_launching"
elif manifest.exists():
    state = "hf_fetch_done"
else:
    state = "bootstrap_or_fetching"

print(
    json.dumps(
        {
            "remote_state": state,
            "repo_dir_exists": repo_dir.exists(),
            "run_dir_exists": run_dir.exists(),
            "hf_manifest_exists": manifest.exists(),
            "context_exists": context.exists(),
            "train_log_exists": train_log.exists(),
            "train_log_lines": line_count(train_log),
            "progress_exists": progress_file.exists(),
            "progress_lines": line_count(progress_file),
            "train_pid_file_exists": pid_file.exists(),
            "train_pid": pid,
            "train_pid_alive": train_pid_alive,
            "train_exit_file_exists": exit_file.exists(),
            "train_exit_code": exit_code,
            "best_checkpoint_exists": best_ckpt.exists(),
            "epoch_checkpoint_count": epoch_ckpt_count,
            "gpu": gpu_line,
        },
        ensure_ascii=True,
    )
)
PY
EOF
  )"; then
    remote_json='{"remote_state":"ssh_unreachable"}'
  fi

  local summary
  summary="$(jq -cn \
    --arg ts_utc "${ts_utc}" \
    --arg run_id "${RUN_ID}" \
    --arg pod_id "${pod_id}" \
    --arg pod_status "${pod_status}" \
    --arg ssh_host "${SSH_HOST}" \
    --arg ssh_port "${SSH_PORT}" \
    --argjson local_watch_running "${local_watch_running}" \
    --argjson local_full_flow_running "${local_full_flow_running}" \
    --argjson remote "${remote_json}" \
    '{
      ts_utc:$ts_utc,
      run_id:$run_id,
      pod:{id:$pod_id, desired_status:$pod_status, ssh_host:$ssh_host, ssh_port:$ssh_port},
      local:{watch_running:$local_watch_running, full_flow_running:$local_full_flow_running},
      remote:$remote
    }'
  )"
  printf '%s\n' "${summary}"
  if [[ "${WRITE_SNAPSHOT}" == "1" ]]; then
    local out="${REPORT_DIR}/status_snapshot_$(date -u +%Y%m%dT%H%M%SZ).json"
    printf '%s\n' "${summary}" > "${out}"
  fi
}

if [[ "${WATCH_MODE}" == "1" ]]; then
  while true; do
    collect_once
    sleep "${POLL_SECONDS}"
  done
fi

collect_once
