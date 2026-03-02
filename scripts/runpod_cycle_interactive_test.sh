#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUNPOD_TEST_SESSION_ID_RAW="${RUNPOD_TEST_SESSION_ID:-default}"
RUNPOD_TEST_SESSION_ID="$(printf '%s' "${RUNPOD_TEST_SESSION_ID_RAW}" | tr -cs 'A-Za-z0-9._-' '-')"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-runpod-interactive-${RUNPOD_TEST_SESSION_ID}}"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
STATE_DIR="${CYCLE_DIR}/interactive"
STATE_JSON="${STATE_DIR}/latest_manual_train.json"
mkdir -p "${STATE_DIR}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/runpod_cycle_interactive_test.sh <command>

Commands:
  up            Provision or resume the interactive test pod/session.
  ssh           Open an interactive shell on the current session pod.
  train-start   Interrupt existing training (default), then launch a new background train.
  train-stop    Stop the last tracked interactive training run.
  train-status  Show latest interactive training + GPU status.
  watch         Attach progress watcher to the last tracked interactive training run.
  status        Show one-shot cycle status snapshot.
  down          Stop pod compute (no terminate).
  terminate     Terminate pod resource (delete pod).

Core defaults:
  RUNPOD_TEST_SESSION_ID=default
  RUNPOD_CYCLE_RUN_ID=runpod-interactive-<session_id>
  RUNPOD_INTERRUPTIBLE=1
  RUNPOD_CLOUD_TYPE=SECURE
  RUNPOD_TERMINATE_ON_SSH_NOT_READY=0

Interactive train controls:
  RUNPOD_INTERACTIVE_TRAIN_ID                 (default: manual_<utc-ts>)
  RUNPOD_INTERACTIVE_STOP_EXISTING            (default: 1)
  RUNPOD_INTERACTIVE_FETCH_POLICY             (if_missing|always|never, default: if_missing)
  RUNPOD_HF_DATASET_REPO_ID                   (default: LogicLark-QuantumQuill/chess-bot-datasets)
  RUNPOD_HF_DATASET_PATH_PREFIX               (default: validated_datasets)
  RUNPOD_HF_DATASET_SCHEMA_FILTER             (default: game_jsonl_runtime_splice_v1)
  RUNPOD_INTERACTIVE_TRAIN_EPOCHS             (default: 1)
  RUNPOD_INTERACTIVE_TRAIN_BATCH_SIZE_OVERRIDE (default: unset -> preset auto)
  RUNPOD_INTERACTIVE_TRAIN_NUM_WORKERS_OVERRIDE (default: unset -> preset auto)
  RUNPOD_INTERACTIVE_TRAIN_MAX_TOTAL_ROWS     (default: 0)
  RUNPOD_INTERACTIVE_TRAIN_ROLLOUT_HORIZON    (default: 8)
  RUNPOD_INTERACTIVE_TRAIN_CLOSENESS_HORIZON  (default: rollout horizon)
  RUNPOD_INTERACTIVE_TRAIN_NPROC_PER_NODE     (default: RUNPOD_FULL_TRAIN_NPROC_PER_NODE or RUNPOD_GPU_COUNT or 1)
  RUNPOD_INTERACTIVE_REQUIRE_RUNTIME_SPLICE_CACHE (default: 1)
USAGE
}

die() {
  echo "[runpod-interactive-test] $*" >&2
  exit 1
}

require_provision() {
  [[ -f "${PROVISION_JSON}" ]] || die "missing provision file: ${PROVISION_JSON}; run '... up' first"
}

load_ssh_details() {
  require_provision
  runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"
  SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
  SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
  SSH_KEY="$(runpod_cycle_ssh_key)"
  SSH_USER="$(runpod_cycle_ssh_user)"
  SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
  SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
  SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"
  SSH_OPTS=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}")
  SCP_OPTS=(-i "${SSH_KEY}" -P "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}")
  REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
  REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
}

compute_manual_paths() {
  local train_id="$1"
  REMOTE_MANUAL_DIR="${REMOTE_RUN_DIR}/${train_id}"
  REMOTE_PROGRESS_JSONL="${REMOTE_MANUAL_DIR}/train_progress_${train_id}.jsonl"
  REMOTE_TRAIN_LOG="${REMOTE_MANUAL_DIR}/train_stdout_${train_id}.log"
  REMOTE_TRAIN_EXIT_CODE_FILE="${REMOTE_MANUAL_DIR}/train_exit_code.txt"
  REMOTE_TRAIN_PID_FILE="${REMOTE_MANUAL_DIR}/train_pid.txt"
  REMOTE_MODEL_OUT="${REMOTE_MANUAL_DIR}/model_${train_id}.pt"
  REMOTE_METRICS_OUT="${REMOTE_MANUAL_DIR}/metrics_${train_id}.json"
  REMOTE_BEST_CHECKPOINT="${REMOTE_MANUAL_DIR}/model_best_${train_id}.pt"
  REMOTE_EPOCH_CHECKPOINT_DIR="${REMOTE_MANUAL_DIR}/epoch_checkpoints"
}

write_state() {
  local train_id="$1"
  jq -nc \
    --arg session_id "${RUNPOD_TEST_SESSION_ID}" \
    --arg run_id "${RUN_ID}" \
    --arg train_id "${train_id}" \
    --arg remote_repo_dir "${REMOTE_REPO_DIR}" \
    --arg remote_run_dir "${REMOTE_RUN_DIR}" \
    --arg remote_manual_dir "${REMOTE_MANUAL_DIR}" \
    --arg remote_progress_jsonl "${REMOTE_PROGRESS_JSONL}" \
    --arg remote_train_log "${REMOTE_TRAIN_LOG}" \
    --arg remote_train_exit_code_file "${REMOTE_TRAIN_EXIT_CODE_FILE}" \
    --arg remote_train_pid_file "${REMOTE_TRAIN_PID_FILE}" \
    --arg remote_best_checkpoint "${REMOTE_BEST_CHECKPOINT}" \
    --arg remote_epoch_checkpoint_dir "${REMOTE_EPOCH_CHECKPOINT_DIR}" \
    --arg updated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
      session_id:$session_id,
      run_id:$run_id,
      train_id:$train_id,
      remote_repo_dir:$remote_repo_dir,
      remote_run_dir:$remote_run_dir,
      remote_manual_dir:$remote_manual_dir,
      remote_progress_jsonl:$remote_progress_jsonl,
      remote_train_log:$remote_train_log,
      remote_train_exit_code_file:$remote_train_exit_code_file,
      remote_train_pid_file:$remote_train_pid_file,
      remote_best_checkpoint:$remote_best_checkpoint,
      remote_epoch_checkpoint_dir:$remote_epoch_checkpoint_dir,
      updated_at_utc:$updated_at_utc
    }' > "${STATE_JSON}"
}

load_state_or_fail() {
  [[ -f "${STATE_JSON}" ]] || die "missing interactive state file: ${STATE_JSON}; run 'train-start' first"
  TRAIN_ID="$(jq -r '.train_id // empty' "${STATE_JSON}")"
  [[ -n "${TRAIN_ID}" ]] || die "state file missing train_id: ${STATE_JSON}"
  REMOTE_REPO_DIR="$(jq -r '.remote_repo_dir // empty' "${STATE_JSON}")"
  REMOTE_RUN_DIR="$(jq -r '.remote_run_dir // empty' "${STATE_JSON}")"
  REMOTE_MANUAL_DIR="$(jq -r '.remote_manual_dir // empty' "${STATE_JSON}")"
  REMOTE_PROGRESS_JSONL="$(jq -r '.remote_progress_jsonl // empty' "${STATE_JSON}")"
  REMOTE_TRAIN_LOG="$(jq -r '.remote_train_log // empty' "${STATE_JSON}")"
  REMOTE_TRAIN_EXIT_CODE_FILE="$(jq -r '.remote_train_exit_code_file // empty' "${STATE_JSON}")"
  REMOTE_TRAIN_PID_FILE="$(jq -r '.remote_train_pid_file // empty' "${STATE_JSON}")"
  REMOTE_BEST_CHECKPOINT="$(jq -r '.remote_best_checkpoint // empty' "${STATE_JSON}")"
  REMOTE_EPOCH_CHECKPOINT_DIR="$(jq -r '.remote_epoch_checkpoint_dir // empty' "${STATE_JSON}")"
}

cmd_up() {
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
  RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-chess-bot-interactive-${RUNPOD_TEST_SESSION_ID}}" \
  RUNPOD_INTERRUPTIBLE="${RUNPOD_INTERRUPTIBLE:-1}" \
  RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}" \
  RUNPOD_RESUME_STOPPED_POD="${RUNPOD_RESUME_STOPPED_POD:-1}" \
  RUNPOD_TERMINATE_ON_SSH_NOT_READY="${RUNPOD_TERMINATE_ON_SSH_NOT_READY:-0}" \
  bash "${REPO_ROOT}/scripts/runpod_cycle_start.sh"
}

cmd_ssh() {
  load_ssh_details
  echo "[runpod-interactive-test] ssh ${SSH_USER}@${SSH_HOST}:${SSH_PORT} (run_id=${RUN_ID})"
  ssh -tt "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}"
}

cmd_train_start() {
  load_ssh_details

  local train_id stop_existing fetch_policy
  train_id="${RUNPOD_INTERACTIVE_TRAIN_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
  train_id="$(printf '%s' "${train_id}" | tr -cs 'A-Za-z0-9._-' '_')"
  [[ -n "${train_id}" ]] || die "resolved interactive train_id is empty"
  stop_existing="${RUNPOD_INTERACTIVE_STOP_EXISTING:-1}"
  fetch_policy="${RUNPOD_INTERACTIVE_FETCH_POLICY:-if_missing}"
  case "${fetch_policy}" in
    if_missing|always|never) ;;
    *) die "invalid RUNPOD_INTERACTIVE_FETCH_POLICY='${fetch_policy}' (expected if_missing|always|never)" ;;
  esac

  compute_manual_paths "${train_id}"

  local hf_repo_id hf_path_prefix hf_schema
  local epochs max_total_rows rollout_horizon closeness_horizon
  local batch_override workers_override nproc require_cache
  hf_repo_id="${RUNPOD_HF_DATASET_REPO_ID:-LogicLark-QuantumQuill/chess-bot-datasets}"
  hf_path_prefix="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
  hf_schema="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-game_jsonl_runtime_splice_v1}"
  epochs="${RUNPOD_INTERACTIVE_TRAIN_EPOCHS:-1}"
  max_total_rows="${RUNPOD_INTERACTIVE_TRAIN_MAX_TOTAL_ROWS:-0}"
  rollout_horizon="${RUNPOD_INTERACTIVE_TRAIN_ROLLOUT_HORIZON:-8}"
  closeness_horizon="${RUNPOD_INTERACTIVE_TRAIN_CLOSENESS_HORIZON:-${rollout_horizon}}"
  batch_override="${RUNPOD_INTERACTIVE_TRAIN_BATCH_SIZE_OVERRIDE:-}"
  workers_override="${RUNPOD_INTERACTIVE_TRAIN_NUM_WORKERS_OVERRIDE:-}"
  nproc="${RUNPOD_INTERACTIVE_TRAIN_NPROC_PER_NODE:-${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${RUNPOD_GPU_COUNT:-1}}}"
  require_cache="${RUNPOD_INTERACTIVE_REQUIRE_RUNTIME_SPLICE_CACHE:-1}"

  [[ "${epochs}" =~ ^[0-9]+$ ]] || die "RUNPOD_INTERACTIVE_TRAIN_EPOCHS must be an integer"
  [[ "${max_total_rows}" =~ ^[0-9]+$ ]] || die "RUNPOD_INTERACTIVE_TRAIN_MAX_TOTAL_ROWS must be an integer"
  [[ "${rollout_horizon}" =~ ^[0-9]+$ ]] || die "RUNPOD_INTERACTIVE_TRAIN_ROLLOUT_HORIZON must be an integer"
  [[ "${closeness_horizon}" =~ ^[0-9]+$ ]] || die "RUNPOD_INTERACTIVE_TRAIN_CLOSENESS_HORIZON must be an integer"
  [[ "${nproc}" =~ ^[0-9]+$ ]] || die "RUNPOD_INTERACTIVE_TRAIN_NPROC_PER_NODE must be an integer"
  [[ "${nproc}" != "0" ]] || die "RUNPOD_INTERACTIVE_TRAIN_NPROC_PER_NODE must be >= 1"
  if [[ -n "${batch_override}" && ! "${batch_override}" =~ ^[0-9]+$ ]]; then
    die "RUNPOD_INTERACTIVE_TRAIN_BATCH_SIZE_OVERRIDE must be an integer when set"
  fi
  if [[ -n "${workers_override}" && ! "${workers_override}" =~ ^[0-9]+$ ]]; then
    die "RUNPOD_INTERACTIVE_TRAIN_NUM_WORKERS_OVERRIDE must be an integer when set"
  fi

  echo "[runpod-interactive-test] launching train_id=${train_id} run_id=${RUN_ID}"
  echo "[runpod-interactive-test] remote_repo_dir=${REMOTE_REPO_DIR}"
  echo "[runpod-interactive-test] remote_manual_dir=${REMOTE_MANUAL_DIR}"

  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "bash -s" <<EOF
set -Eeuo pipefail
mkdir -p '${REMOTE_RUN_DIR}' '${REMOTE_MANUAL_DIR}' '${REMOTE_EPOCH_CHECKPOINT_DIR}'

stop_existing='${stop_existing}'
if [[ "\${stop_existing}" == "1" ]]; then
  if [[ -f '${REMOTE_RUN_DIR}/active_train_pid.txt' ]]; then
    old_pid="\$(cat '${REMOTE_RUN_DIR}/active_train_pid.txt' || true)"
    if [[ -n "\${old_pid}" ]] && kill -0 "\${old_pid}" >/dev/null 2>&1; then
      kill -TERM "\${old_pid}" >/dev/null 2>&1 || true
      sleep 2
      kill -KILL "\${old_pid}" >/dev/null 2>&1 || true
    fi
  fi
  pkill -f '${REMOTE_REPO_DIR}/scripts/train_baseline.py' >/dev/null 2>&1 || true
  pkill -f '/opt/venvs/chessbot/bin/torchrun' >/dev/null 2>&1 || true
  pkill -f '${REMOTE_REPO_DIR}/deploy/runpod_cloud_training/train_baseline_preset.sh' >/dev/null 2>&1 || true
fi

manifest_path='${REMOTE_RUN_DIR}/hf_dataset_fetch_manifest.json'
fetch_policy='${fetch_policy}'
need_fetch=0
case "\${fetch_policy}" in
  always) need_fetch=1 ;;
  if_missing) [[ -s "\${manifest_path}" ]] || need_fetch=1 ;;
  never) need_fetch=0 ;;
esac

if (( need_fetch == 1 )); then
  export PYTHONPATH='${REMOTE_REPO_DIR}'
  '/opt/venvs/chessbot/bin/python' '${REMOTE_REPO_DIR}/scripts/hf_dataset_fetch.py' \\
    --repo-id '${hf_repo_id}' \\
    --repo-path-prefix '${hf_path_prefix}' \\
    --all-latest \\
    --dest-dir '${REMOTE_REPO_DIR}/data/hf_datasets' \\
    --output-manifest "\${manifest_path}"
fi

export REPO_DIR='${REMOTE_REPO_DIR}'
export HF_FETCH_LATEST_ALL_DATASETS=1
export HF_DATASET_FETCH_MANIFEST="\${manifest_path}"
export HF_DATASET_SCHEMA_FILTER='${hf_schema}'
export OUTPUT_PATH='${REMOTE_MODEL_OUT}'
export METRICS_OUT='${REMOTE_METRICS_OUT}'
export TRAIN_PROGRESS_JSONL_OUT='${REMOTE_PROGRESS_JSONL}'
export TRAIN_BEST_CHECKPOINT_OUT='${REMOTE_BEST_CHECKPOINT}'
export TRAIN_EPOCH_CHECKPOINT_DIR='${REMOTE_EPOCH_CHECKPOINT_DIR}'
export TRAIN_NPROC_PER_NODE='${nproc}'
export TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE='${require_cache}'
export TRAIN_MAX_TOTAL_ROWS='${max_total_rows}'
export TRAIN_ROLLOUT_HORIZON='${rollout_horizon}'
export TRAIN_CLOSENESS_HORIZON='${closeness_horizon}'
export TRAIN_EXTRA_ARGS="--epochs ${epochs} --early-stopping-patience 0 --rollout-horizon ${rollout_horizon} --closeness-horizon ${closeness_horizon}"
if [[ '${max_total_rows}' != "0" ]]; then
  export TRAIN_EXTRA_ARGS="\${TRAIN_EXTRA_ARGS} --max-total-rows ${max_total_rows}"
fi

if [[ -n '${batch_override}' ]]; then
  export TRAIN_BATCH_SIZE='${batch_override}'
fi
if [[ -n '${workers_override}' ]]; then
  export TRAIN_NUM_WORKERS='${workers_override}'
fi

if (( ${nproc} > 1 )); then
  : "\${NCCL_IB_DISABLE:=1}"
  : "\${NCCL_P2P_DISABLE:=1}"
  : "\${NCCL_P2P_LEVEL:=LOC}"
  : "\${TORCH_NCCL_ASYNC_ERROR_HANDLING:=1}"
  : "\${TORCH_NCCL_ENABLE_MONITORING:=1}"
  : "\${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:=1800}"
  : "\${TORCH_NCCL_BLOCKING_WAIT:=1}"
  : "\${TORCH_NCCL_DUMP_ON_TIMEOUT:=1}"
  : "\${TORCH_NCCL_TRACE_BUFFER_SIZE:=2000}"
  : "\${NCCL_DEBUG:=WARN}"
  export NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_P2P_LEVEL TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_ENABLE_MONITORING TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_DUMP_ON_TIMEOUT TORCH_NCCL_TRACE_BUFFER_SIZE NCCL_DEBUG
fi

train_preset_image='/opt/runpod_cloud_training/train_baseline_preset.sh'
train_preset_repo='${REMOTE_REPO_DIR}/deploy/runpod_cloud_training/train_baseline_preset.sh'
train_preset_script="\${train_preset_image}"
if [[ -f "\${train_preset_repo}" ]]; then
  train_preset_script="\${train_preset_repo}"
fi
export train_preset_script

cat > '${REMOTE_MANUAL_DIR}/launch_env_summary.txt' <<'TXT'
run_id=${RUN_ID}
train_id=${train_id}
remote_repo_dir=${REMOTE_REPO_DIR}
remote_manual_dir=${REMOTE_MANUAL_DIR}
manifest_path=\${manifest_path}
fetch_policy=${fetch_policy}
epochs=${epochs}
batch_override=${batch_override:-<unset>}
workers_override=${workers_override:-<unset>}
nproc=${nproc}
max_total_rows=${max_total_rows}
rollout_horizon=${rollout_horizon}
closeness_horizon=${closeness_horizon}
require_runtime_splice_cache=${require_cache}
train_preset_script=\${train_preset_script}
TXT

cat > '${REMOTE_MANUAL_DIR}/train_background_launcher.sh' <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
rc=0
bash "\${train_preset_script}" >> '${REMOTE_TRAIN_LOG}' 2>&1 || rc=$?
printf '%s\n' "\${rc}" > '${REMOTE_TRAIN_EXIT_CODE_FILE}'
exit "\${rc}"
SH
chmod +x '${REMOTE_MANUAL_DIR}/train_background_launcher.sh'

nohup bash '${REMOTE_MANUAL_DIR}/train_background_launcher.sh' >/dev/null 2>&1 &
bg_pid=\$!
printf '%s\n' "\${bg_pid}" > '${REMOTE_TRAIN_PID_FILE}'
printf '%s\n' "\${bg_pid}" > '${REMOTE_RUN_DIR}/active_train_pid.txt'

echo "[runpod-interactive-test-remote] launched_pid=\${bg_pid}"
echo "[runpod-interactive-test-remote] progress_jsonl=${REMOTE_PROGRESS_JSONL}"
echo "[runpod-interactive-test-remote] exit_code_file=${REMOTE_TRAIN_EXIT_CODE_FILE}"
echo "[runpod-interactive-test-remote] train_log=${REMOTE_TRAIN_LOG}"
EOF

  write_state "${train_id}"
  echo "[runpod-interactive-test] state_json=${STATE_JSON}"
}

cmd_train_stop() {
  load_ssh_details
  load_state_or_fail
  echo "[runpod-interactive-test] stopping train_id=${TRAIN_ID}"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "bash -s" <<EOF
set -Eeuo pipefail
pid_file='${REMOTE_TRAIN_PID_FILE}'
exit_file='${REMOTE_TRAIN_EXIT_CODE_FILE}'
if [[ -f "\${pid_file}" ]]; then
  pid="\$(cat "\${pid_file}" || true)"
  if [[ -n "\${pid}" ]] && kill -0 "\${pid}" >/dev/null 2>&1; then
    kill -TERM "\${pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL "\${pid}" >/dev/null 2>&1 || true
  fi
fi
pkill -f '${REMOTE_REPO_DIR}/scripts/train_baseline.py' >/dev/null 2>&1 || true
pkill -f '/opt/venvs/chessbot/bin/torchrun' >/dev/null 2>&1 || true
pkill -f '${REMOTE_REPO_DIR}/deploy/runpod_cloud_training/train_baseline_preset.sh' >/dev/null 2>&1 || true
printf '%s\n' "130" > "\${exit_file}"
echo "[runpod-interactive-test-remote] stopped train_id=${TRAIN_ID}"
EOF
}

cmd_train_status() {
  load_ssh_details
  load_state_or_fail
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "bash -s" <<EOF
set -Eeuo pipefail
echo "timestamp_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "train_id=${TRAIN_ID}"
echo "train_pid_file=${REMOTE_TRAIN_PID_FILE}"
if [[ -f '${REMOTE_TRAIN_PID_FILE}' ]]; then
  train_pid="\$(cat '${REMOTE_TRAIN_PID_FILE}' || true)"
else
  train_pid=""
fi
echo "train_pid=\${train_pid:-<none>}"
if [[ -n "\${train_pid}" ]] && kill -0 "\${train_pid}" >/dev/null 2>&1; then
  ps -o pid,ppid,stat,etime,pcpu,pmem,cmd -p "\${train_pid}" || true
else
  echo "train_pid_alive=0"
fi
echo "exit_code_file=${REMOTE_TRAIN_EXIT_CODE_FILE}"
cat '${REMOTE_TRAIN_EXIT_CODE_FILE}' 2>/dev/null | sed 's/^/exit_code=/' || true
echo "--- gpu ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
echo "--- progress tail ---"
tail -n 8 '${REMOTE_PROGRESS_JSONL}' 2>/dev/null || echo "no_progress_jsonl"
echo "--- train log tail ---"
tail -n 20 '${REMOTE_TRAIN_LOG}' 2>/dev/null || echo "no_train_log"
EOF
}

cmd_watch() {
  load_ssh_details
  load_state_or_fail
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
  RUNPOD_REMOTE_REPO_DIR="${REMOTE_REPO_DIR}" \
  RUNPOD_REMOTE_PROGRESS_JSONL="${REMOTE_PROGRESS_JSONL}" \
  RUNPOD_REMOTE_TRAIN_EXIT_CODE_FILE="${REMOTE_TRAIN_EXIT_CODE_FILE}" \
  RUNPOD_REMOTE_TRAIN_LOG="${REMOTE_TRAIN_LOG}" \
  RUNPOD_REMOTE_BEST_CHECKPOINT="${REMOTE_BEST_CHECKPOINT}" \
  RUNPOD_REMOTE_EPOCH_CHECKPOINT_DIR="${REMOTE_EPOCH_CHECKPOINT_DIR}" \
  bash "${REPO_ROOT}/scripts/runpod_cycle_watch_progress.sh"
}

cmd_status() {
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_status.sh"
}

cmd_down() {
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh"
}

cmd_terminate() {
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh"
}

main() {
  local cmd="${1:-help}"
  case "${cmd}" in
    up) cmd_up ;;
    ssh) cmd_ssh ;;
    train-start) cmd_train_start ;;
    train-stop) cmd_train_stop ;;
    train-status) cmd_train_status ;;
    watch) cmd_watch ;;
    status) cmd_status ;;
    down) cmd_down ;;
    terminate) cmd_terminate ;;
    help|-h|--help) usage ;;
    *)
      usage
      die "unknown command: ${cmd}"
      ;;
  esac
}

main "${1:-help}"
