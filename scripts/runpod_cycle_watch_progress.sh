#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${LOGS_DIR}"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd ssh
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"
SSH_FORCE_TTY="${RUNPOD_SSH_FORCE_TTY:-0}"
SSH_TTY_ARGS=()
if [[ "${SSH_FORCE_TTY}" == "1" ]]; then
  SSH_TTY_ARGS=(-tt)
fi
SSH_OPTS=("${SSH_TTY_ARGS[@]}" -i "${SSH_KEY}" -p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}")

REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_PROGRESS_JSONL="${RUNPOD_REMOTE_PROGRESS_JSONL:-${REMOTE_RUN_DIR}/train_progress_${RUN_ID}.jsonl}"
REMOTE_EXIT_CODE_FILE="${RUNPOD_REMOTE_TRAIN_EXIT_CODE_FILE:-${REMOTE_RUN_DIR}/train_exit_code.txt}"
REMOTE_TRAIN_LOG="${RUNPOD_REMOTE_TRAIN_LOG:-${REMOTE_RUN_DIR}/train_stdout_${RUN_ID}.log}"
POLL_SECONDS="${RUNPOD_PROGRESS_POLL_SECONDS:-5}"
BAR_WIDTH="${RUNPOD_PROGRESS_BAR_WIDTH:-32}"
WATCH_LOG="${LOGS_DIR}/train_progress_watch.log"
WATCH_DEBUG="${RUNPOD_WATCH_DEBUG:-0}"
TTY_STATE_ORIG="$(stty -g 2>/dev/null || true)"

cleanup_child_processes() {
  local pids=()
  mapfile -t pids < <(jobs -pr 2>/dev/null || true)
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" >/dev/null 2>&1 || true
    wait "${pids[@]}" >/dev/null 2>&1 || true
  fi
}

restore_tty() {
  if [[ -n "${TTY_STATE_ORIG}" ]]; then
    stty "${TTY_STATE_ORIG}" >/dev/null 2>&1 || stty sane >/dev/null 2>&1 || true
  else
    stty sane >/dev/null 2>&1 || true
  fi
}

handle_interrupt() {
  printf '\n' >&2
  echo "[runpod-cycle-watch] interrupted" >&2
  cleanup_child_processes
  restore_tty
  exit 130
}

trap handle_interrupt INT TERM
trap restore_tty EXIT

watch_debug() {
  [[ "${WATCH_DEBUG}" == "1" ]] || return 0
  local msg="$1"
  printf '%s\n' "[runpod-cycle-watch-debug] ${msg}" >> "${WATCH_LOG}"
  printf '%s\n' "[runpod-cycle-watch-debug] ${msg}" >&2
}

last_progress_line=""
last_exit_code=""
started_epoch=""
completed_epoch=0
total_epochs=0
last_event=""
last_metrics=""
remote_stream_ended_without_exit=0
status_source="none"
warned_stdout_fallback=0
last_meta_line=""

render_bar() {
  local done="$1"
  local total="$2"
  local width="$3"
  local filled=0
  if (( total > 0 )); then
    filled=$(( done * width / total ))
  fi
  (( filled < 0 )) && filled=0
  (( filled > width )) && filled="${width}"
  local empty=$(( width - filled ))
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '-'
}

print_status() {
  local pct="0"
  if (( total_epochs > 0 )); then
    pct=$(( completed_epoch * 100 / total_epochs ))
  fi
  local bar
  bar="$(render_bar "${completed_epoch}" "${total_epochs:-1}" "${BAR_WIDTH}")"
  local line
  line="[runpod-cycle-watch] [${bar}] ${pct}% epoch=${completed_epoch}/${total_epochs:-0}"
  if [[ -n "${started_epoch}" && "${started_epoch}" != "${completed_epoch}" ]]; then
    line+=" (running ${started_epoch})"
  fi
  if [[ -n "${last_event}" ]]; then
    line+=" event=${last_event}"
  fi
  if [[ -n "${last_metrics}" ]]; then
    line+=" ${last_metrics}"
  fi
  if [[ -n "${status_source}" && "${status_source}" != "none" ]]; then
    line+=" src=${status_source}"
  fi
  printf '\r%s\033[K' "${line}"
}

latest_progress_event_line() {
  local blob="$1"
  local line=""
  local candidate=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -n "${line//[[:space:]]/}" ]] || continue
    if printf '%s\n' "${line}" | jq -e 'type=="object" and (.event? != null)' >/dev/null 2>&1; then
      candidate="${line}"
    fi
  done <<< "${blob}"
  printf '%s\n' "${candidate}"
}

latest_stdout_epoch_line() {
  local blob="$1"
  local line=""
  local candidate=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -n "${line//[[:space:]]/}" ]] || continue
    if [[ "${line}" =~ ^\[train\]\ epoch\ ([0-9]+)/([0-9]+)\ start$ ]]; then
      candidate="${line}"
      continue
    fi
    if [[ "${line}" =~ ^\{\'epoch\':\ ([0-9]+), ]]; then
      candidate="${line}"
      continue
    fi
    if [[ "${line}" == *"'train_setup': {"* || "${line}" == *"'train_start': {"* ]]; then
      candidate="${line}"
      continue
    fi
  done <<< "${blob}"
  printf '%s\n' "${candidate}"
}

latest_stdout_epoch_start_line() {
  local blob="$1"
  local line=""
  local candidate=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -n "${line//[[:space:]]/}" ]] || continue
    if [[ "${line}" =~ ^\[train\]\ epoch\ ([0-9]+)/([0-9]+)\ start$ ]]; then
      candidate="${line}"
    fi
  done <<< "${blob}"
  printf '%s\n' "${candidate}"
}

parse_stdout_metrics() {
  local line="$1"
  local train_loss=""
  local val_loss=""
  local top1=""
  if [[ "${line}" =~ \'train_loss\':\ ([0-9.]+) ]]; then
    train_loss="${BASH_REMATCH[1]}"
  fi
  if [[ "${line}" =~ \'val_loss\':\ ([0-9.]+) ]]; then
    val_loss="${BASH_REMATCH[1]}"
  fi
  if [[ "${line}" =~ \'top1\':\ ([0-9.]+) ]]; then
    top1="${BASH_REMATCH[1]}"
  fi
  local out=""
  [[ -n "${train_loss}" ]] && out+="train_loss=${train_loss} "
  [[ -n "${val_loss}" ]] && out+="val_loss=${val_loss} "
  [[ -n "${top1}" ]] && out+="top1=${top1}"
  printf '%s\n' "${out%" "}"
}

count_progress_event_rows() {
  local blob="$1"
  local line=""
  local n=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -n "${line//[[:space:]]/}" ]] || continue
    if printf '%s\n' "${line}" | jq -e 'type=="object" and (.event? != null)' >/dev/null 2>&1; then
      n=$((n + 1))
    fi
  done <<< "${blob}"
  printf '%s\n' "${n}"
}

process_snapshot() {
  local snapshot="$1"
  local progress_blob exit_blob log_blob meta_blob split_marker split_log_marker split_meta_marker progress_line stdout_line ec
  snapshot="${snapshot//$'\r'/}"
  split_marker='__RUNPOD_PROGRESS_SPLIT__'
  split_log_marker='__RUNPOD_PROGRESS_LOG_SPLIT__'
  split_meta_marker='__RUNPOD_PROGRESS_META_SPLIT__'
  if [[ "${snapshot}" == *"${split_marker}"* ]]; then
    progress_blob="${snapshot%%${split_marker}*}"
    exit_blob="${snapshot#*"${split_marker}"}"
  else
    progress_blob="${snapshot}"
    exit_blob=""
  fi
  if [[ "${exit_blob}" == *"${split_log_marker}"* ]]; then
    log_blob="${exit_blob#*"${split_log_marker}"}"
    exit_blob="${exit_blob%%${split_log_marker}*}"
  else
    log_blob=""
  fi
  if [[ "${log_blob}" == *"${split_meta_marker}"* ]]; then
    meta_blob="${log_blob#*"${split_meta_marker}"}"
    log_blob="${log_blob%%${split_meta_marker}*}"
  else
    meta_blob=""
  fi
  progress_blob="${progress_blob%$'\n'}"
  exit_blob="${exit_blob#$'\n'}"
  exit_blob="${exit_blob//$'\r'/}"
  exit_blob="$(printf '%s' "${exit_blob}" | tr -d '[:space:]')"
  log_blob="${log_blob#$'\n'}"
  log_blob="${log_blob%$'\n'}"
  meta_blob="${meta_blob#$'\n'}"
  meta_blob="${meta_blob%$'\n'}"

  if [[ -n "${meta_blob}" && "${meta_blob}" != "${last_meta_line}" ]]; then
    last_meta_line="${meta_blob}"
    watch_debug "remote_meta=${meta_blob}"
  fi

  if [[ -n "${progress_blob//[[:space:]]/}" ]]; then
    progress_line="$(latest_progress_event_line "${progress_blob}")"
    watch_debug "jsonl_tail_nonempty=1 json_event_rows=$(count_progress_event_rows "${progress_blob}") selected_event_line=${progress_line:-<none>}"
    if [[ "${progress_line}" != "${last_progress_line}" ]]; then
      last_progress_line="${progress_line}"
      last_event="$(printf '%s\n' "${progress_line}" | jq -r '.event // ""' 2>/dev/null || true)"
      case "${last_event}" in
        script_start)
          total_epochs="$(printf '%s\n' "${progress_line}" | jq -r '.epochs_requested // 0' 2>/dev/null || echo 0)"
          ;;
        train_setup)
          if (( total_epochs <= 0 )); then
            total_epochs="$(printf '%s\n' "${progress_line}" | jq -r '.epochs // 0' 2>/dev/null || echo 0)"
          fi
          ;;
        epoch_start)
          started_epoch="$(printf '%s\n' "${progress_line}" | jq -r '.epoch // ""' 2>/dev/null || true)"
          if (( total_epochs <= 0 )); then
            total_epochs="$(printf '%s\n' "${progress_line}" | jq -r '.epochs // 0' 2>/dev/null || echo 0)"
          fi
          last_metrics=""
          ;;
        epoch_end)
          completed_epoch="$(printf '%s\n' "${progress_line}" | jq -r '.epoch // 0' 2>/dev/null || echo 0)"
          started_epoch=""
          if (( total_epochs <= 0 )); then
            total_epochs="$(printf '%s\n' "${progress_line}" | jq -r '.epochs // 0' 2>/dev/null || echo 0)"
          fi
          last_metrics="$(
            printf '%s\n' "${progress_line}" | jq -r '
              .metrics // {} |
              "train_loss=" + ((.train_loss // 0) | tostring) +
              " val_loss=" + ((.val_loss // 0) | tostring) +
              " top1=" + ((.top1 // 0) | tostring)
            ' 2>/dev/null || true
          )"
          ;;
        train_complete|script_complete)
          ec="$(printf '%s\n' "${progress_line}" | jq -r '.epochs_completed // .history_last_epoch // empty' 2>/dev/null || true)"
          [[ -n "${ec}" ]] && completed_epoch="${ec}"
          ;;
      esac
      status_source="jsonl"
      printf '%s\n' "${progress_line}" >> "${WATCH_LOG}"
    fi
  else
    watch_debug "jsonl_tail_nonempty=0 selected_event_line=<none>"
  fi

  if [[ -z "${last_progress_line}" || "${status_source}" != "jsonl" || ( ${total_epochs:-0} -le 0 && ${completed_epoch:-0} -eq 0 ) ]]; then
    local stdout_start_line=""
    stdout_start_line="$(latest_stdout_epoch_start_line "${log_blob}")"
    if [[ "${stdout_start_line}" =~ ^\[train\]\ epoch\ ([0-9]+)/([0-9]+)\ start$ ]]; then
      if (( total_epochs <= 0 )); then
        total_epochs="${BASH_REMATCH[2]}"
      fi
    elif [[ "${log_blob}" =~ \'epochs\'\:\ ([0-9]+) ]]; then
      if (( total_epochs <= 0 )); then
        total_epochs="${BASH_REMATCH[1]}"
      fi
    fi
    stdout_line="$(latest_stdout_epoch_line "${log_blob}")"
    if [[ -n "${stdout_line}" ]]; then
      if [[ "${stdout_line}" =~ ^\[train\]\ epoch\ ([0-9]+)/([0-9]+)\ start$ ]]; then
        started_epoch="${BASH_REMATCH[1]}"
        total_epochs="${BASH_REMATCH[2]}"
        last_event="epoch_start"
        status_source="stdout"
      elif [[ "${stdout_line}" =~ ^\{\'epoch\':\ ([0-9]+), ]]; then
        completed_epoch="${BASH_REMATCH[1]}"
        started_epoch=""
        last_event="epoch_end"
        status_source="stdout"
        last_metrics="$(parse_stdout_metrics "${stdout_line}")"
      elif [[ "${stdout_line}" =~ \'epochs\'\:\ ([0-9]+) ]]; then
        if (( total_epochs <= 0 )); then
          total_epochs="${BASH_REMATCH[1]}"
        fi
        status_source="stdout"
      fi
      if [[ "${warned_stdout_fallback}" != "1" && "${status_source}" == "stdout" ]]; then
        printf '\n[runpod-cycle-watch] using stdout fallback progress parser (progress JSONL missing/unreadable)\n' >&2
        warned_stdout_fallback=1
      fi
    fi
  fi

  print_status

  if [[ -n "${exit_blob}" ]]; then
    last_exit_code="${exit_blob}"
  fi
}

BLOCK_BEGIN='__RUNPOD_PROGRESS_BLOCK_BEGIN__'
BLOCK_SPLIT='__RUNPOD_PROGRESS_SPLIT__'
BLOCK_LOG_SPLIT='__RUNPOD_PROGRESS_LOG_SPLIT__'
BLOCK_META_SPLIT='__RUNPOD_PROGRESS_META_SPLIT__'
BLOCK_END='__RUNPOD_PROGRESS_BLOCK_END__'
block_mode="idle"
snapshot_progress=""
snapshot_exit=""
snapshot_log=""
snapshot_meta=""
snapshot_line=""

while IFS= read -r snapshot_line || [[ -n "${snapshot_line}" ]]; do
  case "${snapshot_line}" in
    "${BLOCK_BEGIN}")
      block_mode="progress"
      snapshot_progress=""
      snapshot_exit=""
      snapshot_log=""
      snapshot_meta=""
      ;;
    "${BLOCK_SPLIT}")
      block_mode="exit"
      ;;
    "${BLOCK_LOG_SPLIT}")
      block_mode="log"
      ;;
    "${BLOCK_META_SPLIT}")
      block_mode="meta"
      ;;
    "${BLOCK_END}")
      process_snapshot "${snapshot_progress}"$'\n'"${BLOCK_SPLIT}"$'\n'"${snapshot_exit}"$'\n'"${BLOCK_LOG_SPLIT}"$'\n'"${snapshot_log}"$'\n'"${BLOCK_META_SPLIT}"$'\n'"${snapshot_meta}"
      if [[ -n "${last_exit_code}" ]]; then
        break
      fi
      block_mode="idle"
      ;;
    *)
      if [[ "${block_mode}" == "progress" ]]; then
        snapshot_progress+="${snapshot_line}"$'\n'
      elif [[ "${block_mode}" == "exit" ]]; then
        snapshot_exit+="${snapshot_line}"$'\n'
      elif [[ "${block_mode}" == "log" ]]; then
        snapshot_log+="${snapshot_line}"$'\n'
      elif [[ "${block_mode}" == "meta" ]]; then
        snapshot_meta+="${snapshot_line}"$'\n'
      fi
      ;;
  esac
done < <(
  watch_debug "start run_id=${RUN_ID} ssh=${SSH_USER}@${SSH_HOST}:${SSH_PORT} progress_jsonl=${REMOTE_PROGRESS_JSONL} exit_code_file=${REMOTE_EXIT_CODE_FILE} train_log=${REMOTE_TRAIN_LOG} poll_seconds=${POLL_SECONDS}"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
    "RUNPOD_PROGRESS_POLL_SECONDS='${POLL_SECONDS}' RUNPOD_REMOTE_PROGRESS_JSONL='${REMOTE_PROGRESS_JSONL}' RUNPOD_REMOTE_EXIT_CODE_FILE='${REMOTE_EXIT_CODE_FILE}' RUNPOD_REMOTE_TRAIN_LOG='${REMOTE_TRAIN_LOG}' bash -s" <<'EOF' 2>/dev/null
set -Eeuo pipefail
poll_s="${RUNPOD_PROGRESS_POLL_SECONDS:-5}"
progress_file="${RUNPOD_REMOTE_PROGRESS_JSONL}"
exit_file="${RUNPOD_REMOTE_EXIT_CODE_FILE}"
train_log="${RUNPOD_REMOTE_TRAIN_LOG:-}"
while true; do
  printf '%s\n' '__RUNPOD_PROGRESS_BLOCK_BEGIN__'
  if [[ -f "${progress_file}" ]]; then
    tail -n 200 "${progress_file}" || true
  fi
  printf '%s\n' '__RUNPOD_PROGRESS_SPLIT__'
  if [[ -f "${exit_file}" ]]; then
    cat "${exit_file}" || true
  fi
  printf '%s\n' '__RUNPOD_PROGRESS_LOG_SPLIT__'
  if [[ -n "${train_log}" && -f "${train_log}" ]]; then
    tail -n 120 "${train_log}" || true
  fi
  printf '%s\n' '__RUNPOD_PROGRESS_META_SPLIT__'
  progress_exists=0
  exit_exists=0
  train_log_exists=0
  [[ -f "${progress_file}" ]] && progress_exists=1
  [[ -f "${exit_file}" ]] && exit_exists=1
  [[ -n "${train_log}" && -f "${train_log}" ]] && train_log_exists=1
  printf '{"progress_file":"%s","progress_exists":%s,"exit_file":"%s","exit_exists":%s,"train_log":"%s","train_log_exists":%s}\n' \
    "${progress_file}" "${progress_exists}" "${exit_file}" "${exit_exists}" "${train_log}" "${train_log_exists}"
  printf '%s\n' '__RUNPOD_PROGRESS_BLOCK_END__'
  if [[ -f "${exit_file}" ]]; then
    break
  fi
  sleep "${poll_s}"
done
EOF
)

if [[ -z "${last_exit_code}" ]]; then
  remote_stream_ended_without_exit=1
fi

printf '\n'
if [[ "${last_exit_code}" =~ ^[0-9]+$ ]]; then
  echo "[runpod-cycle-watch] remote_train_exit_code=${last_exit_code}"
  exit "${last_exit_code}"
fi
if [[ "${remote_stream_ended_without_exit}" == "1" ]]; then
  echo "[runpod-cycle-watch] ssh_stream_ended_before_exit_sentinel (training may still be running; check remote train log/progress jsonl)" >&2
fi
echo "[runpod-cycle-watch] remote_train_exit_code_unparseable=${last_exit_code}"
exit 1
