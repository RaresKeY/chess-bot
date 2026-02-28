#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry/telemetry_common.sh"

AS_JSON=0
RUN_ID_OVERRIDE=""

while (($#)); do
  case "$1" in
    --json) AS_JSON=1; shift ;;
    --run-id) RUN_ID_OVERRIDE="${2:-}"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/telemetry_status.sh [--run-id <id>] [--json]"
      exit 0
      ;;
    *) echo "[telemetry-status] unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(telemetry_repo_root)"
if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  RUN_ID="${RUN_ID_OVERRIDE}"
else
  RUN_ID="$(telemetry_run_id)"
fi

RUN_DIR="${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}"
TELEMETRY_DIR="${RUN_DIR}/telemetry"
EVENTS_FILE="${TELEMETRY_DIR}/events.jsonl"
CHECKPOINTS_FILE="${TELEMETRY_DIR}/checkpoints.jsonl"

remote_status_json="$(RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_status.sh" --no-write 2>/dev/null || echo '{}')"
latest_event='{}'
latest_checkpoint='{}'

if [[ -f "${EVENTS_FILE}" ]]; then
  latest_event="$(tail -n 1 "${EVENTS_FILE}" 2>/dev/null || echo '{}')"
fi
if [[ -f "${CHECKPOINTS_FILE}" ]]; then
  latest_checkpoint="$(tail -n 1 "${CHECKPOINTS_FILE}" 2>/dev/null || echo '{}')"
fi

summary="$(jq -cn \
  --arg run_id "${RUN_ID}" \
  --arg run_dir "${RUN_DIR}" \
  --arg events_file "${EVENTS_FILE}" \
  --arg checkpoints_file "${CHECKPOINTS_FILE}" \
  --argjson remote "${remote_status_json}" \
  --argjson latest_event "${latest_event}" \
  --argjson latest_checkpoint "${latest_checkpoint}" \
  '{
    run_id:$run_id,
    run_dir:$run_dir,
    files:{events:$events_file,checkpoints:$checkpoints_file},
    remote:$remote,
    latest_event:$latest_event,
    latest_checkpoint:$latest_checkpoint
  }')"

if [[ "${AS_JSON}" == "1" ]]; then
  printf '%s\n' "${summary}"
  exit 0
fi

remote_state="$(printf '%s\n' "${summary}" | jq -r '.remote.remote.remote_state // .remote.remote_state // "unknown"')"
progress_lines="$(printf '%s\n' "${summary}" | jq -r '.remote.remote.progress_lines // .remote.progress_lines // 0')"
train_log_lines="$(printf '%s\n' "${summary}" | jq -r '.remote.remote.train_log_lines // .remote.train_log_lines // 0')"
last_event_name="$(printf '%s\n' "${summary}" | jq -r '.latest_event.event // "n/a"')"
last_event_status="$(printf '%s\n' "${summary}" | jq -r '.latest_event.status // "n/a"')"
last_checkpoint="$(printf '%s\n' "${summary}" | jq -r '.latest_checkpoint.checkpoint // "n/a"')"
last_checkpoint_state="$(printf '%s\n' "${summary}" | jq -r '.latest_checkpoint.state // "n/a"')"

cat <<TXT
[telemetry-status] run_id=${RUN_ID}
[telemetry-status] remote_state=${remote_state} progress_lines=${progress_lines} train_log_lines=${train_log_lines}
[telemetry-status] latest_event=${last_event_name} status=${last_event_status}
[telemetry-status] latest_checkpoint=${last_checkpoint} state=${last_checkpoint_state}
[telemetry-status] events_file=${EVENTS_FILE}
[telemetry-status] checkpoints_file=${CHECKPOINTS_FILE}
TXT
