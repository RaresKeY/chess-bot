#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
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
SSH_OPTS=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}")

REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_DATASET_NAME="${RUNPOD_REMOTE_DATASET_NAME:-_smoke_runpod_${RUN_ID}}"
REMOTE_DATASET_DIR="${REMOTE_REPO_DIR}/data/dataset/${REMOTE_DATASET_NAME}"
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_MODEL_PATH="${REMOTE_RUN_DIR}/model_${RUN_ID}.pt"
REMOTE_METRICS_PATH="${REMOTE_RUN_DIR}/metrics_${RUN_ID}.json"
REMOTE_INFER_OUT="${REMOTE_RUN_DIR}/remote_infer_smoke.jsonl"
REMOTE_HF_FETCH_MANIFEST="${REMOTE_REPO_DIR}/artifacts/hf_dataset_fetch_manifest.json"
USE_HF_LATEST_ALL="${RUNPOD_TRAIN_FROM_HF_LATEST_ALL:-0}"
HF_DATASET_REPO_ID="${RUNPOD_HF_DATASET_REPO_ID:-${HF_DATASET_REPO_ID:-}}"
HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-${HF_DATASET_PATH_PREFIX:-validated_datasets}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:---epochs 1 --no-progress --no-verbose}"
INFER_CONTEXT="${RUNPOD_INFER_CONTEXT:-e2e4 e7e5 g1f3}"

REMOTE_READY_TIMEOUT_SECONDS="${RUNPOD_REMOTE_READY_TIMEOUT_SECONDS:-300}"
REMOTE_READY_POLL_SECONDS="${RUNPOD_REMOTE_READY_POLL_SECONDS:-5}"
remote_ready_deadline=$(( $(date +%s) + REMOTE_READY_TIMEOUT_SECONDS ))
READY_CHECK_LOG="${LOGS_DIR}/train_ready_check.log"
REMOTE_TRAIN_LOG="${LOGS_DIR}/train_remote_ssh.log"

while true; do
  if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
    "test -d '${REMOTE_REPO_DIR}' && test -f '${REMOTE_REPO_DIR}/scripts/infer_move.py' && test -w '${REMOTE_REPO_DIR}'" >/dev/null 2>&1; then
    break
  fi
  if (( $(date +%s) >= remote_ready_deadline )); then
    echo "[runpod-cycle-train] remote repo not ready after ${REMOTE_READY_TIMEOUT_SECONDS}s: ${REMOTE_REPO_DIR}" >&2
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
      "ls -ld '${REMOTE_REPO_DIR}' 2>/dev/null || true; ls -l '${REMOTE_REPO_DIR}/scripts' 2>/dev/null | head || true; id" \
      2>&1 | tee -a "${READY_CHECK_LOG}" || true
    exit 1
  fi
  echo "[runpod-cycle-train] waiting for remote repo readiness: ${REMOTE_REPO_DIR}" >&2
  sleep "${REMOTE_READY_POLL_SECONDS}"
done

REMOTE_CMD=$(cat <<EOF
set -Eeuo pipefail
mkdir -p '${REMOTE_RUN_DIR}'
export REPO_DIR='${REMOTE_REPO_DIR}'
export RUNPOD_PHASE_TIMING_LOG='${REMOTE_REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl'
export OUTPUT_PATH='${REMOTE_MODEL_PATH}'
export METRICS_OUT='${REMOTE_METRICS_PATH}'
export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE}'
export TRAIN_NUM_WORKERS='${TRAIN_NUM_WORKERS}'
export TRAIN_EXTRA_ARGS='${TRAIN_EXTRA_ARGS}'
EOF
)

if [[ "${USE_HF_LATEST_ALL}" == "1" ]]; then
  [[ -n "${HF_DATASET_REPO_ID}" ]] || { echo "[runpod-cycle-train] RUNPOD_TRAIN_FROM_HF_LATEST_ALL=1 requires RUNPOD_HF_DATASET_REPO_ID (or HF_DATASET_REPO_ID)" >&2; exit 1; }
  REMOTE_CMD+=$'\n'"export HF_FETCH_LATEST_ALL_DATASETS='1'"
  REMOTE_CMD+=$'\n'"export HF_DATASET_REPO_ID='${HF_DATASET_REPO_ID}'"
  REMOTE_CMD+=$'\n'"export HF_DATASET_PATH_PREFIX='${HF_DATASET_PATH_PREFIX}'"
  REMOTE_CMD+=$'\n'"export HF_DATASET_FETCH_MANIFEST='${REMOTE_HF_FETCH_MANIFEST}'"
  REMOTE_CMD+=$'\n'"export TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE='1'"
else
  REMOTE_CMD+=$'\n'"export TRAIN_DATASET_DIR='${REMOTE_DATASET_DIR}'"
fi

REMOTE_CMD+=$(cat <<EOF

bash /opt/runpod_cloud_training/train_baseline_preset.sh
cd '${REMOTE_REPO_DIR}'
export PYTHONPATH='${REMOTE_REPO_DIR}'
'/opt/venvs/chessbot/bin/python' scripts/infer_move.py --model '${REMOTE_MODEL_PATH}' --context '${INFER_CONTEXT}' --winner-side W --device cpu > '${REMOTE_INFER_OUT}'
EOF
)

{
  printf '[runpod-cycle-train] ssh=%s@%s:%s\n' "${SSH_USER}" "${SSH_HOST}" "${SSH_PORT}"
  printf '[runpod-cycle-train] remote_repo=%s\n' "${REMOTE_REPO_DIR}"
  printf '[runpod-cycle-train] remote_dataset=%s\n' "${REMOTE_DATASET_DIR}"
  printf '[runpod-cycle-train] remote_run_dir=%s\n' "${REMOTE_RUN_DIR}"
  printf '[runpod-cycle-train] use_hf_latest_all=%s\n' "${USE_HF_LATEST_ALL}"
  printf '[runpod-cycle-train] hf_dataset_repo_id=%s\n' "${HF_DATASET_REPO_ID}"
  printf '[runpod-cycle-train] hf_dataset_path_prefix=%s\n' "${HF_DATASET_PATH_PREFIX}"
  printf '[runpod-cycle-train] train_extra_args=%s\n' "${TRAIN_EXTRA_ARGS}"
} > "${REMOTE_TRAIN_LOG}"

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "${REMOTE_CMD}" 2>&1 | tee -a "${REMOTE_TRAIN_LOG}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Remote Training + Inference Smoke" \
  "- Remote run dir: \`${REMOTE_RUN_DIR}\`" \
  "- Remote model: \`${REMOTE_MODEL_PATH}\`" \
  "- Remote metrics: \`${REMOTE_METRICS_PATH}\`" \
  "- Remote inference smoke output: \`${REMOTE_INFER_OUT}\`" \
  "- SSH endpoint used: \`${SSH_USER}@${SSH_HOST}:${SSH_PORT}\`" \
  "- TRAIN_EXTRA_ARGS: \`${TRAIN_EXTRA_ARGS}\`" \
  "- Train data source mode: \`$( [[ "${USE_HF_LATEST_ALL}" == "1" ]] && printf 'hf_latest_all' || printf 'pushed_dataset' )\`" \
  "- HF dataset repo (when enabled): \`${HF_DATASET_REPO_ID}\`" \
  "- Remote HF fetch manifest (when enabled): \`${REMOTE_HF_FETCH_MANIFEST}\`" \
  "- Ready-check log: \`${READY_CHECK_LOG}\`" \
  "- Remote SSH/train log: \`${REMOTE_TRAIN_LOG}\`" \
  ""

echo "[runpod-cycle-train] remote_run_dir=${REMOTE_RUN_DIR}"
echo "[runpod-cycle-train] remote_model=${REMOTE_MODEL_PATH}"
