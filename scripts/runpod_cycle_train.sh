#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd ssh

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_OPTS=(-i "${SSH_KEY}" -p "${SSH_PORT}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/runpod_known_hosts)

REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_DATASET_NAME="${RUNPOD_REMOTE_DATASET_NAME:-_smoke_runpod_${RUN_ID}}"
REMOTE_DATASET_DIR="${REMOTE_REPO_DIR}/data/dataset/${REMOTE_DATASET_NAME}"
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_MODEL_PATH="${REMOTE_RUN_DIR}/model_${RUN_ID}.pt"
REMOTE_METRICS_PATH="${REMOTE_RUN_DIR}/metrics_${RUN_ID}.json"
REMOTE_INFER_OUT="${REMOTE_RUN_DIR}/remote_infer_smoke.jsonl"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:---epochs 1 --no-progress --no-verbose}"
INFER_CONTEXT="${RUNPOD_INFER_CONTEXT:-e2e4 e7e5 g1f3}"

REMOTE_CMD=$(cat <<EOF
set -Eeuo pipefail
mkdir -p '${REMOTE_RUN_DIR}'
export TRAIN_DATASET_DIR='${REMOTE_DATASET_DIR}'
export OUTPUT_PATH='${REMOTE_MODEL_PATH}'
export METRICS_OUT='${REMOTE_METRICS_PATH}'
export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE}'
export TRAIN_NUM_WORKERS='${TRAIN_NUM_WORKERS}'
export TRAIN_EXTRA_ARGS='${TRAIN_EXTRA_ARGS}'
bash /opt/runpod_cloud_training/train_baseline_preset.sh
cd '${REMOTE_REPO_DIR}'
'/opt/venvs/chessbot/bin/python' scripts/infer_move.py --model '${REMOTE_MODEL_PATH}' --context '${INFER_CONTEXT}' --winner-side W --device cpu > '${REMOTE_INFER_OUT}'
EOF
)

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "${REMOTE_CMD}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Remote Training + Inference Smoke" \
  "- Remote run dir: \`${REMOTE_RUN_DIR}\`" \
  "- Remote model: \`${REMOTE_MODEL_PATH}\`" \
  "- Remote metrics: \`${REMOTE_METRICS_PATH}\`" \
  "- Remote inference smoke output: \`${REMOTE_INFER_OUT}\`" \
  "- SSH endpoint used: \`${SSH_USER}@${SSH_HOST}:${SSH_PORT}\`" \
  "- TRAIN_EXTRA_ARGS: \`${TRAIN_EXTRA_ARGS}\`" \
  ""

echo "[runpod-cycle-train] remote_run_dir=${REMOTE_RUN_DIR}"
echo "[runpod-cycle-train] remote_model=${REMOTE_MODEL_PATH}"
