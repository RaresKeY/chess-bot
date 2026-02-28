#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

mkdir -p "${CYCLE_DIR}" "${LOGS_DIR}" "${CYCLE_DIR}/benchmarks"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd ssh
runpod_cycle_require_cmd rsync
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

FLOW_SKIP_START="${RUNPOD_CYCLE_SKIP_START:-0}"
FLOW_STOP_POD="${RUNPOD_BENCH_STOP_POD:-1}"
FLOW_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA A40}"
FLOW_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"
FLOW_TRAIN_NPROC_PER_NODE="${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${FLOW_GPU_COUNT}}"
FLOW_HF_REPO_ID="${RUNPOD_HF_DATASET_REPO_ID:-LogicLark-QuantumQuill/chess-bot-datasets}"
FLOW_HF_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
FLOW_HF_SCHEMA_FILTER="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-game_jsonl_runtime_splice_v1}"
FLOW_EPOCHS="${RUNPOD_BENCH_EPOCHS:-1}"
FLOW_BATCH_SIZE="${RUNPOD_BENCH_BATCH_SIZE:-2048}"
FLOW_NUM_WORKERS="${RUNPOD_BENCH_NUM_WORKERS:-8}"
FLOW_RUNTIME_MAX_SAMPLES_PER_GAME="${RUNPOD_BENCH_RUNTIME_MAX_SAMPLES_PER_GAME:-200000}"
FLOW_MAX_TOTAL_ROWS="${RUNPOD_BENCH_MAX_TOTAL_ROWS:-0}"
FLOW_TRIALS_RAW="${RUNPOD_BENCH_TRIALS:-fp32,tf32,fp16,bf16,sparsity}"

if [[ "${FLOW_SKIP_START}" != "1" ]]; then
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
  RUNPOD_GPU_TYPE_ID="${FLOW_GPU_TYPE_ID}" \
  RUNPOD_GPU_COUNT="${FLOW_GPU_COUNT}" \
  bash "${REPO_ROOT}/scripts/runpod_cycle_start.sh"
fi

if [[ ! -f "${PROVISION_JSON}" ]]; then
  echo "[runpod-bench-matrix] missing provision file: ${PROVISION_JSON}" >&2
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
REMOTE_MATRIX_DIR="${REMOTE_RUN_DIR}/manual_bench"
REMOTE_MANIFEST="${REMOTE_RUN_DIR}/hf_dataset_fetch_manifest.json"
LOCAL_BENCH_ROOT="${CYCLE_DIR}/benchmarks"
LOCAL_SUMMARY_JSONL="${LOCAL_BENCH_ROOT}/trial_summary.jsonl"
LOCAL_SUMMARY_MD="${LOCAL_BENCH_ROOT}/trial_summary.md"

: > "${LOCAL_SUMMARY_JSONL}"
cat > "${LOCAL_SUMMARY_MD}" <<MD
# RunPod Matrix Benchmark Summary

- run_id: \`${RUN_ID}\`
- gpu_type_id: \`${FLOW_GPU_TYPE_ID}\`
- gpu_count: \`${FLOW_GPU_COUNT}\`
- train_nproc_per_node: \`${FLOW_TRAIN_NPROC_PER_NODE}\`
- hf_repo: \`${FLOW_HF_REPO_ID}\`
- hf_prefix: \`${FLOW_HF_PREFIX}\`
- hf_schema_filter: \`${FLOW_HF_SCHEMA_FILTER}\`
- epochs_per_trial: \`${FLOW_EPOCHS}\`
- batch_size: \`${FLOW_BATCH_SIZE}\`
- num_workers_per_rank: \`${FLOW_NUM_WORKERS}\`

| trial | status | exit_code | local_dir |
|---|---:|---:|---|
MD

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "mkdir -p '${REMOTE_RUN_DIR}' '${REMOTE_MATRIX_DIR}'"

IFS=',' read -r -a FLOW_TRIALS <<<"${FLOW_TRIALS_RAW}"

for raw_trial in "${FLOW_TRIALS[@]}"; do
  trial="$(printf '%s' "${raw_trial}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ -n "${trial}" ]] || continue

  trial_status="ok"
  trial_exit=0
  train_extra_args="--epochs ${FLOW_EPOCHS} --early-stopping-patience 0 --progress --verbose"

  case "${trial}" in
    fp32)
      train_extra_args+=" --no-amp --tf32 off"
      ;;
    tf32)
      train_extra_args+=" --no-amp --tf32 on"
      ;;
    fp16)
      train_extra_args+=" --amp --amp-dtype fp16 --tf32 on"
      ;;
    bf16)
      train_extra_args+=" --amp --amp-dtype bf16 --tf32 on"
      ;;
    sparsity)
      trial_status="skipped"
      trial_exit=125
      ;;
    *)
      trial_status="skipped"
      trial_exit=126
      ;;
  esac

  local_trial_dir="${LOCAL_BENCH_ROOT}/${trial}"
  remote_trial_dir="${REMOTE_MATRIX_DIR}/${trial}"
  mkdir -p "${local_trial_dir}"

  if [[ "${trial_status}" == "ok" ]]; then
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
      "TRIAL='${trial}' REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' REMOTE_MANIFEST='${REMOTE_MANIFEST}' REMOTE_TRIAL_DIR='${remote_trial_dir}' FLOW_HF_REPO_ID='${FLOW_HF_REPO_ID}' FLOW_HF_PREFIX='${FLOW_HF_PREFIX}' FLOW_HF_SCHEMA_FILTER='${FLOW_HF_SCHEMA_FILTER}' FLOW_TRAIN_NPROC_PER_NODE='${FLOW_TRAIN_NPROC_PER_NODE}' FLOW_BATCH_SIZE='${FLOW_BATCH_SIZE}' FLOW_NUM_WORKERS='${FLOW_NUM_WORKERS}' FLOW_RUNTIME_MAX_SAMPLES_PER_GAME='${FLOW_RUNTIME_MAX_SAMPLES_PER_GAME}' FLOW_MAX_TOTAL_ROWS='${FLOW_MAX_TOTAL_ROWS}' TRAIN_EXTRA_ARGS='${train_extra_args}' /bin/bash -s" <<'EOF_REMOTE'
set -Eeuo pipefail
mkdir -p "${REMOTE_TRIAL_DIR}"
cd "${REMOTE_REPO_DIR}"

export REPO_DIR="${REMOTE_REPO_DIR}"
export HF_FETCH_LATEST_ALL_DATASETS=1
export HF_USE_EXISTING_FETCH_MANIFEST=1
export HF_DATASET_REPO_ID="${FLOW_HF_REPO_ID}"
export HF_DATASET_PATH_PREFIX="${FLOW_HF_PREFIX}"
export HF_DATASET_SCHEMA_FILTER="${FLOW_HF_SCHEMA_FILTER}"
export HF_DATASET_FETCH_MANIFEST="${REMOTE_MANIFEST}"
export HF_DATASET_CACHE_DIR="${REMOTE_REPO_DIR}/data/hf_datasets"

export OUTPUT_PATH="${REMOTE_TRIAL_DIR}/model_${TRIAL}.pt"
export METRICS_OUT="${REMOTE_TRIAL_DIR}/metrics_${TRIAL}.json"
export TRAIN_PROGRESS_JSONL_OUT="${REMOTE_TRIAL_DIR}/progress_${TRIAL}.jsonl"
export TRAIN_BEST_CHECKPOINT_OUT="${REMOTE_TRIAL_DIR}/model_best_${TRIAL}.pt"
export TRAIN_EPOCH_CHECKPOINT_DIR="${REMOTE_TRIAL_DIR}/epoch_checkpoints"

export TRAIN_NPROC_PER_NODE="${FLOW_TRAIN_NPROC_PER_NODE}"
export TRAIN_BATCH_SIZE="${FLOW_BATCH_SIZE}"
export TRAIN_NUM_WORKERS="${FLOW_NUM_WORKERS}"
export TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME="${FLOW_RUNTIME_MAX_SAMPLES_PER_GAME}"
export TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS} --max-total-rows ${FLOW_MAX_TOTAL_ROWS} --telemetry-dir ${REMOTE_TRIAL_DIR}/telemetry"

log_path="${REMOTE_TRIAL_DIR}/train_stdout_${TRIAL}.log"
rc=0
bash "${REMOTE_REPO_DIR}/deploy/runpod_cloud_training/train_baseline_preset.sh" >"${log_path}" 2>&1 || rc=$?
printf '%s\n' "${rc}" > "${REMOTE_TRIAL_DIR}/train_exit_code.txt"
exit 0
EOF_REMOTE

    trial_exit="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "cat '${remote_trial_dir}/train_exit_code.txt' 2>/dev/null || echo 99")"
    if [[ "${trial_exit}" != "0" ]]; then
      trial_status="failed"
    fi
  fi

  printf '{"trial":"%s","status":"%s","exit_code":%s,"remote_trial_dir":"%s"}\n' \
    "${trial}" "${trial_status}" "${trial_exit}" "${remote_trial_dir}" >> "${LOCAL_SUMMARY_JSONL}"

  rsync -az -e "ssh -i ${SSH_KEY} -p ${SSH_PORT} -o BatchMode=yes -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING} -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
    "${SSH_USER}@${SSH_HOST}:${remote_trial_dir}/" "${local_trial_dir}/" >/dev/null 2>&1 || true

  printf '| %s | %s | %s | `%s` |\n' "${trial}" "${trial_status}" "${trial_exit}" "${local_trial_dir}" >> "${LOCAL_SUMMARY_MD}"
done

RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh" || true

runpod_cycle_append_report "${REPORT_MD}" \
  "## Benchmark Matrix" \
  "- Script: \`scripts/runpod_cycle_benchmark_matrix.sh\`" \
  "- Local summary jsonl: \`${LOCAL_SUMMARY_JSONL}\`" \
  "- Local summary markdown: \`${LOCAL_SUMMARY_MD}\`" \
  "- Local benchmark root: \`${LOCAL_BENCH_ROOT}\`" \
  ""

if [[ "${FLOW_STOP_POD}" == "1" ]]; then
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh"
fi

echo "[runpod-bench-matrix] run_id=${RUN_ID}"
echo "[runpod-bench-matrix] summary=${LOCAL_SUMMARY_MD}"
