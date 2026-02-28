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

telemetry_event() {
  local ev="$1"
  local st="$2"
  local msg="${3:-}"
  local extra_json="${4:-{}}"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/telemetry_emit_event.sh" \
    --event "${ev}" --status "${st}" --message "${msg}" --extra-json "${extra_json}" >/dev/null 2>&1 || true
}

telemetry_checkpoint() {
  local name="$1"
  local state="$2"
  local note="${3:-}"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/telemetry_checkpoint.sh" \
    --name "${name}" --state "${state}" --note "${note}" >/dev/null 2>&1 || true
}

telemetry_healthcheck() {
  local kind="$1"
  local msg="${2:-}"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/telemetry_healthchecks_ping.sh" \
    "${kind}" "${msg}" >/dev/null 2>&1 || true
}

telemetry_event "benchmark_matrix_start" "info" "benchmark matrix started"
telemetry_checkpoint "benchmark_matrix" "running" "matrix flow started"
telemetry_healthcheck start "run_id=${RUN_ID} benchmark_matrix_start"

FLOW_SKIP_START="${RUNPOD_CYCLE_SKIP_START:-0}"
FLOW_STOP_POD="${RUNPOD_BENCH_STOP_POD:-1}"
FLOW_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA A40}"
FLOW_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"
FLOW_TRAIN_NPROC_PER_NODE="${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${FLOW_GPU_COUNT}}"
FLOW_HF_REPO_ID="${RUNPOD_HF_DATASET_REPO_ID:-LogicLark-QuantumQuill/chess-bot-datasets}"
FLOW_HF_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
FLOW_HF_DATASET_NAME="${RUNPOD_BENCH_HF_DATASET_NAME:-}"
FLOW_HF_DATASET_VERSION="${RUNPOD_BENCH_HF_DATASET_VERSION:-}"
FLOW_HF_SCHEMA_FILTER="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-game_jsonl_runtime_splice_v1}"
FLOW_EPOCHS="${RUNPOD_BENCH_EPOCHS:-1}"
FLOW_BATCH_SIZE_RAW="${RUNPOD_BENCH_BATCH_SIZE:-auto}"
FLOW_NUM_WORKERS="${RUNPOD_BENCH_NUM_WORKERS:-8}"
FLOW_DISTRIBUTED_BACKEND="${RUNPOD_BENCH_DISTRIBUTED_BACKEND:-nccl}"
FLOW_RUNTIME_MAX_SAMPLES_PER_GAME="${RUNPOD_BENCH_RUNTIME_MAX_SAMPLES_PER_GAME:-200000}"
FLOW_MAX_TOTAL_ROWS="${RUNPOD_BENCH_MAX_TOTAL_ROWS:-0}"
FLOW_TRIALS_RAW="${RUNPOD_BENCH_TRIALS:-fp32,tf32,fp16,bf16,sparsity}"
FLOW_SPARSITY_L1_LAMBDA="${RUNPOD_BENCH_SPARSITY_L1_LAMBDA:-1e-6}"
FLOW_SPARSITY_L1_LAMBDAS_RAW="${RUNPOD_BENCH_SPARSITY_L1_LAMBDAS:-}"
FLOW_TERMINATE_POD="${RUNPOD_BENCH_TERMINATE_POD:-0}"
FLOW_TRANSFER_TOOL="${RUNPOD_BENCH_TRANSFER_TOOL:-rclone}"
FLOW_TRANSFER_STRICT="${RUNPOD_BENCH_TRANSFER_STRICT:-0}"
FLOW_TRANSFER_RETRIES="${RUNPOD_BENCH_TRANSFER_RETRIES:-3}"
FLOW_TRANSFER_TIMEOUT_SECONDS="${RUNPOD_BENCH_TRANSFER_TIMEOUT_SECONDS:-1800}"
FLOW_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS="${RUNPOD_BENCH_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS:-0}"
FLOW_RCLONE_TRANSFERS="${RUNPOD_BENCH_RCLONE_TRANSFERS:-8}"
FLOW_RCLONE_CHECKERS="${RUNPOD_BENCH_RCLONE_CHECKERS:-16}"
FLOW_RCLONE_MULTI_THREAD_STREAMS="${RUNPOD_BENCH_RCLONE_MULTI_THREAD_STREAMS:-4}"
FLOW_COLLECT_INCLUDE_EPOCH_CHECKPOINTS="${RUNPOD_BENCH_COLLECT_INCLUDE_EPOCH_CHECKPOINTS:-0}"
FLOW_SKIP_FINAL_COLLECT="${RUNPOD_BENCH_SKIP_FINAL_COLLECT:-0}"
FLOW_EXPECTED_GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

FLOW_BATCH_SIZE_OVERRIDE=""
if [[ "${FLOW_BATCH_SIZE_RAW}" =~ ^[0-9]+$ ]] && (( FLOW_BATCH_SIZE_RAW > 0 )); then
  FLOW_BATCH_SIZE_OVERRIDE="${FLOW_BATCH_SIZE_RAW}"
fi
FLOW_BATCH_SIZE_RESOLVED="${FLOW_BATCH_SIZE_OVERRIDE:-2048}"
FLOW_BATCH_ATTEMPTS=("${FLOW_BATCH_SIZE_RESOLVED}")

build_batch_attempt_plan() {
  local base="$1"
  local -a raw=()
  if (( base >= 8192 )); then
    raw=(8192 6144 4096 3072 2048 1536 1024)
  elif (( base >= 4096 )); then
    raw=(4096 3072 2048 1536 1024 768 512)
  elif (( base >= 2048 )); then
    raw=(2048 1536 1024 768 512)
  elif (( base >= 1024 )); then
    raw=(1024 768 512)
  else
    raw=("${base}")
  fi
  declare -A seen=()
  FLOW_BATCH_ATTEMPTS=()
  for b in "${raw[@]}"; do
    if [[ "${b}" =~ ^[0-9]+$ ]] && (( b > 0 )) && [[ -z "${seen[${b}]:-}" ]]; then
      seen["${b}"]=1
      FLOW_BATCH_ATTEMPTS+=("${b}")
    fi
  done
  if (( ${#FLOW_BATCH_ATTEMPTS[@]} == 0 )); then
    FLOW_BATCH_ATTEMPTS=("${base}")
  fi
}

if [[ "${FLOW_SKIP_START}" != "1" ]]; then
  telemetry_checkpoint "benchmark_provision" "running" "starting benchmark pod"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
  RUNPOD_GPU_TYPE_ID="${FLOW_GPU_TYPE_ID}" \
  RUNPOD_GPU_COUNT="${FLOW_GPU_COUNT}" \
  bash "${REPO_ROOT}/scripts/runpod_cycle_start.sh"
fi
telemetry_checkpoint "benchmark_provision" "done" "benchmark pod ready"

if [[ ! -f "${PROVISION_JSON}" ]]; then
  echo "[runpod-bench-matrix] missing provision file: ${PROVISION_JSON}" >&2
  exit 1
fi

IMAGE_USED="$(jq -r '(.pod_status.imageName // .create_response.imageName // "")' "${PROVISION_JSON}")"
if [[ -n "${IMAGE_USED}" ]]; then
  telemetry_event "benchmark_image_used" "info" "image resolved from provision record" "{\"image\":\"${IMAGE_USED}\"}"
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
- hf_dataset_name: \`${FLOW_HF_DATASET_NAME:-<all-latest>}\`
- hf_dataset_version: \`${FLOW_HF_DATASET_VERSION:-<latest>}\`
- hf_schema_filter: \`${FLOW_HF_SCHEMA_FILTER}\`
- expected_git_sha: \`${FLOW_EXPECTED_GIT_SHA}\`
- epochs_per_trial: \`${FLOW_EPOCHS}\`
- batch_size_request: \`${FLOW_BATCH_SIZE_RAW}\`
- batch_size_resolved: \`${FLOW_BATCH_SIZE_RESOLVED}\`
- batch_attempt_plan: \`${FLOW_BATCH_ATTEMPTS[*]}\`
- num_workers_per_rank: \`${FLOW_NUM_WORKERS}\`
- distributed_backend: \`${FLOW_DISTRIBUTED_BACKEND}\`
- transfer_tool: \`${FLOW_TRANSFER_TOOL}\`
- transfer_strict: \`${FLOW_TRANSFER_STRICT}\`
- transfer_retries: \`${FLOW_TRANSFER_RETRIES}\`
- transfer_timeout_seconds: \`${FLOW_TRANSFER_TIMEOUT_SECONDS}\`
- transfer_include_epoch_checkpoints: \`${FLOW_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS}\`
- final_collect_include_epoch_checkpoints: \`${FLOW_COLLECT_INCLUDE_EPOCH_CHECKPOINTS}\`
- skip_final_collect: \`${FLOW_SKIP_FINAL_COLLECT}\`
- sparsity_l1_lambda_default: \`${FLOW_SPARSITY_L1_LAMBDA}\`
- sparsity_l1_lambdas: \`${FLOW_SPARSITY_L1_LAMBDAS_RAW:-<default-only>}\`
- image_used: \`${IMAGE_USED:-<unknown>}\`

| trial | status | exit_code | local_dir |
|---|---:|---:|---|
MD

extract_trial_metrics_json() {
  local trial_dir="$1"
  python3 - "$trial_dir" <<'PY'
import json
import sys
from pathlib import Path

trial_dir = Path(sys.argv[1])
out = {
    "duration_seconds": None,
    "rows_per_second": None,
    "epochs_completed": None,
    "train_rows": None,
    "last_train_loss": None,
    "last_val_loss": None,
    "last_top1": None,
    "last_top5": None,
}

transfer_trial_artifacts() {
  local trial="$1"
  local remote_trial_dir="$2"
  local local_trial_dir="$3"
  local attempt=1
  while (( attempt <= FLOW_TRANSFER_RETRIES )); do
    if [[ "${FLOW_TRANSFER_TOOL}" == "rclone" ]]; then
      if command -v rclone >/dev/null 2>&1; then
        local rclone_src rclone_cmd
        rclone_src=":sftp,host=${SSH_HOST},user=${SSH_USER},port=${SSH_PORT},key_file=${SSH_KEY},known_hosts_file=${SSH_KNOWN_HOSTS_FILE}:${remote_trial_dir}"
        rclone_cmd=(
          rclone copy --create-empty-src-dirs
          --transfers "${FLOW_RCLONE_TRANSFERS}"
          --checkers "${FLOW_RCLONE_CHECKERS}"
          --multi-thread-streams "${FLOW_RCLONE_MULTI_THREAD_STREAMS}"
          --fast-list
          "${rclone_src}" "${local_trial_dir}"
          --include "metrics_*.json"
          --include "progress_*.jsonl"
          --include "train_stdout_*.log"
          --include "train_exit_code.txt"
          --include "model_*.pt"
          --include "model_best_*.pt"
          --include "telemetry/**"
        )
        if [[ "${FLOW_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS}" == "1" ]]; then
          rclone_cmd+=( --include "epoch_checkpoints/**" )
        fi
        if timeout "${FLOW_TRANSFER_TIMEOUT_SECONDS}" "${rclone_cmd[@]}" >/dev/null 2>&1; then
          telemetry_event "benchmark_transfer" "ok" "trial transfer complete" "{\"trial\":\"${trial}\",\"tool\":\"rclone\",\"attempt\":${attempt}}"
          return 0
        fi
        telemetry_event "benchmark_transfer" "warn" "rclone_copy_failed_retry" "{\"trial\":\"${trial}\",\"attempt\":${attempt}}"
      elif [[ "${FLOW_TRANSFER_STRICT}" == "1" ]]; then
        echo "[runpod-bench-matrix] transfer tool rclone requested but command is missing on host" >&2
        return 1
      else
        telemetry_event "benchmark_transfer" "warn" "rclone_missing_fallback_rsync" "{\"trial\":\"${trial}\",\"attempt\":${attempt}}"
      fi
    fi

    local rsync_cmd
    rsync_cmd=(
      rsync -az
      --include "metrics_*.json"
      --include "progress_*.jsonl"
      --include "train_stdout_*.log"
      --include "train_exit_code.txt"
      --include "model_*.pt"
      --include "model_best_*.pt"
      --include "telemetry/***"
    )
    if [[ "${FLOW_TRANSFER_INCLUDE_EPOCH_CHECKPOINTS}" == "1" ]]; then
      rsync_cmd+=( --include "epoch_checkpoints/***" )
    fi
    rsync_cmd+=( --exclude "*" )
    rsync_cmd+=( -e "ssh -i ${SSH_KEY} -p ${SSH_PORT} -o BatchMode=yes -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING} -o UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" )
    rsync_cmd+=( "${SSH_USER}@${SSH_HOST}:${remote_trial_dir}/" "${local_trial_dir}/" )

    if timeout "${FLOW_TRANSFER_TIMEOUT_SECONDS}" "${rsync_cmd[@]}" >/dev/null 2>&1; then
      telemetry_event "benchmark_transfer" "ok" "trial transfer complete" "{\"trial\":\"${trial}\",\"tool\":\"rsync\",\"attempt\":${attempt}}"
      return 0
    fi
    telemetry_event "benchmark_transfer" "warn" "rsync_copy_failed_retry" "{\"trial\":\"${trial}\",\"attempt\":${attempt}}"
    attempt=$((attempt + 1))
  done
  telemetry_event "benchmark_transfer" "error" "trial transfer failed after retries" "{\"trial\":\"${trial}\",\"retries\":${FLOW_TRANSFER_RETRIES}}"
  return 1
}
metrics_files = sorted(trial_dir.glob("metrics_*.json"))
progress_files = sorted(trial_dir.glob("progress_*.jsonl"))
if metrics_files:
    try:
        m = json.loads(metrics_files[-1].read_text(encoding="utf-8"))
        out["train_rows"] = int(m.get("train_rows", 0) or 0)
        history = list(m.get("history", []) or [])
        out["epochs_completed"] = int(len(history))
        if history:
            last = history[-1]
            out["last_train_loss"] = float(last.get("train_loss", 0.0))
            out["last_val_loss"] = float(last.get("val_loss", 0.0))
            out["last_top1"] = float(last.get("top1", 0.0))
            out["last_top5"] = float(last.get("top5", 0.0))
    except Exception:
        pass
if progress_files:
    first_ts = None
    last_ts = None
    try:
        with progress_files[-1].open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("ts_epoch_ms")
                if isinstance(ts, int):
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
    except Exception:
        pass
    if isinstance(first_ts, int) and isinstance(last_ts, int) and last_ts >= first_ts:
        dur = (last_ts - first_ts) / 1000.0
        out["duration_seconds"] = float(dur)
        tr = out["train_rows"]
        ep = out["epochs_completed"]
        if isinstance(tr, int) and tr > 0 and isinstance(ep, int) and ep > 0 and dur > 0:
            out["rows_per_second"] = float((tr * ep) / dur)
print(json.dumps(out, ensure_ascii=True))
PY
}

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "mkdir -p '${REMOTE_RUN_DIR}' '${REMOTE_MATRIX_DIR}'"

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "FLOW_EXPECTED_GIT_SHA='${FLOW_EXPECTED_GIT_SHA}' REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' /bin/bash -s" <<'EOF_REMOTE_SYNC'
set -Eeuo pipefail
cd "${REMOTE_REPO_DIR}"
if [[ ! -d .git ]]; then
  echo "[runpod-bench-matrix] remote repo missing git metadata: ${REMOTE_REPO_DIR}" >&2
  exit 1
fi
git fetch origin main >/dev/null 2>&1
git checkout main >/dev/null 2>&1
git pull --ff-only origin main >/dev/null 2>&1
remote_sha="$(git rev-parse HEAD)"
if [[ "${remote_sha}" != "${FLOW_EXPECTED_GIT_SHA}" ]]; then
  echo "[runpod-bench-matrix] remote sha mismatch after sync: ${remote_sha} expected ${FLOW_EXPECTED_GIT_SHA}" >&2
  exit 1
fi
echo "[runpod-bench-matrix] remote_repo_synced sha=${remote_sha}"
EOF_REMOTE_SYNC

if [[ -z "${FLOW_BATCH_SIZE_OVERRIDE}" ]]; then
  remote_vram_mib="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | awk '{print int(\$1)}'" \
    2>/dev/null || true)"
  if [[ "${remote_vram_mib}" =~ ^[0-9]+$ ]]; then
    if (( remote_vram_mib >= 70000 )); then
      FLOW_BATCH_SIZE_RESOLVED=8192
    elif (( remote_vram_mib >= 44000 )); then
      FLOW_BATCH_SIZE_RESOLVED=4096
    elif (( remote_vram_mib >= 22000 )); then
      FLOW_BATCH_SIZE_RESOLVED=2048
    elif (( remote_vram_mib >= 15000 )); then
      FLOW_BATCH_SIZE_RESOLVED=1024
    else
      FLOW_BATCH_SIZE_RESOLVED=512
    fi
  fi
fi
build_batch_attempt_plan "${FLOW_BATCH_SIZE_RESOLVED}"
telemetry_event "benchmark_batch_plan" "info" "resolved benchmark batch plan" "{\"batch_size_request\":\"${FLOW_BATCH_SIZE_RAW}\",\"batch_size_resolved\":${FLOW_BATCH_SIZE_RESOLVED},\"batch_attempt_plan\":\"${FLOW_BATCH_ATTEMPTS[*]}\"}"
{
  echo
  echo "## Runtime Batch Plan"
  echo
  echo "- batch_size_request: \`${FLOW_BATCH_SIZE_RAW}\`"
  echo "- batch_size_resolved: \`${FLOW_BATCH_SIZE_RESOLVED}\`"
  echo "- batch_attempt_plan: \`${FLOW_BATCH_ATTEMPTS[*]}\`"
  echo
} >> "${LOCAL_SUMMARY_MD}"

telemetry_checkpoint "benchmark_dependencies" "running" "checking pod dependency freshness"
REMOTE_DEP_CHECK_JSON="${REMOTE_RUN_DIR}/dependency_check.json"
LOCAL_DEP_CHECK_JSON="${CYCLE_DIR}/reports/dependency_check.json"
mkdir -p "$(dirname "${LOCAL_DEP_CHECK_JSON}")"
if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' REMOTE_DEP_CHECK_JSON='${REMOTE_DEP_CHECK_JSON}' /bin/bash -s" <<'EOF_REMOTE_DEPS'
set -Eeuo pipefail
"/opt/venvs/chessbot/bin/python" - <<'PY' "${REMOTE_REPO_DIR}" "${REMOTE_DEP_CHECK_JSON}"
import json
import re
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
out_path = Path(sys.argv[2])
req_path = repo / "requirements.txt"
rows = []
missing = 0
mismatch = 0
checked = 0
if req_path.is_file():
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(>=|==)\s*([A-Za-z0-9_.-]+)$", line)
        if not m:
            continue
        name, op, need = m.group(1), m.group(2), m.group(3)
        checked += 1
        proc = subprocess.run(
            ["/opt/venvs/chessbot/bin/python", "-m", "pip", "show", name],
            capture_output=True,
            text=True,
            check=False,
        )
        got = ""
        if proc.returncode == 0:
            for pline in proc.stdout.splitlines():
                if pline.startswith("Version:"):
                    got = pline.split(":", 1)[1].strip()
                    break
        status = "ok"
        if not got:
            status = "missing"
            missing += 1
        else:
            try:
                from packaging.version import Version

                lhs = Version(got)
                rhs = Version(need)
                if op == ">=":
                    status = "ok" if lhs >= rhs else "mismatch"
                elif op == "==":
                    status = "ok" if lhs == rhs else "mismatch"
            except Exception:
                if op == "==" and got != need:
                    status = "mismatch"
                elif op == ">=" and got < need:
                    status = "mismatch"
        if status == "mismatch":
            mismatch += 1
        rows.append({"requirement": line, "name": name, "operator": op, "required": need, "installed": got, "status": status})

payload = {
    "requirements_path": str(req_path),
    "checked": int(checked),
    "missing": int(missing),
    "mismatch": int(mismatch),
    "up_to_date": bool(missing == 0 and mismatch == 0),
    "rows": rows,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=True))
PY
EOF_REMOTE_DEPS
then
  scp -i "${SSH_KEY}" -P "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
    "${SSH_USER}@${SSH_HOST}:${REMOTE_DEP_CHECK_JSON}" "${LOCAL_DEP_CHECK_JSON}" >/dev/null 2>&1 || true
  dep_missing="$(jq -r '.missing // 0' "${LOCAL_DEP_CHECK_JSON}" 2>/dev/null || echo 0)"
  dep_mismatch="$(jq -r '.mismatch // 0' "${LOCAL_DEP_CHECK_JSON}" 2>/dev/null || echo 0)"
  dep_up_to_date="$(jq -r '.up_to_date // false' "${LOCAL_DEP_CHECK_JSON}" 2>/dev/null || echo false)"
  if [[ "${dep_up_to_date}" == "true" ]]; then
    telemetry_checkpoint "benchmark_dependencies" "done" "dependencies up to date"
    telemetry_event "benchmark_dependencies" "ok" "dependency check complete" "{\"missing\":${dep_missing},\"mismatch\":${dep_mismatch},\"up_to_date\":true}"
  else
    telemetry_checkpoint "benchmark_dependencies" "error" "dependency mismatch detected"
    telemetry_event "benchmark_dependencies" "warn" "dependency check complete with issues" "{\"missing\":${dep_missing},\"mismatch\":${dep_mismatch},\"up_to_date\":false}"
  fi
else
  telemetry_checkpoint "benchmark_dependencies" "error" "dependency check failed"
  telemetry_event "benchmark_dependencies" "error" "dependency check command failed"
fi

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "FLOW_HF_REPO_ID='${FLOW_HF_REPO_ID}' FLOW_HF_PREFIX='${FLOW_HF_PREFIX}' FLOW_HF_DATASET_NAME='${FLOW_HF_DATASET_NAME}' FLOW_HF_DATASET_VERSION='${FLOW_HF_DATASET_VERSION}' REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' REMOTE_MANIFEST='${REMOTE_MANIFEST}' /bin/bash -s" <<'EOF_REMOTE_FETCH'
set -Eeuo pipefail
mkdir -p "$(dirname "${REMOTE_MANIFEST}")"
export HF_READ_TOKEN="${HF_READ_TOKEN:-}"
export HF_WRITE_TOKEN=""
export HF_TOKEN=""
fetch_args=(
  --repo-id "${FLOW_HF_REPO_ID}"
  --repo-path-prefix "${FLOW_HF_PREFIX}"
  --dest-dir "${REMOTE_REPO_DIR}/data/hf_datasets"
  --output-manifest "${REMOTE_MANIFEST}"
)
if [[ -n "${FLOW_HF_DATASET_NAME}" ]]; then
  fetch_args+=( --dataset-name "${FLOW_HF_DATASET_NAME}" )
  if [[ -n "${FLOW_HF_DATASET_VERSION}" ]]; then
    fetch_args+=( --version "${FLOW_HF_DATASET_VERSION}" )
  fi
else
  fetch_args+=( --all-latest )
fi
"/opt/venvs/chessbot/bin/python" "${REMOTE_REPO_DIR}/scripts/hf_dataset_fetch.py" "${fetch_args[@]}"
EOF_REMOTE_FETCH

IFS=',' read -r -a FLOW_TRIALS_BASE <<<"${FLOW_TRIALS_RAW}"
IFS=',' read -r -a FLOW_SPARSITY_LAMBDAS <<<"${FLOW_SPARSITY_L1_LAMBDAS_RAW}"
FLOW_TRIALS=()
for raw_trial in "${FLOW_TRIALS_BASE[@]}"; do
  trial_base="$(printf '%s' "${raw_trial}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ -n "${trial_base}" ]] || continue
  if [[ "${trial_base}" == "fp32_sparse" || "${trial_base}" == "fp16_sparse" || "${trial_base}" == "bf16_sparse" || "${trial_base}" == "sparsity" ]]; then
    if [[ -n "${FLOW_SPARSITY_L1_LAMBDAS_RAW}" ]]; then
      for lam in "${FLOW_SPARSITY_LAMBDAS[@]}"; do
        lam_clean="$(printf '%s' "${lam}" | tr -d '[:space:]')"
        [[ -n "${lam_clean}" ]] || continue
        FLOW_TRIALS+=( "${trial_base}@${lam_clean}" )
      done
    else
      FLOW_TRIALS+=( "${trial_base}" )
    fi
  else
    FLOW_TRIALS+=( "${trial_base}" )
  fi
done

for raw_trial in "${FLOW_TRIALS[@]}"; do
  trial="$(printf '%s' "${raw_trial}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ -n "${trial}" ]] || continue
  trial_name="${trial}"
  trial_slug="$(printf '%s' "${trial_name}" | tr '@/:' '____')"
  trial_base="${trial_name%%@*}"
  trial_sparse_lambda="${FLOW_SPARSITY_L1_LAMBDA}"
  if [[ "${trial_name}" == *"@"* ]]; then
    trial_sparse_lambda="${trial_name#*@}"
  fi

  trial_status="ok"
  trial_exit=0
  trial_batch_used=""
  train_extra_args="--epochs ${FLOW_EPOCHS} --early-stopping-patience 0 --progress --verbose"

  case "${trial_base}" in
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
    fp32_sparse)
      train_extra_args+=" --no-amp --tf32 off --sparsity-mode l1 --sparsity-l1-lambda ${trial_sparse_lambda}"
      ;;
    fp16_sparse)
      train_extra_args+=" --amp --amp-dtype fp16 --tf32 on --sparsity-mode l1 --sparsity-l1-lambda ${trial_sparse_lambda}"
      ;;
    bf16_sparse|sparsity)
      train_extra_args+=" --amp --amp-dtype bf16 --tf32 on --sparsity-mode l1 --sparsity-l1-lambda ${trial_sparse_lambda}"
      ;;
    *)
      trial_status="skipped"
      trial_exit=126
      ;;
  esac

  local_trial_dir="${LOCAL_BENCH_ROOT}/${trial_slug}"
  remote_trial_dir="${REMOTE_MATRIX_DIR}/${trial_slug}"
  mkdir -p "${local_trial_dir}"

  if [[ "${trial_status}" == "ok" ]]; then
    telemetry_checkpoint "trial_${trial_slug}" "running" "trial started"
    train_extra_args+=" --distributed-backend ${FLOW_DISTRIBUTED_BACKEND}"
    for batch_try in "${FLOW_BATCH_ATTEMPTS[@]}"; do
      trial_batch_used="${batch_try}"
      telemetry_event "benchmark_trial_batch_attempt" "info" "starting trial batch attempt" "{\"trial\":\"${trial_name}\",\"trial_slug\":\"${trial_slug}\",\"batch_size\":${batch_try}}"
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
      "TRIAL='${trial_slug}' REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' REMOTE_MANIFEST='${REMOTE_MANIFEST}' REMOTE_TRIAL_DIR='${remote_trial_dir}' FLOW_HF_REPO_ID='${FLOW_HF_REPO_ID}' FLOW_HF_PREFIX='${FLOW_HF_PREFIX}' FLOW_HF_SCHEMA_FILTER='${FLOW_HF_SCHEMA_FILTER}' FLOW_TRAIN_NPROC_PER_NODE='${FLOW_TRAIN_NPROC_PER_NODE}' FLOW_BATCH_SIZE_TRY='${batch_try}' FLOW_NUM_WORKERS='${FLOW_NUM_WORKERS}' FLOW_RUNTIME_MAX_SAMPLES_PER_GAME='${FLOW_RUNTIME_MAX_SAMPLES_PER_GAME}' FLOW_MAX_TOTAL_ROWS='${FLOW_MAX_TOTAL_ROWS}' TRAIN_EXTRA_ARGS='${train_extra_args}' /bin/bash -s" <<'EOF_REMOTE'
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
export HF_WRITE_TOKEN=""
export HF_TOKEN=""

export OUTPUT_PATH="${REMOTE_TRIAL_DIR}/model_${TRIAL}.pt"
export METRICS_OUT="${REMOTE_TRIAL_DIR}/metrics_${TRIAL}.json"
export TRAIN_PROGRESS_JSONL_OUT="${REMOTE_TRIAL_DIR}/progress_${TRIAL}.jsonl"
export TRAIN_BEST_CHECKPOINT_OUT="${REMOTE_TRIAL_DIR}/model_best_${TRIAL}.pt"
export TRAIN_EPOCH_CHECKPOINT_DIR="${REMOTE_TRIAL_DIR}/epoch_checkpoints"

export TRAIN_NPROC_PER_NODE="${FLOW_TRAIN_NPROC_PER_NODE}"
export TRAIN_BATCH_SIZE="${FLOW_BATCH_SIZE_TRY}"
export TRAIN_NUM_WORKERS="${FLOW_NUM_WORKERS}"
export TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME="${FLOW_RUNTIME_MAX_SAMPLES_PER_GAME}"
export TRAIN_REQUIRE_RUNTIME_SPLICE_CACHE=1
export TRAIN_MAX_TOTAL_ROWS="${FLOW_MAX_TOTAL_ROWS}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS} --max-total-rows ${FLOW_MAX_TOTAL_ROWS} --telemetry-dir ${REMOTE_TRIAL_DIR}/telemetry"

log_path="${REMOTE_TRIAL_DIR}/train_stdout_${TRIAL}.log"
rc=0
bash "${REMOTE_REPO_DIR}/deploy/runpod_cloud_training/train_baseline_preset.sh" >"${log_path}" 2>&1 || rc=$?
printf '%s\n' "${rc}" > "${REMOTE_TRIAL_DIR}/train_exit_code.txt"
exit 0
EOF_REMOTE

      trial_exit="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "cat '${remote_trial_dir}/train_exit_code.txt' 2>/dev/null || echo 99")"
      if [[ "${trial_exit}" == "0" ]]; then
        trial_status="ok"
        break
      fi
      if [[ -n "${FLOW_BATCH_SIZE_OVERRIDE}" ]]; then
        trial_status="failed"
        break
      fi
      if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
        "tail -n 220 '${remote_trial_dir}/train_stdout_${trial_slug}.log' 2>/dev/null | grep -Eiq 'cuda out of memory|out of memory|cublas_status_alloc_failed|cuda error: out of memory'"; then
        trial_status="failed"
        telemetry_event "benchmark_trial_batch_attempt" "warn" "oom detected; retrying lower batch" "{\"trial\":\"${trial_name}\",\"trial_slug\":\"${trial_slug}\",\"failed_batch\":${batch_try}}"
        continue
      fi
      trial_status="failed"
      break
    done
  fi

  if [[ "${trial_status}" == "ok" ]]; then
    telemetry_checkpoint "trial_${trial_slug}" "done" "trial completed successfully"
  elif [[ "${trial_status}" == "failed" ]]; then
    telemetry_checkpoint "trial_${trial_slug}" "error" "trial failed"
  else
    telemetry_checkpoint "trial_${trial_slug}" "done" "trial skipped"
  fi
  telemetry_event "benchmark_trial" "info" "trial processed" "{\"trial\":\"${trial_name}\",\"trial_slug\":\"${trial_slug}\",\"sparsity_l1_lambda\":\"${trial_sparse_lambda}\",\"batch_size_used\":\"${trial_batch_used}\",\"status\":\"${trial_status}\",\"exit_code\":${trial_exit}}"

  transfer_trial_artifacts "${trial_slug}" "${remote_trial_dir}" "${local_trial_dir}" || true

  trial_metrics_json="$(extract_trial_metrics_json "${local_trial_dir}")"
  printf '{"trial":"%s","trial_slug":"%s","sparsity_l1_lambda":"%s","batch_size_used":"%s","status":"%s","exit_code":%s,"remote_trial_dir":"%s","metrics":%s}\n' \
    "${trial_name}" "${trial_slug}" "${trial_sparse_lambda}" "${trial_batch_used}" "${trial_status}" "${trial_exit}" "${remote_trial_dir}" "${trial_metrics_json}" >> "${LOCAL_SUMMARY_JSONL}"

  printf '| %s | %s | %s | `%s` |\n' "${trial_name}" "${trial_status}" "${trial_exit}" "${local_trial_dir}" >> "${LOCAL_SUMMARY_MD}"
done

cat >> "${LOCAL_SUMMARY_MD}" <<'MD'

## Trial Metrics

| trial | rows_per_second | last_val_loss | last_top1 | last_top5 |
|---|---:|---:|---:|---:|
MD
while IFS= read -r row || [[ -n "${row}" ]]; do
  [[ -n "${row}" ]] || continue
  t="$(printf '%s\n' "${row}" | jq -r '.trial')"
  s="$(printf '%s\n' "${row}" | jq -r '.metrics.rows_per_second // "n/a"')"
  vl="$(printf '%s\n' "${row}" | jq -r '.metrics.last_val_loss // "n/a"')"
  t1="$(printf '%s\n' "${row}" | jq -r '.metrics.last_top1 // "n/a"')"
  t5="$(printf '%s\n' "${row}" | jq -r '.metrics.last_top5 // "n/a"')"
  printf '| %s | `%s` | `%s` | `%s` | `%s` |\n' "${t}" "${s}" "${vl}" "${t1}" "${t5}" >> "${LOCAL_SUMMARY_MD}"
done < "${LOCAL_SUMMARY_JSONL}"

if [[ "${FLOW_SKIP_FINAL_COLLECT}" != "1" ]]; then
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
  RUNPOD_COLLECT_INCLUDE_EPOCH_CHECKPOINTS="${FLOW_COLLECT_INCLUDE_EPOCH_CHECKPOINTS}" \
    bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh" || true
fi

runpod_cycle_append_report "${REPORT_MD}" \
  "## Benchmark Matrix" \
  "- Script: \`scripts/runpod_cycle_benchmark_matrix.sh\`" \
  "- Local summary jsonl: \`${LOCAL_SUMMARY_JSONL}\`" \
  "- Local summary markdown: \`${LOCAL_SUMMARY_MD}\`" \
  "- Local benchmark root: \`${LOCAL_BENCH_ROOT}\`" \
  ""

if [[ "${FLOW_TERMINATE_POD}" == "1" ]]; then
  telemetry_checkpoint "benchmark_terminate_pod" "running" "terminating benchmark pod"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh"
  telemetry_checkpoint "benchmark_terminate_pod" "done" "benchmark pod terminated"
elif [[ "${FLOW_STOP_POD}" == "1" ]]; then
  telemetry_checkpoint "benchmark_stop_pod" "running" "stopping benchmark pod"
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh"
  telemetry_checkpoint "benchmark_stop_pod" "done" "benchmark pod stopped"
fi

telemetry_checkpoint "benchmark_matrix" "done" "benchmark matrix finished"
telemetry_event "benchmark_matrix_complete" "ok" "benchmark matrix completed"
telemetry_healthcheck success "run_id=${RUN_ID} benchmark_matrix_complete"

echo "[runpod-bench-matrix] run_id=${RUN_ID}"
echo "[runpod-bench-matrix] summary=${LOCAL_SUMMARY_MD}"
