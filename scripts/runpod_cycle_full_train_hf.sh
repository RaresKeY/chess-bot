#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${CYCLE_DIR}" "${LOGS_DIR}" "$(dirname "${REPORT_MD}")"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd ssh

HF_DATASET_REPO_ID="${RUNPOD_HF_DATASET_REPO_ID:-${HF_DATASET_REPO_ID:-}}"
HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-${HF_DATASET_PATH_PREFIX:-validated_datasets}}"
HF_DATASET_SCHEMA_FILTER="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-${HF_DATASET_SCHEMA_FILTER:-auto}}"
HF_DATASET_NAME="${RUNPOD_HF_DATASET_NAME:-${HF_DATASET_NAME:-}}"
HF_DATASET_VERSION="${RUNPOD_HF_DATASET_VERSION:-${HF_DATASET_VERSION:-}}"
[[ -n "${HF_DATASET_REPO_ID}" ]] || { echo "[runpod-cycle-full-train-hf] missing RUNPOD_HF_DATASET_REPO_ID/HF_DATASET_REPO_ID" >&2; exit 1; }

FLOW_EPOCHS="${RUNPOD_FULL_TRAIN_EPOCHS:-100}"
FLOW_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
FLOW_MIN_MEMORY_GB="${RUNPOD_GPU_MIN_MEMORY_GB:-24}"
FLOW_MAX_HOURLY_PRICE="${RUNPOD_GPU_MAX_HOURLY_PRICE:-0}"
FLOW_GPU_SAMPLE_SECONDS="${RUNPOD_GPU_SAMPLE_SECONDS:-5}"
FLOW_AUTO_STOP_ON_FAILURE="${RUNPOD_STOP_ON_FAILURE:-1}"
FLOW_SKIP_START="${RUNPOD_CYCLE_SKIP_START:-0}"
FLOW_RUNTIME_MIN_CONTEXT="${RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT:-${TRAIN_RUNTIME_MIN_CONTEXT:-8}}"
FLOW_RUNTIME_MIN_TARGET="${RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET:-${TRAIN_RUNTIME_MIN_TARGET:-1}}"
FLOW_RUNTIME_MAX_SAMPLES_PER_GAME="${RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME:-${TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME:-0}}"

GPU_SEARCH_JSON="${CYCLE_DIR}/gpu_search.json"
GPU_SELECTION_JSON="${CYCLE_DIR}/gpu_selection.json"
GPU_SEARCH_LOG="${LOGS_DIR}/gpu_search.log"
REMOTE_FETCH_LOG="${LOGS_DIR}/hf_fetch_remote.log"
REMOTE_CONTEXT_LOG="${LOGS_DIR}/context_probe_remote.log"
REMOTE_TRAIN_LAUNCH_LOG="${LOGS_DIR}/train_launch_remote.log"
PLAY_CMD_TXT="${CYCLE_DIR}/quick_play_command.txt"

POD_STARTED=0
FLOW_SUCCESS=0
FLOW_INTERRUPTED=0

cleanup_child_processes() {
  local pids=()
  mapfile -t pids < <(jobs -pr 2>/dev/null || true)
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" >/dev/null 2>&1 || true
    wait "${pids[@]}" >/dev/null 2>&1 || true
  fi
}

handle_interrupt() {
  FLOW_INTERRUPTED=1
  printf '\n' >&2
  echo "[runpod-cycle-full-train-hf] interrupted; stopping local child processes" >&2
  cleanup_child_processes
  stty sane >/dev/null 2>&1 || true
  exit 130
}

cleanup_on_error() {
  stty sane >/dev/null 2>&1 || true
  if [[ "${FLOW_SUCCESS}" == "1" ]]; then
    return 0
  fi
  if [[ "${FLOW_INTERRUPTED}" == "1" ]]; then
    cleanup_child_processes
  fi
  if [[ "${POD_STARTED}" != "1" ]]; then
    return 0
  fi
  if [[ "${FLOW_AUTO_STOP_ON_FAILURE}" != "1" ]]; then
    return 0
  fi
  echo "[runpod-cycle-full-train-hf] flow failed; attempting best-effort pod stop" >&2
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh" >/dev/null 2>&1 || true
}
trap handle_interrupt INT TERM
trap cleanup_on_error EXIT

pick_gpu() {
  if [[ -n "${RUNPOD_GPU_TYPE_ID:-}" ]]; then
    printf '%s\n' "${RUNPOD_GPU_TYPE_ID}"
    jq -nc --arg source "explicit_env" --arg gpu_type_id "${RUNPOD_GPU_TYPE_ID}" \
      '{selection_source:$source, chosen_gpu:{id:$gpu_type_id, display_name:$gpu_type_id}}' > "${GPU_SELECTION_JSON}"
    return 0
  fi

  local search_cmd=(
    "${PY_BIN}" "${REPO_ROOT}/scripts/runpod_provision.py"
    --keyring-service runpod
    --keyring-username RUNPOD_API_KEY
    gpu-search
    --cloud-type "${FLOW_CLOUD_TYPE}"
    --min-memory-gb "${FLOW_MIN_MEMORY_GB}"
    --limit 25
  )
  if [[ "${FLOW_MAX_HOURLY_PRICE}" != "0" ]]; then
    search_cmd+=( --max-hourly-price "${FLOW_MAX_HOURLY_PRICE}" )
  fi

  {
    printf '[runpod-cycle-full-train-hf] exec:'
    printf ' %q' "${search_cmd[@]}"
    printf '\n'
  } > "${GPU_SEARCH_LOG}"

  if ! "${search_cmd[@]}" > "${GPU_SEARCH_JSON}" 2>>"${GPU_SEARCH_LOG}"; then
    local fallback="${RUNPOD_GPU_FALLBACK_TYPE_ID:-NVIDIA GeForce RTX 4090}"
    echo "[runpod-cycle-full-train-hf] gpu-search unavailable; falling back to ${fallback}" >&2
    jq -nc --arg source "fallback_after_gpu_search_error" --arg gpu_type_id "${fallback}" \
      '{selection_source:$source, chosen_gpu:{id:$gpu_type_id, display_name:$gpu_type_id}}' > "${GPU_SELECTION_JSON}"
    printf '%s\n' "${fallback}"
    return 0
  fi

  local pref
  for pref in \
    "NVIDIA H100" \
    "NVIDIA A100" \
    "NVIDIA L40S" \
    "RTX 6000 Ada" \
    "RTX 4090" \
    "RTX 3090"
  do
    local picked
    picked="$(jq -r --arg pref "${pref}" '
      (.gpus // []) | map(select((.display_name // "") | test($pref; "i")))
      | sort_by((-.memory_gb), .price_per_hr)
      | .[0].id // empty
    ' "${GPU_SEARCH_JSON}")"
    if [[ -n "${picked}" ]]; then
      jq -nc --arg source "gpu_search_preferred_name" --arg pref "${pref}" \
        --argjson gpu "$(jq -c --arg id "${picked}" '(.gpus // []) | map(select(.id == $id))[0]' "${GPU_SEARCH_JSON}")" \
        '{selection_source:$source, preferred_match:$pref, chosen_gpu:$gpu}' > "${GPU_SELECTION_JSON}"
      printf '%s\n' "${picked}"
      return 0
    fi
  done

  local picked
  picked="$(jq -r '(.gpus // []) | sort_by((-.memory_gb), .price_per_hr) | .[0].id // empty' "${GPU_SEARCH_JSON}")"
  [[ -n "${picked}" ]] || { echo "[runpod-cycle-full-train-hf] no GPU candidates from gpu-search" >&2; return 1; }
  jq -nc --arg source "gpu_search_highest_memory" \
    --argjson gpu "$(jq -c --arg id "${picked}" '(.gpus // []) | map(select(.id == $id))[0]' "${GPU_SEARCH_JSON}")" \
    '{selection_source:$source, chosen_gpu:$gpu}' > "${GPU_SELECTION_JSON}"
  printf '%s\n' "${picked}"
}

wait_remote_repo_ready() {
  local ssh_host="$1"
  local ssh_port="$2"
  local ssh_user="$3"
  local ssh_key="$4"
  local remote_repo_dir="$5"
  local timeout_s="${RUNPOD_REMOTE_READY_TIMEOUT_SECONDS:-300}"
  local poll_s="${RUNPOD_REMOTE_READY_POLL_SECONDS:-5}"
  local deadline=$(( $(date +%s) + timeout_s ))
  local ssh_connect_timeout="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
  local ssh_host_key_checking ssh_known_hosts_file
  ssh_host_key_checking="$(runpod_cycle_ssh_host_key_checking)"
  ssh_known_hosts_file="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"
  local ssh_opts=(-i "${ssh_key}" -p "${ssh_port}" -o BatchMode=yes -o ConnectTimeout="${ssh_connect_timeout}" -o IdentitiesOnly=yes -o "StrictHostKeyChecking=${ssh_host_key_checking}" -o "UserKnownHostsFile=${ssh_known_hosts_file}")
  while true; do
    if ssh "${ssh_opts[@]}" "${ssh_user}@${ssh_host}" \
      "test -d '${remote_repo_dir}' && test -f '${remote_repo_dir}/scripts/train_baseline.py' && test -w '${remote_repo_dir}'" \
      >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "[runpod-cycle-full-train-hf] remote repo not ready: ${remote_repo_dir}" >&2
      return 1
    fi
    sleep "${poll_s}"
  done
}

SELECTED_GPU_TYPE_ID="$(pick_gpu)"
SELECTED_GPU_DISPLAY_NAME="$(jq -r '.chosen_gpu.display_name // .chosen_gpu.id // empty' "${GPU_SELECTION_JSON}")"
runpod_cycle_append_report "${REPORT_MD}" \
  "## Full HF Training Flow (planned)" \
  "- Requested epochs: \`${FLOW_EPOCHS}\`" \
  "- HF dataset repo: \`${HF_DATASET_REPO_ID}\`" \
  "- HF dataset path prefix: \`${HF_DATASET_PATH_PREFIX}\`" \
  "- HF dataset name override: \`${HF_DATASET_NAME:-<all-latest under prefix>}\`" \
  "- HF dataset version override: \`${HF_DATASET_VERSION:-<latest>}\`" \
  "- HF dataset schema filter: \`${HF_DATASET_SCHEMA_FILTER}\`" \
  "- GPU selection source: \`$(jq -r '.selection_source // "unknown"' "${GPU_SELECTION_JSON}")\`" \
  "- Selected GPU type: \`${SELECTED_GPU_DISPLAY_NAME:-$SELECTED_GPU_TYPE_ID}\`" \
  "- GPU search JSON: \`${GPU_SEARCH_JSON}\`" \
  "- GPU selection JSON: \`${GPU_SELECTION_JSON}\`" \
  ""

if [[ "${FLOW_SKIP_START}" == "1" ]]; then
  if [[ ! -f "${PROVISION_JSON}" ]]; then
    echo "[runpod-cycle-full-train-hf] RUNPOD_CYCLE_SKIP_START=1 but provision.json not found: ${PROVISION_JSON}" >&2
    exit 1
  fi
  echo "[runpod-cycle-full-train-hf] skipping pod start; reusing existing provision.json for run_id=${RUN_ID}"
  POD_STARTED=1
else
  RUNPOD_GPU_TYPE_ID="${SELECTED_GPU_TYPE_ID}" \
  RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
  RUNPOD_SET_SMOKE_SERVICE_ENVS="${RUNPOD_SET_SMOKE_SERVICE_ENVS:-1}" \
  bash "${REPO_ROOT}/scripts/runpod_cycle_start.sh"
  POD_STARTED=1
fi
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
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_HF_FETCH_MANIFEST="${REMOTE_RUN_DIR}/hf_dataset_fetch_manifest.json"
REMOTE_CONTEXT_JSON="${REMOTE_RUN_DIR}/context_probe_${RUN_ID}.json"
REMOTE_SUGGESTIONS_JSON="${REMOTE_RUN_DIR}/spec_suggestions_gpu_training_${RUN_ID}.json"
REMOTE_SUGGESTIONS_MD="${REMOTE_RUN_DIR}/spec_suggestions_gpu_training_${RUN_ID}.md"
REMOTE_PROGRESS_JSONL="${REMOTE_RUN_DIR}/train_progress_${RUN_ID}.jsonl"
REMOTE_TRAIN_LOG="${REMOTE_RUN_DIR}/train_stdout_${RUN_ID}.log"
REMOTE_TRAIN_PID_FILE="${REMOTE_RUN_DIR}/train_pid.txt"
REMOTE_TRAIN_EXIT_CODE_FILE="${REMOTE_RUN_DIR}/train_exit_code.txt"
REMOTE_GPU_SAMPLES_CSV="${REMOTE_RUN_DIR}/gpu_usage_samples_${RUN_ID}.csv"

wait_remote_repo_ready "${SSH_HOST}" "${SSH_PORT}" "${SSH_USER}" "${SSH_KEY}" "${REMOTE_REPO_DIR}"

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "bash -s" <<EOF 2>&1 | tee "${REMOTE_FETCH_LOG}"
set -Eeuo pipefail
mkdir -p '${REMOTE_RUN_DIR}'
export PYTHONPATH='${REMOTE_REPO_DIR}'
fetch_args=(
  --repo-id '${HF_DATASET_REPO_ID}'
  --repo-path-prefix '${HF_DATASET_PATH_PREFIX}'
  --dest-dir '${REMOTE_REPO_DIR}/data/hf_datasets'
  --output-manifest '${REMOTE_HF_FETCH_MANIFEST}'
)
if [[ -n "${HF_DATASET_NAME}" ]]; then
  fetch_args+=( --dataset-name '${HF_DATASET_NAME}' )
  if [[ -n "${HF_DATASET_VERSION}" ]]; then
    echo "[runpod-cycle-full-train-hf] remote_hf_fetch_exec: single dataset ${HF_DATASET_NAME}@${HF_DATASET_VERSION}" >&2
    fetch_args+=( --version '${HF_DATASET_VERSION}' )
  else
    echo "[runpod-cycle-full-train-hf] remote_hf_fetch_exec: single dataset ${HF_DATASET_NAME}@latest" >&2
  fi
else
  echo "[runpod-cycle-full-train-hf] remote_hf_fetch_exec: all-latest under prefix ${HF_DATASET_PATH_PREFIX}" >&2
  fetch_args+=( --all-latest )
fi
'/opt/venvs/chessbot/bin/python' '${REMOTE_REPO_DIR}/scripts/hf_dataset_fetch.py' "${fetch_args[@]}"
EOF

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "CONTEXT_JSON='${REMOTE_CONTEXT_JSON}' SUGGESTIONS_JSON='${REMOTE_SUGGESTIONS_JSON}' SUGGESTIONS_MD='${REMOTE_SUGGESTIONS_MD}' HF_MANIFEST='${REMOTE_HF_FETCH_MANIFEST}' REMOTE_RUN_DIR='${REMOTE_RUN_DIR}' bash -s" \
  <<'EOF' 2>&1 | tee "${REMOTE_CONTEXT_LOG}"
set -Eeuo pipefail
mkdir -p "${REMOTE_RUN_DIR}"
'/opt/venvs/chessbot/bin/python' - <<'PY'
import csv
import json
import math
import os
import subprocess
from pathlib import Path

def count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

manifest_path = Path(os.environ["HF_MANIFEST"])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
train_paths = [Path(p) for p in data.get("aggregate", {}).get("train_paths", []) if p]
val_paths = [Path(p) for p in data.get("aggregate", {}).get("val_paths", []) if p]

dataset_files = []
total_bytes = 0
total_rows = {"train": 0, "val": 0}
for split, paths in (("train", train_paths), ("val", val_paths)):
    for p in paths:
        size_b = p.stat().st_size if p.exists() else 0
        rows = count_jsonl_rows(p) if p.exists() else 0
        total_bytes += size_b
        total_rows[split] += rows
        dataset_files.append({
            "split": split,
            "path": str(p),
            "size_bytes": size_b,
            "rows": rows,
        })

gpu_rows = []
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 5:
            gpu_rows.append(
                {
                    "name": parts[0],
                    "memory_total_mib": int(float(parts[1] or 0)),
                    "memory_free_mib": int(float(parts[2] or 0)),
                    "memory_used_mib": int(float(parts[3] or 0)),
                    "driver_version": parts[4],
                }
            )
except Exception as exc:
    gpu_rows.append({"nvidia_smi_error": str(exc)})

torch_info = {}
try:
    import torch  # type: ignore
    torch_info = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        prop = torch.cuda.get_device_properties(0)
        torch_info["cuda0"] = {
            "name": prop.name,
            "total_memory_mib": int(prop.total_memory // (1024 * 1024)),
            "multi_processor_count": int(getattr(prop, "multi_processor_count", 0)),
        }
except Exception as exc:
    torch_info = {"torch_import_error": str(exc)}

vram_mib = 0
if gpu_rows and "memory_total_mib" in gpu_rows[0]:
    vram_mib = int(gpu_rows[0]["memory_total_mib"])
elif isinstance(torch_info.get("cuda0"), dict):
    vram_mib = int(torch_info["cuda0"].get("total_memory_mib") or 0)

if vram_mib >= 70000:
    batch_size, workers = 8192, 8
elif vram_mib >= 44000:
    batch_size, workers = 4096, 8
elif vram_mib >= 22000:
    batch_size, workers = 2048, 6
elif vram_mib >= 15000:
    batch_size, workers = 1024, 4
else:
    batch_size, workers = 512, 2

context = {
    "run_id": os.environ["CONTEXT_JSON"].split("/")[-1].replace("context_probe_", "").replace(".json", ""),
    "hf_manifest": str(manifest_path),
    "dataset": {
        "dataset_count": int(data.get("aggregate", {}).get("dataset_count", 0)),
        "train_file_count": int(data.get("aggregate", {}).get("train_count", 0)),
        "val_file_count": int(data.get("aggregate", {}).get("val_count", 0)),
        "total_bytes": int(total_bytes),
        "total_gib": round(total_bytes / (1024 ** 3), 4),
        "total_rows": total_rows,
        "files": dataset_files,
    },
    "gpu": {"devices": gpu_rows},
    "torch": torch_info,
}

suggestions = {
    "purpose": "runpod_full_training_hf_sequential",
    "gpu_name": (gpu_rows[0].get("name") if gpu_rows and isinstance(gpu_rows[0], dict) else None),
    "vram_total_mib": vram_mib,
    "dataset_summary": {
        "dataset_count": context["dataset"]["dataset_count"],
        "train_rows": context["dataset"]["total_rows"]["train"],
        "val_rows": context["dataset"]["total_rows"]["val"],
        "total_gib": context["dataset"]["total_gib"],
    },
    "suggested_training_params": {
        "epochs": 100,
        "batch_size": batch_size,
        "num_workers": workers,
        "amp": True,
        "phase_weight_endgame": 2.0,
        "disable_early_stopping_for_full_run": True,
        "train_extra_args": "--epochs 100 --early-stopping-patience 0 --no-progress",
    },
    "notes": [
        "Batch-size suggestion is a conservative VRAM-tier heuristic; verify with collected peak GPU memory samples.",
        "Full run keeps HF aggregate dataset training and reuses the fetched manifest to avoid a second download.",
    ],
}

Path(os.environ["CONTEXT_JSON"]).write_text(json.dumps(context, indent=2) + "\\n", encoding="utf-8")
Path(os.environ["SUGGESTIONS_JSON"]).write_text(json.dumps(suggestions, indent=2) + "\\n", encoding="utf-8")
Path(os.environ["SUGGESTIONS_MD"]).write_text(
    "\\n".join(
        [
            "# GPU Training Spec Suggestions",
            "",
            f"- GPU: `{suggestions.get('gpu_name')}`",
            f"- VRAM (MiB): `{suggestions.get('vram_total_mib')}`",
            f"- Dataset count: `{suggestions['dataset_summary']['dataset_count']}`",
            f"- Train rows: `{suggestions['dataset_summary']['train_rows']}`",
            f"- Val rows: `{suggestions['dataset_summary']['val_rows']}`",
            f"- Dataset total GiB: `{suggestions['dataset_summary']['total_gib']}`",
            f"- Suggested batch size: `{batch_size}`",
            f"- Suggested num_workers: `{workers}`",
            "- Suggested run: `--epochs 100 --early-stopping-patience 0 --no-progress` with progress JSONL enabled",
            "",
            "These are pre-training suggestions. Compare against collected peak VRAM samples after the run.",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps({"context_probe_written": os.environ["CONTEXT_JSON"], "suggestions_written": os.environ["SUGGESTIONS_JSON"]}))
PY
EOF

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "RUN_ID='${RUN_ID}' REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' REMOTE_RUN_DIR='${REMOTE_RUN_DIR}' REMOTE_CONTEXT_JSON='${REMOTE_CONTEXT_JSON}' REMOTE_PROGRESS_JSONL='${REMOTE_PROGRESS_JSONL}' REMOTE_TRAIN_LOG='${REMOTE_TRAIN_LOG}' REMOTE_TRAIN_PID_FILE='${REMOTE_TRAIN_PID_FILE}' REMOTE_TRAIN_EXIT_CODE_FILE='${REMOTE_TRAIN_EXIT_CODE_FILE}' REMOTE_GPU_SAMPLES_CSV='${REMOTE_GPU_SAMPLES_CSV}' REMOTE_HF_FETCH_MANIFEST='${REMOTE_HF_FETCH_MANIFEST}' HF_DATASET_REPO_ID='${HF_DATASET_REPO_ID}' HF_DATASET_PATH_PREFIX='${HF_DATASET_PATH_PREFIX}' HF_DATASET_SCHEMA_FILTER='${HF_DATASET_SCHEMA_FILTER}' HF_DATASET_NAME='${HF_DATASET_NAME}' HF_DATASET_VERSION='${HF_DATASET_VERSION}' FLOW_EPOCHS='${FLOW_EPOCHS}' FLOW_GPU_SAMPLE_SECONDS='${FLOW_GPU_SAMPLE_SECONDS}' FLOW_RUNTIME_MIN_CONTEXT='${FLOW_RUNTIME_MIN_CONTEXT}' FLOW_RUNTIME_MIN_TARGET='${FLOW_RUNTIME_MIN_TARGET}' FLOW_RUNTIME_MAX_SAMPLES_PER_GAME='${FLOW_RUNTIME_MAX_SAMPLES_PER_GAME}' RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE='${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-}' RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE='${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-}' bash -s" \
  <<'EOF' 2>&1 | tee "${REMOTE_TRAIN_LAUNCH_LOG}"
set -Eeuo pipefail
mkdir -p "${REMOTE_RUN_DIR}"
cat > "${REMOTE_RUN_DIR}/train_background_launcher.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "${REMOTE_REPO_DIR}"
mkdir -p "${REMOTE_RUN_DIR}"
if [[ -f "${REMOTE_TRAIN_LOG}" || -f "${REMOTE_PROGRESS_JSONL}" || -f "${REMOTE_GPU_SAMPLES_CSV}" ]]; then
  echo "[runpod-cycle-full-train-hf] clearing stale per-run files under reused run_id=${RUN_ID}" >> "${REMOTE_TRAIN_LOG}.bootstrap" 2>/dev/null || true
fi
rm -f "${REMOTE_PROGRESS_JSONL}" "${REMOTE_GPU_SAMPLES_CSV}" "${REMOTE_TRAIN_PID_FILE}" "${REMOTE_TRAIN_LOG}"
rm -f "${REMOTE_TRAIN_EXIT_CODE_FILE}"
gpu_sampler_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -v ts="${ts}" 'BEGIN{FS=", *"; OFS=","} {print ts,$1,$2,$3,$4}' >> "${REMOTE_GPU_SAMPLES_CSV}" 2>/dev/null || true
      sleep "${FLOW_GPU_SAMPLE_SECONDS}"
    done
  ) &
  gpu_sampler_pid="$!"
fi
readarray -t suggested < <('/opt/venvs/chessbot/bin/python' - "${REMOTE_CONTEXT_JSON}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
vram = 0
for d in data.get("gpu", {}).get("devices", []):
    if isinstance(d, dict) and "memory_total_mib" in d:
        vram = int(d["memory_total_mib"])
        break
if vram >= 70000:
    print(8192); print(8)
elif vram >= 44000:
    print(4096); print(8)
elif vram >= 22000:
    print(2048); print(6)
elif vram >= 15000:
    print(1024); print(4)
else:
    print(512); print(2)
PY
)
TRAIN_BATCH_SIZE="${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-${suggested[0]:-2048}}"
cpu_threads="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"
if ! [[ "${cpu_threads}" =~ ^[0-9]+$ ]]; then
  cpu_threads=0
fi
auto_num_workers=1
if (( cpu_threads > 1 )); then
  auto_num_workers=$((cpu_threads - 1))
fi
TRAIN_NUM_WORKERS="${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-${auto_num_workers}}"
export REPO_DIR="${REMOTE_REPO_DIR}"
export OUTPUT_PATH="${REMOTE_RUN_DIR}/model_${RUN_ID}.pt"
export METRICS_OUT="${REMOTE_RUN_DIR}/metrics_${RUN_ID}.json"
export TRAIN_BATCH_SIZE
export TRAIN_NUM_WORKERS
export TRAIN_PROGRESS_JSONL_OUT="${REMOTE_PROGRESS_JSONL}"
export HF_FETCH_LATEST_ALL_DATASETS=1
export HF_USE_EXISTING_FETCH_MANIFEST=1
export HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID}"
export HF_DATASET_PATH_PREFIX="${HF_DATASET_PATH_PREFIX}"
export HF_DATASET_SCHEMA_FILTER="${HF_DATASET_SCHEMA_FILTER}"
export HF_DATASET_FETCH_MANIFEST="${REMOTE_HF_FETCH_MANIFEST}"
export HF_DATASET_CACHE_DIR="${REMOTE_REPO_DIR}/data/hf_datasets"
export TRAIN_RUNTIME_MIN_CONTEXT="${FLOW_RUNTIME_MIN_CONTEXT}"
export TRAIN_RUNTIME_MIN_TARGET="${FLOW_RUNTIME_MIN_TARGET}"
export TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME="${FLOW_RUNTIME_MAX_SAMPLES_PER_GAME}"
export TRAIN_EXTRA_ARGS="--epochs ${FLOW_EPOCHS} --early-stopping-patience 0 --no-progress"
TRAIN_PRESET_IMAGE="/opt/runpod_cloud_training/train_baseline_preset.sh"
TRAIN_PRESET_REPO="${REMOTE_REPO_DIR}/deploy/runpod_cloud_training/train_baseline_preset.sh"
TRAIN_PRESET_SCRIPT="${TRAIN_PRESET_IMAGE}"
if [[ -f "${TRAIN_PRESET_REPO}" ]]; then
  TRAIN_PRESET_SCRIPT="${TRAIN_PRESET_REPO}"
fi
preset_has_hf=0
preset_has_progress=0
if [[ -f "${TRAIN_PRESET_SCRIPT}" ]] && grep -q 'HF_FETCH_LATEST_ALL_DATASETS' "${TRAIN_PRESET_SCRIPT}"; then
  preset_has_hf=1
fi
if [[ -f "${TRAIN_PRESET_SCRIPT}" ]] && grep -q 'TRAIN_PROGRESS_JSONL_OUT' "${TRAIN_PRESET_SCRIPT}"; then
  preset_has_progress=1
fi
{
  echo "[runpod-cycle-full-train-hf] training launch"
  echo "[runpod-cycle-full-train-hf] override_batch_size=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>} override_num_workers=${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-<unset>}"
  echo "[runpod-cycle-full-train-hf] cpu_threads=${cpu_threads} auto_num_workers=${auto_num_workers} vram_suggested_num_workers=${suggested[1]:-6}"
  echo "[runpod-cycle-full-train-hf] batch_size=${TRAIN_BATCH_SIZE} num_workers=${TRAIN_NUM_WORKERS} epochs=${FLOW_EPOCHS}"
  echo "[runpod-cycle-full-train-hf] hf_dataset_schema_filter=${HF_DATASET_SCHEMA_FILTER}"
  echo "[runpod-cycle-full-train-hf] runtime_min_context=${TRAIN_RUNTIME_MIN_CONTEXT} runtime_min_target=${TRAIN_RUNTIME_MIN_TARGET} runtime_max_samples_per_game=${TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME}"
  echo "[runpod-cycle-full-train-hf] progress_jsonl=${REMOTE_PROGRESS_JSONL}"
  echo "[runpod-cycle-full-train-hf] hf_manifest=${REMOTE_HF_FETCH_MANIFEST}"
  echo "[runpod-cycle-full-train-hf] train_preset_script=${TRAIN_PRESET_SCRIPT}"
  echo "[runpod-cycle-full-train-hf] train_preset_has_hf=${preset_has_hf} train_preset_has_progress=${preset_has_progress}"
} >> "${REMOTE_TRAIN_LOG}"
rc=0
if [[ "${preset_has_hf}" == "1" ]]; then
  if [[ "${preset_has_progress}" != "1" ]]; then
    unset TRAIN_PROGRESS_JSONL_OUT
    echo "[runpod-cycle-full-train-hf] preset lacks TRAIN_PROGRESS_JSONL_OUT support; progress JSONL disabled for this run" >> "${REMOTE_TRAIN_LOG}"
  fi
  bash "${TRAIN_PRESET_SCRIPT}" >> "${REMOTE_TRAIN_LOG}" 2>&1 || rc=$?
else
  echo "[runpod-cycle-full-train-hf] preset lacks HF aggregate support; using direct train_baseline.py fallback" >> "${REMOTE_TRAIN_LOG}"
  mapfile -t hf_train_paths < <('/opt/venvs/chessbot/bin/python' - "${REMOTE_HF_FETCH_MANIFEST}" "${HF_DATASET_SCHEMA_FILTER}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
schema_filter = (sys.argv[2] or "auto").strip()
agg = data.get("aggregate", {}) or {}
agg_by_format = data.get("aggregate_by_format", {}) or {}
def chosen():
    if schema_filter and schema_filter not in {"", "auto"}:
        return schema_filter
    for cand in ("game_jsonl_runtime_splice_v1", "splice_rows_legacy"):
        if cand in agg_by_format:
            return cand
    return ""
fmt = chosen()
paths = agg_by_format.get(fmt, {}).get("train_paths", []) if fmt else agg.get("train_paths", [])
for p in paths:
    if p:
        print(p)
PY
)
  mapfile -t hf_val_paths < <('/opt/venvs/chessbot/bin/python' - "${REMOTE_HF_FETCH_MANIFEST}" "${HF_DATASET_SCHEMA_FILTER}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
schema_filter = (sys.argv[2] or "auto").strip()
agg = data.get("aggregate", {}) or {}
agg_by_format = data.get("aggregate_by_format", {}) or {}
def chosen():
    if schema_filter and schema_filter not in {"", "auto"}:
        return schema_filter
    for cand in ("game_jsonl_runtime_splice_v1", "splice_rows_legacy"):
        if cand in agg_by_format:
            return cand
    return ""
fmt = chosen()
paths = agg_by_format.get(fmt, {}).get("val_paths", []) if fmt else agg.get("val_paths", [])
for p in paths:
    if p:
        print(p)
PY
)
  selected_schema="$('/opt/venvs/chessbot/bin/python' - "${REMOTE_HF_FETCH_MANIFEST}" "${HF_DATASET_SCHEMA_FILTER}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
schema_filter = (sys.argv[2] or "auto").strip()
agg_by_format = data.get("aggregate_by_format", {}) or {}
if schema_filter and schema_filter not in {"", "auto"}:
    print(schema_filter)
elif "game_jsonl_runtime_splice_v1" in agg_by_format:
    print("game_jsonl_runtime_splice_v1")
elif "splice_rows_legacy" in agg_by_format:
    print("splice_rows_legacy")
else:
    print("")
PY
)"
  if (( ${#hf_train_paths[@]} == 0 || ${#hf_val_paths[@]} == 0 )); then
    echo "[runpod-cycle-full-train-hf] no train/val paths found in HF fetch manifest: ${REMOTE_HF_FETCH_MANIFEST}" >> "${REMOTE_TRAIN_LOG}"
    rc=1
  else
    direct_cmd=( '/opt/venvs/chessbot/bin/python' "${REMOTE_REPO_DIR}/scripts/train_baseline.py"
      --output "${OUTPUT_PATH}"
      --metrics-out "${METRICS_OUT}"
      --epochs "${FLOW_EPOCHS}"
      --lr 0.0002
      --embed-dim 256
      --hidden-dim 512
      --num-layers 2
      --dropout 0.15
      --batch-size "${TRAIN_BATCH_SIZE}"
      --num-workers "${TRAIN_NUM_WORKERS}"
      --amp
      --phase-feature
      --side-to-move-feature
      --phase-weight-endgame 2.0
      --lr-scheduler plateau
      --lr-scheduler-metric val_loss
      --lr-plateau-factor 0.5
      --lr-plateau-patience 3
      --lr-plateau-threshold 0.0001
      --early-stopping-patience 0
      --no-progress
    )
    if grep -q 'progress-jsonl-out' "${REMOTE_REPO_DIR}/scripts/train_baseline.py" 2>/dev/null; then
      direct_cmd+=( --progress-jsonl-out "${REMOTE_PROGRESS_JSONL}" )
    else
      echo "[runpod-cycle-full-train-hf] remote train_baseline.py lacks --progress-jsonl-out; local watcher will wait for exit sentinel only" >> "${REMOTE_TRAIN_LOG}"
    fi
    for p in "${hf_train_paths[@]}"; do
      direct_cmd+=( --train "${p}" )
    done
    for p in "${hf_val_paths[@]}"; do
      direct_cmd+=( --val "${p}" )
    done
    if [[ "${selected_schema}" == "game_jsonl_runtime_splice_v1" ]]; then
      direct_cmd+=( --runtime-min-context "${TRAIN_RUNTIME_MIN_CONTEXT}" --runtime-min-target "${TRAIN_RUNTIME_MIN_TARGET}" --runtime-max-samples-per-game "${TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME}" )
    fi
    {
      printf '[runpod-cycle-full-train-hf] direct_fallback_exec:'
      printf ' %q' "${direct_cmd[@]}"
      printf '\n'
    } >> "${REMOTE_TRAIN_LOG}"
    "${direct_cmd[@]}" >> "${REMOTE_TRAIN_LOG}" 2>&1 || rc=$?
  fi
fi
if [[ -n "${gpu_sampler_pid}" ]]; then
  kill "${gpu_sampler_pid}" >/dev/null 2>&1 || true
fi
printf '%s\n' "${rc}" > "${REMOTE_TRAIN_EXIT_CODE_FILE}"
exit "${rc}"
SH
chmod +x "${REMOTE_RUN_DIR}/train_background_launcher.sh"
nohup bash "${REMOTE_RUN_DIR}/train_background_launcher.sh" >/dev/null 2>&1 &
bg_pid=$!
printf '%s\n' "${bg_pid}" > "${REMOTE_TRAIN_PID_FILE}"
echo "[runpod-cycle-full-train-hf] launched_remote_train_pid=${bg_pid}"
echo "[runpod-cycle-full-train-hf] progress_jsonl=${REMOTE_PROGRESS_JSONL}"
echo "[runpod-cycle-full-train-hf] exit_code_file=${REMOTE_TRAIN_EXIT_CODE_FILE}"
EOF

RUNPOD_CYCLE_RUN_ID="${RUN_ID}" \
RUNPOD_REMOTE_TRAIN_EXIT_CODE_FILE="${REMOTE_TRAIN_EXIT_CODE_FILE}" \
RUNPOD_REMOTE_PROGRESS_JSONL="${REMOTE_PROGRESS_JSONL}" \
bash "${REPO_ROOT}/scripts/runpod_cycle_watch_progress.sh"

RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_collect.sh"

LOCAL_COLLECT_DIR="${CYCLE_DIR}/collected/run_artifacts"
LOCAL_SPEC_SUGGESTIONS_DIR="${CYCLE_DIR}/spec_suggestions"
mkdir -p "${LOCAL_SPEC_SUGGESTIONS_DIR}"
LOCAL_OBS_JSON="${LOCAL_SPEC_SUGGESTIONS_DIR}/gpu_full_training_observation_${RUN_ID}.json"
LOCAL_OBS_MD="${LOCAL_SPEC_SUGGESTIONS_DIR}/gpu_full_training_observation_${RUN_ID}.md"

"${PY_BIN}" - "${LOCAL_COLLECT_DIR}" "${LOCAL_OBS_JSON}" "${LOCAL_OBS_MD}" "${RUN_ID}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_artifacts = Path(sys.argv[1])
out_json = Path(sys.argv[2])
out_md = Path(sys.argv[3])
run_id = sys.argv[4]

context_path = next(iter(sorted(run_artifacts.glob(f"context_probe_{run_id}.json"))), None)
metrics_path = next(iter(sorted(run_artifacts.glob(f"metrics_{run_id}.json"))), None)
gpu_csv_path = next(iter(sorted(run_artifacts.glob(f"gpu_usage_samples_{run_id}.csv"))), None)
model_path = next(iter(sorted(run_artifacts.glob(f"model_{run_id}.pt"))), None)

context = json.loads(context_path.read_text(encoding="utf-8")) if context_path and context_path.exists() else {}
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}

gpu_peak = {"memory_used_mib_peak": None, "utilization_gpu_pct_peak": None, "sample_count": 0}
if gpu_csv_path and gpu_csv_path.exists():
    mem_peak = 0
    util_peak = 0
    count = 0
    with gpu_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            count += 1
            try:
                util_peak = max(util_peak, int(float(row[2])))
            except Exception:
                pass
            try:
                mem_peak = max(mem_peak, int(float(row[3])))
            except Exception:
                pass
    gpu_peak = {
        "memory_used_mib_peak": mem_peak or None,
        "utilization_gpu_pct_peak": util_peak or None,
        "sample_count": count,
    }

history = metrics.get("history") or []
last_epoch = history[-1] if history else {}
obs = {
    "run_id": run_id,
    "model_path": str(model_path) if model_path else "",
    "metrics_path": str(metrics_path) if metrics_path else "",
    "context_probe_path": str(context_path) if context_path else "",
    "gpu_samples_path": str(gpu_csv_path) if gpu_csv_path else "",
    "dataset_summary": (context.get("dataset") or {}),
    "gpu_context": (context.get("gpu") or {}),
    "gpu_peak_observed": gpu_peak,
    "training_summary": {
        "epochs_requested": metrics.get("epochs"),
        "epochs_ran": len(history),
        "last_epoch": last_epoch.get("epoch"),
        "last_val_loss": last_epoch.get("val_loss"),
        "last_top1": last_epoch.get("top1"),
        "best_checkpoint": metrics.get("best_checkpoint"),
        "early_stopping": metrics.get("early_stopping"),
    },
    "spec_suggestion_note": "Use peak memory plus dataset size/rows to tune future RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE by GPU SKU.",
}
out_json.write_text(json.dumps(obs, indent=2) + "\n", encoding="utf-8")

gpu_name = ""
devices = ((context.get("gpu") or {}).get("devices") or [])
if devices and isinstance(devices[0], dict):
    gpu_name = str(devices[0].get("name") or "")
total_rows = ((context.get("dataset") or {}).get("total_rows") or {})
dataset_total_gib = (context.get("dataset") or {}).get("total_gib")

md_lines = [
    f"# RunPod Full Training Observation ({run_id})",
    "",
    f"- GPU: `{gpu_name or 'unknown'}`",
    f"- Dataset GiB (fetched aggregate): `{dataset_total_gib}`",
    f"- Train rows: `{total_rows.get('train')}`",
    f"- Val rows: `{total_rows.get('val')}`",
    f"- Peak GPU memory used (MiB): `{gpu_peak.get('memory_used_mib_peak')}`",
    f"- Peak GPU util (%): `{gpu_peak.get('utilization_gpu_pct_peak')}`",
    f"- GPU sample count: `{gpu_peak.get('sample_count')}`",
    f"- Epochs ran: `{len(history)}` / requested `{metrics.get('epochs')}`",
    f"- Last val_loss: `{last_epoch.get('val_loss')}`",
    f"- Last top1: `{last_epoch.get('top1')}`",
    f"- Model artifact: `{model_path}`" if model_path else "- Model artifact: ``",
    "",
    "Use this observation with the pre-training `spec_suggestions_gpu_training_*` artifact to tune future full-run defaults per GPU SKU.",
    "",
]
out_md.write_text("\n".join(md_lines), encoding="utf-8")
PY

MODEL_PATH_LOCAL="$(runpod_cycle_find_model_artifact "${LOCAL_COLLECT_DIR}" "${RUN_ID}")"
if [[ -z "${MODEL_PATH_LOCAL}" || ! -f "${MODEL_PATH_LOCAL}" ]]; then
  echo "[runpod-cycle-full-train-hf] no collected model artifact found under ${LOCAL_COLLECT_DIR}" >&2
  exit 1
fi
{
  printf '.venv/bin/python main.py --model %q\n' "${MODEL_PATH_LOCAL}"
} > "${PLAY_CMD_TXT}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## HF Full Training Result" \
  "- Remote progress JSONL: \`${REMOTE_PROGRESS_JSONL}\`" \
  "- Remote training log: \`${REMOTE_TRAIN_LOG}\`" \
  "- Remote context probe JSON: \`${REMOTE_CONTEXT_JSON}\`" \
  "- Remote pre-train spec suggestions JSON: \`${REMOTE_SUGGESTIONS_JSON}\`" \
  "- Remote pre-train spec suggestions MD: \`${REMOTE_SUGGESTIONS_MD}\`" \
  "- Local collected run artifacts: \`${LOCAL_COLLECT_DIR}\`" \
  "- Local post-run observation JSON: \`${LOCAL_OBS_JSON}\`" \
  "- Local post-run observation MD: \`${LOCAL_OBS_MD}\`" \
  "- Quick play command file: \`${PLAY_CMD_TXT}\`" \
  ""

RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_stop.sh"
FLOW_SUCCESS=1

echo "[runpod-cycle-full-train-hf] run_id=${RUN_ID}"
echo "[runpod-cycle-full-train-hf] collected_artifacts=${LOCAL_COLLECT_DIR}"
echo "[runpod-cycle-full-train-hf] quick_play_command=$(cat "${PLAY_CMD_TXT}")"
