#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUNPOD_CYCLE_RUN_ID:-easy-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUNPOD_CYCLE_RUN_ID="${RUN_ID}"

CYCLE_DIR="${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}"
LOGS_DIR="${CYCLE_DIR}/logs"
REPORTS_DIR="${CYCLE_DIR}/reports"
mkdir -p "${LOGS_DIR}" "${REPORTS_DIR}"

SMOKE_EVENT_LOG="${RUNPOD_SMOKE_EVENT_LOG:-${LOGS_DIR}/full_smoke_events.log}"
SMOKE_SUMMARY_JSON="${RUNPOD_SMOKE_SUMMARY_JSON:-${REPORTS_DIR}/smoke_summary.json}"
mkdir -p "$(dirname "${SMOKE_EVENT_LOG}")" "$(dirname "${SMOKE_SUMMARY_JSON}")"

# Prefix every emitted line with an ISO-8601 UTC timestamp and persist to event log.
exec > >(awk '{ cmd="date -u +%Y-%m-%dT%H:%M:%SZ"; cmd | getline ts; close(cmd); print "[" ts "] " $0; fflush(); }' | tee -a "${SMOKE_EVENT_LOG}") 2>&1

START_EPOCH="$(date -u +%s)"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
AFTER_STOP_EPOCH="${START_EPOCH}"

cleanup_on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    echo "[runpod-full-train-easy-smoke] failed run_id=${RUN_ID}; best-effort terminate"
    RUNPOD_CYCLE_RUN_ID="${RUN_ID}" bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh" >/dev/null 2>&1 || true
  fi
}
trap cleanup_on_exit EXIT

# Smoke defaults tuned for quick reproducible cloud validation; caller may override any env.
export RUNPOD_FULL_TRAIN_EPOCHS="${RUNPOD_FULL_TRAIN_EPOCHS:-1}"
export RUNPOD_HF_DATASET_PATH_PREFIX="${RUNPOD_HF_DATASET_PATH_PREFIX:-validated_datasets}"
export RUNPOD_HF_DATASET_NAME="${RUNPOD_HF_DATASET_NAME:-elite_2025-11_game}"
export RUNPOD_HF_DATASET_SCHEMA_FILTER="${RUNPOD_HF_DATASET_SCHEMA_FILTER:-game_jsonl_runtime_splice_v1}"
export RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS="${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS:-5000}"
export RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME="${RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME:-auto}"
export RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT="${RUNPOD_FULL_TRAIN_RUNTIME_MIN_CONTEXT:-8}"
export RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET="${RUNPOD_FULL_TRAIN_RUNTIME_MIN_TARGET:-1}"
export RUNPOD_GPU_SAMPLE_SECONDS="${RUNPOD_GPU_SAMPLE_SECONDS:-2}"
export RUNPOD_GPU_TYPE_ID="${RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 5090}"
export RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"
export RUNPOD_FULL_TRAIN_NPROC_PER_NODE="${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${RUNPOD_GPU_COUNT}}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"

# Auto batch/worker selection is the smoke default.
if [[ -z "${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE+x}" ]]; then
  unset RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE || true
fi
if [[ -z "${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE+x}" ]]; then
  unset RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE || true
fi

echo "[runpod-full-train-easy-smoke] run_id=${RUN_ID}"
echo "[runpod-full-train-easy-smoke] event_log=${SMOKE_EVENT_LOG}"
echo "[runpod-full-train-easy-smoke] summary_json=${SMOKE_SUMMARY_JSON}"
echo "[runpod-full-train-easy-smoke] start_utc=${START_ISO}"
echo "[runpod-full-train-easy-smoke] hf_prefix=${RUNPOD_HF_DATASET_PATH_PREFIX}"
echo "[runpod-full-train-easy-smoke] hf_dataset_name=${RUNPOD_HF_DATASET_NAME}"
echo "[runpod-full-train-easy-smoke] hf_schema_filter=${RUNPOD_HF_DATASET_SCHEMA_FILTER}"
echo "[runpod-full-train-easy-smoke] epochs=${RUNPOD_FULL_TRAIN_EPOCHS}"
echo "[runpod-full-train-easy-smoke] gpu=${RUNPOD_GPU_TYPE_ID}"
echo "[runpod-full-train-easy-smoke] gpu_count=${RUNPOD_GPU_COUNT} nproc_per_node=${RUNPOD_FULL_TRAIN_NPROC_PER_NODE}"
echo "[runpod-full-train-easy-smoke] max_total_rows=${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS}"
echo "[runpod-full-train-easy-smoke] runtime_max_samples_per_game=${RUNPOD_FULL_TRAIN_RUNTIME_MAX_SAMPLES_PER_GAME}"
echo "[runpod-full-train-easy-smoke] batch_size_override=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<auto>}"
echo "[runpod-full-train-easy-smoke] num_workers_override=${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-<auto>}"

bash "${REPO_ROOT}/scripts/runpod_full_train_easy.sh"
AFTER_STOP_EPOCH="$(date -u +%s)"

PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

"${PY_BIN}" "${REPO_ROOT}/scripts/runpod_cycle_verify_full_hf_run.py" \
  --run-id "${RUN_ID}" \
  --output-json "${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/full_hf_verify_after_stop.json"

bash "${REPO_ROOT}/scripts/runpod_cycle_terminate.sh"

"${PY_BIN}" "${REPO_ROOT}/scripts/runpod_cycle_verify_full_hf_run.py" \
  --run-id "${RUN_ID}" \
  --require-terminated \
  --output-json "${REPO_ROOT}/artifacts/runpod_cycles/${RUN_ID}/full_hf_verify_after_terminate.json"

END_EPOCH="$(date -u +%s)"
END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PY_BIN}" - "${RUN_ID}" "${CYCLE_DIR}" "${START_EPOCH}" "${AFTER_STOP_EPOCH}" "${END_EPOCH}" "${SMOKE_SUMMARY_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_id, cycle_dir_s, start_epoch_s, after_stop_epoch_s, end_epoch_s, out_json_s = sys.argv[1:]
cycle_dir = Path(cycle_dir_s)
out_json = Path(out_json_s)
run_artifacts = cycle_dir / "collected" / "run_artifacts"

start_epoch = int(start_epoch_s)
after_stop_epoch = int(after_stop_epoch_s)
end_epoch = int(end_epoch_s)

metrics_path = run_artifacts / f"metrics_{run_id}.json"
progress_path = run_artifacts / f"train_progress_{run_id}.jsonl"
gpu_csv_path = run_artifacts / f"gpu_usage_samples_{run_id}.csv"
train_log_path = run_artifacts / f"train_stdout_{run_id}.log"

summary = {
    "run_id": run_id,
    "timing": {
        "start_epoch": start_epoch,
        "after_train_stop_seconds": max(after_stop_epoch - start_epoch, 0),
        "total_seconds": max(end_epoch - start_epoch, 0),
    },
    "artifacts": {
        "metrics_path": str(metrics_path),
        "progress_path": str(progress_path),
        "gpu_csv_path": str(gpu_csv_path),
        "train_log_path": str(train_log_path),
    },
}

if metrics_path.is_file():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics.get("history") or []
    summary["metrics"] = {
        "history_len": len(history),
        "last": history[-1] if history else {},
    }

if progress_path.is_file():
    line_count = 0
    last_event = {}
    with progress_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                last_event = json.loads(line)
            except Exception:
                pass
    summary["progress"] = {
        "line_count": line_count,
        "last_event": last_event,
    }

if gpu_csv_path.is_file():
    rows = []
    with gpu_csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            try:
                rows.append(
                    {
                        "ts": row[0],
                        "name": row[1],
                        "util_pct": float(row[2]),
                        "mem_used_mib": float(row[3]),
                        "mem_total_mib": float(row[4]),
                    }
                )
            except Exception:
                continue
    if rows:
        util = [r["util_pct"] for r in rows]
        mem = [r["mem_used_mib"] for r in rows]
        summary["gpu_samples"] = {
            "line_count": len(rows),
            "first_ts": rows[0]["ts"],
            "last_ts": rows[-1]["ts"],
            "util_avg_pct": sum(util) / len(util),
            "util_peak_pct": max(util),
            "mem_avg_mib": sum(mem) / len(mem),
            "mem_peak_mib": max(mem),
            "mem_total_mib": rows[0]["mem_total_mib"],
        }

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"smoke_summary_json": str(out_json), "run_id": run_id}, indent=2))
PY

echo "[runpod-full-train-easy-smoke] timing_summary run_id=${RUN_ID} start=${START_ISO} after_train_stop_s=$((AFTER_STOP_EPOCH-START_EPOCH)) total_s=$((END_EPOCH-START_EPOCH)) end=${END_ISO}"
echo "[runpod-full-train-easy-smoke] completed run_id=${RUN_ID}"

trap - EXIT
