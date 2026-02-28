#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
LOGS_DIR="$(runpod_cycle_logs_dir "${REPO_ROOT}" "${RUN_ID}")"
mkdir -p "${LOGS_DIR}"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd rsync
runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"

SSH_HOST="$(runpod_cycle_ssh_host "${PROVISION_JSON}")"
SSH_PORT="$(runpod_cycle_ssh_port "${PROVISION_JSON}")"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_USER="$(runpod_cycle_ssh_user)"
SSH_CONNECT_TIMEOUT="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-15}"
SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"

REMOTE_REPO_DIR="${RUNPOD_REMOTE_REPO_DIR:-$(runpod_cycle_remote_repo_dir "${PROVISION_JSON}")}"
REMOTE_RUN_DIR="${REMOTE_REPO_DIR}/artifacts/runpod_cycles/${RUN_ID}"
REMOTE_TIMING_LOG="${REMOTE_REPO_DIR}/artifacts/timings/runpod_phase_times.jsonl"
LOCAL_COLLECT_DIR="${RUNPOD_LOCAL_COLLECT_DIR:-${CYCLE_DIR}/collected}"
LOCAL_AUTO_LOGS_DIR="${LOCAL_COLLECT_DIR}/logs_auto"
RSYNC_ARTIFACTS_LOG="${LOGS_DIR}/collect_rsync_run_artifacts.log"
RSYNC_TIMING_LOG="${LOGS_DIR}/collect_rsync_timing.log"
AUTO_REMOTE_STATE_LOG="${LOCAL_AUTO_LOGS_DIR}/remote_state_snapshot.txt"
AUTO_INDEX_SUMMARY_JSON="${LOCAL_AUTO_LOGS_DIR}/train_log_indexing_summary.json"
AUTO_COLLECT_MANIFEST_JSON="${LOCAL_AUTO_LOGS_DIR}/collection_manifest.json"

mkdir -p "${LOCAL_COLLECT_DIR}" "${LOCAL_AUTO_LOGS_DIR}"

printf -v RSYNC_SSH 'ssh -i %q -p %q -o BatchMode=yes -o ConnectTimeout=%q -o IdentitiesOnly=yes -o AddKeysToAgent=no -o IdentityAgent=none -o StrictHostKeyChecking=%q -o UserKnownHostsFile=%q' \
  "${SSH_KEY}" "${SSH_PORT}" "${SSH_CONNECT_TIMEOUT}" "${SSH_HOST_KEY_CHECKING}" "${SSH_KNOWN_HOSTS_FILE}"

{
  printf '[runpod-cycle-collect] ssh=%s@%s:%s\n' "${SSH_USER}" "${SSH_HOST}" "${SSH_PORT}"
  printf '[runpod-cycle-collect] remote_run_dir=%s\n' "${REMOTE_RUN_DIR}"
  printf '[runpod-cycle-collect] local_collect_dir=%s/run_artifacts\n' "${LOCAL_COLLECT_DIR}"
} > "${RSYNC_ARTIFACTS_LOG}"

rsync -az --info=stats1 --progress -e "${RSYNC_SSH}" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_RUN_DIR}/" "${LOCAL_COLLECT_DIR}/run_artifacts/" \
  2>&1 | tee -a "${RSYNC_ARTIFACTS_LOG}"

rsync -az --info=stats1 -e "${RSYNC_SSH}" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_TIMING_LOG}" "${LOCAL_COLLECT_DIR}/runpod_phase_times.jsonl" \
  2>&1 | tee "${RSYNC_TIMING_LOG}" || true

# Best-effort remote runtime snapshot to aid debugging without needing interactive SSH.
ssh -i "${SSH_KEY}" -p "${SSH_PORT}" \
  -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" -o IdentitiesOnly=yes \
  -o AddKeysToAgent=no -o IdentityAgent=none \
  -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" \
  -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
  "${SSH_USER}@${SSH_HOST}" \
  "RUN_ID='${RUN_ID}' REMOTE_RUN_DIR='${REMOTE_RUN_DIR}' /bin/bash -s" \
  >"${AUTO_REMOTE_STATE_LOG}" 2>&1 <<'EOF_REMOTE_STATE' || true
set -Eeuo pipefail
echo "ts_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname 2>/dev/null || true)"
echo "kernel=$(uname -a 2>/dev/null || true)"
echo "run_dir=${REMOTE_RUN_DIR}"
echo "--- nvidia_smi ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true
echo "--- top_processes ---"
ps -eo pid,ppid,etimes,pcpu,pmem,stat,cmd --sort=-pcpu | head -n 120 2>/dev/null || true
echo "--- service_processes ---"
pgrep -a -f 'sshd|jupyter|uvicorn|otelcol|train_baseline|torchrun|python' 2>/dev/null || true
echo "--- train_logs_tail ---"
for f in "${REMOTE_RUN_DIR}"/train_stdout_*.log "${REMOTE_RUN_DIR}"/manual_*/*.log "${REMOTE_RUN_DIR}"/manual_bench/*/*.log; do
  [[ -f "${f}" ]] || continue
  echo ">>> ${f}"
  tail -n 120 "${f}" 2>/dev/null || true
done
echo "--- progress_logs_tail ---"
for f in "${REMOTE_RUN_DIR}"/train_progress_*.jsonl "${REMOTE_RUN_DIR}"/manual_*/*.jsonl "${REMOTE_RUN_DIR}"/manual_bench/*/*.jsonl; do
  [[ -f "${f}" ]] || continue
  echo ">>> ${f}"
  tail -n 80 "${f}" 2>/dev/null || true
done
EOF_REMOTE_STATE

python3 - "${LOCAL_COLLECT_DIR}/run_artifacts" "${AUTO_INDEX_SUMMARY_JSON}" <<'PY'
import json
import re
import sys
from pathlib import Path

artifacts_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
patterns = [
    ("indexing", re.compile(r"\b(index|indexing|build(?:ing)?\s+index)\b", re.IGNORECASE)),
    ("vocab", re.compile(r"\bvocab(?:ulary)?\b", re.IGNORECASE)),
    ("row_count", re.compile(r"\b(row(?:s)?\s*count|counting\s+rows)\b", re.IGNORECASE)),
    ("cache_miss", re.compile(r"\bcache\s+miss\b", re.IGNORECASE)),
]
log_files = sorted([p for p in artifacts_dir.rglob("*") if p.is_file() and p.suffix in {".log", ".jsonl"}])
counts = {k: 0 for k, _ in patterns}
hits = []
for path in log_files:
    rel = str(path.relative_to(artifacts_dir))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                for key, rx in patterns:
                    if rx.search(line):
                        counts[key] += 1
                        if len(hits) < 200:
                            hits.append({"file": rel, "line": lineno, "kind": key, "text": line.strip()[:300]})
    except Exception:
        continue

payload = {
    "artifacts_dir": str(artifacts_dir),
    "log_files_scanned": len(log_files),
    "keyword_counts": counts,
    "indexing_detected": bool(counts["indexing"] or counts["vocab"] or counts["row_count"]),
    "cache_miss_detected": bool(counts["cache_miss"]),
    "sample_hits": hits,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=True))
PY

python3 - "${LOCAL_COLLECT_DIR}" "${AUTO_COLLECT_MANIFEST_JSON}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    rel = str(path.relative_to(root))
    rows.append({"path": rel, "size_bytes": int(path.stat().st_size)})
payload = {"collected_root": str(root), "files": rows}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"files": len(rows)}, ensure_ascii=True))
PY

runpod_cycle_append_report "${REPORT_MD}" \
  "## Artifact Collection" \
  "- Remote run dir: \`${REMOTE_RUN_DIR}\`" \
  "- Local collect dir: \`${LOCAL_COLLECT_DIR}\`" \
  "- SSH endpoint used: \`${SSH_USER}@${SSH_HOST}:${SSH_PORT}\`" \
  "- Timing log (best effort): \`${LOCAL_COLLECT_DIR}/runpod_phase_times.jsonl\`" \
  "- Auto remote state snapshot: \`${AUTO_REMOTE_STATE_LOG}\`" \
  "- Auto indexing summary: \`${AUTO_INDEX_SUMMARY_JSON}\`" \
  "- Auto collection manifest: \`${AUTO_COLLECT_MANIFEST_JSON}\`" \
  "- Rsync artifact log: \`${RSYNC_ARTIFACTS_LOG}\`" \
  "- Rsync timing-log transfer log: \`${RSYNC_TIMING_LOG}\`" \
  ""

echo "[runpod-cycle-collect] local_collect_dir=${LOCAL_COLLECT_DIR}"
