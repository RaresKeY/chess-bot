#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

runpod_cycle_require_cmd jq

LOCAL_COLLECT_DIR="${RUNPOD_LOCAL_COLLECT_DIR:-${CYCLE_DIR}/collected}"
MODEL_PATH="${RUNPOD_LOCAL_MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(find "${LOCAL_COLLECT_DIR}/run_artifacts" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1)"
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "[runpod-cycle-local-validate] model not found under ${LOCAL_COLLECT_DIR}/run_artifacts" >&2
  exit 1
fi

VALIDATE_DIR="${CYCLE_DIR}/local_validation"
mkdir -p "${VALIDATE_DIR}"
OUT_TXT="${VALIDATE_DIR}/infer_move_output.txt"
INFER_CONTEXT="${RUNPOD_INFER_CONTEXT:-e2e4 e7e5 g1f3}"

(
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  "${PY_BIN}" scripts/infer_move.py \
    --model "${MODEL_PATH}" \
    --context "${INFER_CONTEXT}" \
    --winner-side W \
    --device cpu
) | tee "${OUT_TXT}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Local Model Validation" \
  "- Model path: \`${MODEL_PATH}\`" \
  "- Inference context: \`${INFER_CONTEXT}\`" \
  "- Output log: \`${OUT_TXT}\`" \
  ""

echo "[runpod-cycle-local-validate] output=${OUT_TXT}"
