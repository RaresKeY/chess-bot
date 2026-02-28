#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cloud_checks/common.sh"

PROVIDER="${CLOUD_CHECK_PROVIDER:-runpod}"
LIVE_CHECKS="${CLOUD_CHECK_ENABLE_LIVE:-0}"
TIMEOUT_SECONDS="${CLOUD_CHECK_TIMEOUT_SECONDS:-25}"
CLOUD_CHECK_RUN_ID="${CLOUD_CHECK_RUN_ID:-cloud-check-${PROVIDER}-$(date -u +%Y%m%dT%H%M%SZ)}"
export CLOUD_CHECK_RUN_ID

while (($#)); do
  case "$1" in
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --live) LIVE_CHECKS=1; shift ;;
    --no-live) LIVE_CHECKS=0; shift ;;
    --timeout-seconds) TIMEOUT_SECONDS="${2:-}"; shift 2 ;;
    --run-id) CLOUD_CHECK_RUN_ID="${2:-}"; export CLOUD_CHECK_RUN_ID; shift 2 ;;
    --help|-h)
      cat <<'EOF'
Usage:
  bash scripts/cloud_connectivity_health_checks.sh [--provider runpod|vast] [--live] [--timeout-seconds N] [--run-id ID]

Env:
  CLOUD_CHECK_PROVIDER=runpod
  CLOUD_CHECK_ENABLE_LIVE=0
  CLOUD_CHECK_TIMEOUT_SECONDS=25
  CLOUD_CHECK_RUN_ID=cloud-check-<provider>-<timestamp>
EOF
      exit 0
      ;;
    *)
      echo "[cloud-check] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v timeout >/dev/null 2>&1; then
  echo "[cloud-check] timeout command is required" >&2
  exit 1
fi

REPO_ROOT="$(cloud_check_repo_root)"
PY_BIN="$(cloud_check_py_bin "${REPO_ROOT}")"
PROVIDER_FILE="${REPO_ROOT}/scripts/cloud_checks/providers/${PROVIDER}.sh"
[[ -f "${PROVIDER_FILE}" ]] || { echo "[cloud-check] unsupported provider: ${PROVIDER}" >&2; exit 1; }
source "${PROVIDER_FILE}"

LOCAL_FUNC="${PROVIDER}_provider_local_checks"
LIVE_FUNC="${PROVIDER}_provider_live_checks"
declare -f "${LOCAL_FUNC}" >/dev/null 2>&1 || { echo "[cloud-check] missing provider local function: ${LOCAL_FUNC}" >&2; exit 1; }
declare -f "${LIVE_FUNC}" >/dev/null 2>&1 || { echo "[cloud-check] missing provider live function: ${LIVE_FUNC}" >&2; exit 1; }

echo "[cloud-check] provider=${PROVIDER}"
echo "[cloud-check] repo=${REPO_ROOT}"
echo "[cloud-check] python=${PY_BIN}"
echo "[cloud-check] timeout_seconds=${TIMEOUT_SECONDS}"
echo "[cloud-check] live_checks=${LIVE_CHECKS}"
echo "[cloud-check] run_id=${CLOUD_CHECK_RUN_ID}"

cloud_check_emit_event "${CLOUD_CHECK_RUN_ID}" "cloud_connectivity_check_start" "info" "cloud connectivity checks started" "{\"provider\":\"${PROVIDER}\"}" "${REPO_ROOT}"
cloud_check_emit_checkpoint "${CLOUD_CHECK_RUN_ID}" "cloud_connectivity_checks" "running" "started provider=${PROVIDER}" "${REPO_ROOT}"

"${LOCAL_FUNC}" "${REPO_ROOT}" "${PY_BIN}" "${TIMEOUT_SECONDS}"

if [[ "${LIVE_CHECKS}" == "1" ]]; then
  "${LIVE_FUNC}" "${REPO_ROOT}" "${PY_BIN}" "${TIMEOUT_SECONDS}"
else
  echo
  echo "[cloud-check] skipping live checks (use --live or CLOUD_CHECK_ENABLE_LIVE=1)"
fi

cloud_check_emit_checkpoint "${CLOUD_CHECK_RUN_ID}" "cloud_connectivity_checks" "done" "completed provider=${PROVIDER}" "${REPO_ROOT}"
cloud_check_emit_event "${CLOUD_CHECK_RUN_ID}" "cloud_connectivity_check_complete" "ok" "cloud connectivity checks completed" "{\"provider\":\"${PROVIDER}\",\"live_checks\":${LIVE_CHECKS}}" "${REPO_ROOT}"

echo
echo "[cloud-check] checks completed"
