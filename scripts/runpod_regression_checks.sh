#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

RUN_LOCAL_SMOKE="${RUN_LOCAL_SMOKE:-0}"
RUN_CONNECTIVITY_TELEMETRY_CHECKS="${RUN_CONNECTIVITY_TELEMETRY_CHECKS:-1}"
RUN_CONNECTIVITY_PROVIDER="${RUN_CONNECTIVITY_PROVIDER:-runpod}"

echo "[runpod-regression] repo=${REPO_ROOT}"
echo "[runpod-regression] python=${PY_BIN}"
echo "[runpod-regression] run-local-smoke=${RUN_LOCAL_SMOKE}"
echo "[runpod-regression] run-connectivity-telemetry-checks=${RUN_CONNECTIVITY_TELEMETRY_CHECKS}"
echo "[runpod-regression] run-connectivity-provider=${RUN_CONNECTIVITY_PROVIDER}"

run_step() {
  local name="$1"
  shift
  echo
  echo "[runpod-regression] >>> ${name}"
  "$@"
}

cd "${REPO_ROOT}"

run_step "unit-tests (runpod helpers/scripts)" \
  "${PY_BIN}" -m unittest -v \
    tests.test_runpod_api_helpers \
    tests.test_runpod_local_smoke_script

run_step "cli-doctor (REST + GraphQL auth diagnostics)" \
  bash scripts/runpod_cli_doctor.sh

if [[ "${RUN_CONNECTIVITY_TELEMETRY_CHECKS}" == "1" ]]; then
  run_step "connectivity+telemetry checks (timeout guarded)" \
    bash scripts/cloud_connectivity_health_checks.sh --provider "${RUN_CONNECTIVITY_PROVIDER}"
else
  echo
  echo "[runpod-regression] skipping connectivity+telemetry category (set RUN_CONNECTIVITY_TELEMETRY_CHECKS=1)"
fi

if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  run_step "template-list (manual REST probe)" \
    "${PY_BIN}" scripts/runpod_provision.py template-list --limit 3
  run_step "gpu-search (manual GraphQL GPU probe)" \
    "${PY_BIN}" scripts/runpod_provision.py gpu-search --limit 5
else
  echo
  echo "[runpod-regression] skipping direct template-list/gpu-search probes (RUNPOD_API_KEY not set)"
  echo "[runpod-regression] note: doctor still ran and reported auth/key availability"
fi

if [[ "${RUN_LOCAL_SMOKE}" == "1" ]]; then
  run_step "local-runpod-smoke (docker)" \
    bash scripts/runpod_local_smoke_test.sh
else
  echo
  echo "[runpod-regression] skipping local smoke docker test (set RUN_LOCAL_SMOKE=1 to enable)"
fi

echo
echo "[runpod-regression] all enabled checks completed"
