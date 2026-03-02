#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

SKIP_LOCAL_SMOKE="${VAST_NOAUTH_SKIP_LOCAL_SMOKE:-0}"

echo "[vast-noauth-checks] repo=${REPO_ROOT}"
echo "[vast-noauth-checks] python=${PY_BIN}"
echo "[vast-noauth-checks] skip_local_smoke=${SKIP_LOCAL_SMOKE}"

run_step() {
  local name="$1"
  shift
  echo
  echo "[vast-noauth-checks] >>> ${name}"
  "$@"
}

cd "${REPO_ROOT}"

run_step "unit-tests (vast suite)" \
  "${PY_BIN}" -m unittest discover -s tests -p "test_vast*.py" -v

run_step "cloud connectivity checks (vast provider local mode)" \
  bash scripts/cloud_connectivity_health_checks.sh --provider vast --no-live

if [[ "${SKIP_LOCAL_SMOKE}" == "1" ]]; then
  echo
  echo "[vast-noauth-checks] skipping local smoke training (VAST_NOAUTH_SKIP_LOCAL_SMOKE=1)"
else
  run_step "vast local smoke preset training" \
    bash scripts/vast_local_smoke_test.sh
fi

echo
echo "[vast-noauth-checks] completed"
