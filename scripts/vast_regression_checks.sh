#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

echo "[vast-regression] repo=${REPO_ROOT}"
echo "[vast-regression] python=${PY_BIN}"

run_step() {
  local name="$1"
  shift
  echo
  echo "[vast-regression] >>> ${name}"
  "$@"
}

cd "${REPO_ROOT}"

run_step "unit-tests (vast helpers/scripts)" \
  "${PY_BIN}" -m unittest -v \
    tests.test_vast_api_helpers \
    tests.test_vast_cycle_scripts

run_step "cli-doctor (auth diagnostics)" \
  bash scripts/vast_cli_doctor.sh

echo
if [[ -n "${VAST_API_KEY:-}" ]]; then
  run_step "offer-search (manual probe)" \
    "${PY_BIN}" scripts/vast_provision.py offer-search --limit 3
  run_step "instance-list (manual probe)" \
    "${PY_BIN}" scripts/vast_provision.py instance-list
else
  echo "[vast-regression] skipping direct API probes (VAST_API_KEY not set)"
  echo "[vast-regression] note: doctor still ran and reported auth/key availability"
fi

echo

echo "[vast-regression] all enabled checks completed"
