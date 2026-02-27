#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

echo "[run-all-tests] repo=${REPO_ROOT}"
echo "[run-all-tests] python=${PY_BIN}"

cd "${REPO_ROOT}"
"${PY_BIN}" -m pytest -q "$@"
