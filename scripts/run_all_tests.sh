#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[run-all-tests] deprecated entrypoint; forwarding to scripts/test.sh" >&2
exec bash "${REPO_ROOT}/scripts/test.sh" "$@"
