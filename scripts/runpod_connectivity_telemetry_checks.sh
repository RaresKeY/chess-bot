#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CLOUD_CHECK_PROVIDER="runpod"
export CLOUD_CHECK_ENABLE_LIVE="${RUNPOD_ENABLE_LIVE_CONNECTIVITY_CHECKS:-${CLOUD_CHECK_ENABLE_LIVE:-0}}"
export CLOUD_CHECK_TIMEOUT_SECONDS="${RUNPOD_CONNECTIVITY_TIMEOUT_SECONDS:-${CLOUD_CHECK_TIMEOUT_SECONDS:-25}}"
export CLOUD_CHECK_RUN_ID="${RUNPOD_CYCLE_RUN_ID:-${CLOUD_CHECK_RUN_ID:-runpod-connectivity-$(date -u +%Y%m%dT%H%M%SZ)}}"

exec bash "${REPO_ROOT}/scripts/cloud_connectivity_health_checks.sh" --provider runpod "$@"
