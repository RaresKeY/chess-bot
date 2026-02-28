#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="${RUNPOD_HEALTHCHECKS_URL:-${HEALTHCHECKS_URL:-}}"
KIND="${1:-}"
MESSAGE="${2:-}"

if [[ -z "${BASE_URL}" ]]; then
  echo "[healthchecks] no URL configured (RUNPOD_HEALTHCHECKS_URL/HEALTHCHECKS_URL)"
  exit 0
fi

if [[ -z "${KIND}" || "${KIND}" == "--help" || "${KIND}" == "-h" ]]; then
  echo "Usage: bash deploy/runpod_cloud_training/healthchecks_ping.sh <start|success|fail|log> [message]"
  exit 0
fi

case "${KIND}" in
  start) URL="${BASE_URL%/}/start" ;;
  success) URL="${BASE_URL%/}" ;;
  fail) URL="${BASE_URL%/}/fail" ;;
  log) URL="${BASE_URL%/}/log" ;;
  *) URL="${BASE_URL%/}" ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  echo "[healthchecks] curl is required"
  exit 1
fi

curl -fsS -m 10 -X POST --data-raw "${MESSAGE}" "${URL}" >/dev/null
echo "[healthchecks] ping kind=${KIND} url=${URL}"
