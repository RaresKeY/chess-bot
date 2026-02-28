#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry/telemetry_common.sh"

KIND="${1:-}"
MESSAGE="${2:-}"

if [[ -z "${KIND}" || "${KIND}" == "--help" || "${KIND}" == "-h" ]]; then
  echo "Usage: bash scripts/telemetry_healthchecks_ping.sh <start|success|fail|log> [message]"
  exit 0
fi

telemetry_healthchecks_ping "${KIND}" "${MESSAGE}"
echo "[telemetry-healthchecks] kind=${KIND}"
