#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry/telemetry_common.sh"

CHECKPOINT=""
STATE=""
NOTE=""

while (($#)); do
  case "$1" in
    --name) CHECKPOINT="${2:-}"; shift 2 ;;
    --state) STATE="${2:-}"; shift 2 ;;
    --note) NOTE="${2:-}"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/telemetry_checkpoint.sh --name <checkpoint> --state <pending|running|done|error> [--note <text>]"
      exit 0
      ;;
    *) echo "[telemetry-checkpoint] unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "${CHECKPOINT}" && -n "${STATE}" ]] || { echo "[telemetry-checkpoint] --name and --state are required" >&2; exit 1; }
telemetry_emit_checkpoint "${CHECKPOINT}" "${STATE}" "${NOTE}"
echo "[telemetry-checkpoint] checkpoint=${CHECKPOINT} state=${STATE}"
