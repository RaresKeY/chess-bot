#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry/telemetry_common.sh"

EVENT_NAME=""
STATUS=""
MESSAGE=""
EXTRA_JSON="{}"

while (($#)); do
  case "$1" in
    --event) EVENT_NAME="${2:-}"; shift 2 ;;
    --status) STATUS="${2:-}"; shift 2 ;;
    --message) MESSAGE="${2:-}"; shift 2 ;;
    --extra-json) EXTRA_JSON="${2:-{}}"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/telemetry_emit_event.sh --event <name> --status <ok|warn|error|info> [--message <text>] [--extra-json <json>]"
      exit 0
      ;;
    *) echo "[telemetry-emit-event] unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "${EVENT_NAME}" && -n "${STATUS}" ]] || { echo "[telemetry-emit-event] --event and --status are required" >&2; exit 1; }
telemetry_emit_event "${EVENT_NAME}" "${STATUS}" "${MESSAGE}" "${EXTRA_JSON}"
echo "[telemetry-emit-event] event=${EVENT_NAME} status=${STATUS}"
