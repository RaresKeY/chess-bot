#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/telemetry_control.sh status [--run-id <id>] [--json]
  bash scripts/telemetry_control.sh event --event <name> --status <ok|warn|error|info> [--message <text>] [--extra-json <json>]
  bash scripts/telemetry_control.sh checkpoint --name <checkpoint> --state <pending|running|done|error> [--note <text>]
  bash scripts/telemetry_control.sh watchdog [watchdog args...]
  bash scripts/telemetry_control.sh health <start|success|fail|log> [message]
EOF
}

CMD="${1:-}"
if [[ -z "${CMD}" || "${CMD}" == "--help" || "${CMD}" == "-h" ]]; then
  usage
  exit 0
fi
shift || true

case "${CMD}" in
  status)
    exec bash "${REPO_ROOT}/scripts/telemetry_status.sh" "$@"
    ;;
  event)
    exec bash "${REPO_ROOT}/scripts/telemetry_emit_event.sh" "$@"
    ;;
  checkpoint)
    exec bash "${REPO_ROOT}/scripts/telemetry_checkpoint.sh" "$@"
    ;;
  watchdog)
    exec bash "${REPO_ROOT}/scripts/telemetry_watchdog.sh" "$@"
    ;;
  health)
    exec bash "${REPO_ROOT}/scripts/telemetry_healthchecks_ping.sh" "$@"
    ;;
  *)
    echo "[telemetry-control] unknown command: ${CMD}" >&2
    usage >&2
    exit 1
    ;;
esac
