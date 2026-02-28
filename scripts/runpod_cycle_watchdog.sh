#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible alias to the central telemetry watchdog.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/telemetry_watchdog.sh" "$@"
