#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

if command -v ssh >/dev/null 2>&1 && command -v ssh-keygen >/dev/null 2>&1; then
  echo "[container-openssh] already available: ssh + ssh-keygen"
  exit 0
fi

if runpod_cycle_try_install_openssh_client; then
  runpod_cycle_require_cmd ssh
  runpod_cycle_require_cmd ssh-keygen
  echo "[container-openssh] installed openssh-client"
  exit 0
fi

echo "[container-openssh] unable to install openssh-client automatically in this environment" >&2
echo "[container-openssh] install openssh-client manually, then rerun." >&2
exit 1
