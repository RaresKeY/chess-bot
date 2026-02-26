#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

PROVISION_SCRIPT="${REPO_ROOT}/scripts/runpod_provision.py"

echo "[runpod-doctor] repo=${REPO_ROOT}"
echo "[runpod-doctor] python=${PY_BIN}"

if [[ ! -f "${PROVISION_SCRIPT}" ]]; then
  echo "[runpod-doctor] missing ${PROVISION_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  echo "[runpod-doctor] api-key-source=env RUNPOD_API_KEY"
else
  key_probe="$("${PY_BIN}" - <<'PY'
import sys
try:
    import keyring
except Exception:
    print("keyring-unavailable")
    raise SystemExit(0)
v = keyring.get_password("runpod", "RUNPOD_API_KEY")
if not v:
    print("keyring-missing")
else:
    print(f"keyring-found prefix={(v[:8] + '...') if len(v) >= 8 else v}")
PY
)"
  echo "[runpod-doctor] api-key-source=${key_probe}"
fi

echo "[runpod-doctor] checking REST template-list auth..."
if "${PY_BIN}" "${PROVISION_SCRIPT}" template-list --limit 1 >/tmp/runpod_doctor_templates.$$ 2>/tmp/runpod_doctor_templates_err.$$; then
  echo "[runpod-doctor] rest-template-list=ok"
else
  echo "[runpod-doctor] rest-template-list=error"
  sed -n '1,80p' /tmp/runpod_doctor_templates_err.$$ >&2 || true
fi

echo "[runpod-doctor] checking GraphQL gpu-search auth..."
if "${PY_BIN}" "${PROVISION_SCRIPT}" gpu-search --limit 1 >/tmp/runpod_doctor_gpu.$$ 2>/tmp/runpod_doctor_gpu_err.$$; then
  echo "[runpod-doctor] graphql-gpu-search=ok"
else
  echo "[runpod-doctor] graphql-gpu-search=error"
  sed -n '1,120p' /tmp/runpod_doctor_gpu_err.$$ >&2 || true
  echo "[runpod-doctor] hint: if template-list works but gpu-search fails with 403, the key likely lacks GraphQL access/scopes; use RunPod UI launch or provision with explicit --gpu-type-id."
fi

rm -f /tmp/runpod_doctor_templates.$$ /tmp/runpod_doctor_templates_err.$$ /tmp/runpod_doctor_gpu.$$ /tmp/runpod_doctor_gpu_err.$$ || true
