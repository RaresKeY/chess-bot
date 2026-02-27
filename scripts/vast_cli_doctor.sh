#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

PROVISION_SCRIPT="${REPO_ROOT}/scripts/vast_provision.py"

echo "[vast-doctor] repo=${REPO_ROOT}"
echo "[vast-doctor] python=${PY_BIN}"

if [[ ! -f "${PROVISION_SCRIPT}" ]]; then
  echo "[vast-doctor] missing ${PROVISION_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${VAST_API_KEY:-}" ]]; then
  echo "[vast-doctor] api-key-source=env VAST_API_KEY"
else
  key_probe="$("${PY_BIN}" - <<'PY'
import sys
try:
    import keyring
except Exception:
    print("keyring-unavailable")
    raise SystemExit(0)
try:
    v = keyring.get_password("vast", "VAST_API_KEY")
except Exception as exc:
    print(f"keyring-error:{exc.__class__.__name__}")
    raise SystemExit(0)
if not v:
    print("keyring-missing")
else:
    print(f"keyring-found prefix={(v[:8] + '...') if len(v) >= 8 else v}")
PY
)"
  echo "[vast-doctor] api-key-source=${key_probe}"
fi

echo "[vast-doctor] checking offer-search auth..."
if "${PY_BIN}" "${PROVISION_SCRIPT}" offer-search --limit 1 >/tmp/vast_doctor_offers.$$ 2>/tmp/vast_doctor_offers_err.$$; then
  echo "[vast-doctor] offer-search=ok"
else
  echo "[vast-doctor] offer-search=error"
  sed -n '1,120p' /tmp/vast_doctor_offers_err.$$ >&2 || true
fi

echo "[vast-doctor] checking instance-list auth..."
if "${PY_BIN}" "${PROVISION_SCRIPT}" instance-list >/tmp/vast_doctor_instances.$$ 2>/tmp/vast_doctor_instances_err.$$; then
  echo "[vast-doctor] instance-list=ok"
else
  echo "[vast-doctor] instance-list=error"
  sed -n '1,120p' /tmp/vast_doctor_instances_err.$$ >&2 || true
fi

rm -f /tmp/vast_doctor_offers.$$ /tmp/vast_doctor_offers_err.$$ /tmp/vast_doctor_instances.$$ /tmp/vast_doctor_instances_err.$$ || true
