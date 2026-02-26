#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd curl

POD_ID="${RUNPOD_POD_ID:-$(runpod_cycle_pod_id "${PROVISION_JSON}")}"
GRAPHQL_ENDPOINT="${RUNPOD_GRAPHQL_ENDPOINT:-https://api.runpod.io/graphql}"
TOKEN="$(runpod_cycle_keyring_token "${PY_BIN}")"
[[ -n "${TOKEN}" ]] || { echo "[runpod-cycle-stop] missing RunPod API token in keyring" >&2; exit 1; }
[[ -n "${POD_ID}" ]] || { echo "[runpod-cycle-stop] missing pod id (set RUNPOD_POD_ID or provide provision json)" >&2; exit 1; }

mkdir -p "${CYCLE_DIR}"
OUT_JSON="${CYCLE_DIR}/stop_response.json"
POD_NAME="${RUNPOD_POD_NAME:-}"
if [[ -z "${POD_NAME}" && -f "${PROVISION_JSON}" ]]; then
  POD_NAME="$(runpod_cycle_pod_name "${PROVISION_JSON}" || true)"
fi

resp="$(
  curl -sS "${GRAPHQL_ENDPOINT}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    --data '{"query":"mutation StopPod($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }","variables":{"input":{"podId":"'"${POD_ID}"'"}}}'
)"

printf '%s\n' "${resp}" | jq . | tee "${OUT_JSON}"

STOP_STATE="STOP_REQUESTED"
STOP_NOTE="Requested podStop via GraphQL; pod storage may still incur charges until terminated"
if printf '%s\n' "${resp}" | jq -e 'has("errors") | not' >/dev/null 2>&1; then
  STOP_STATE="STOPPED"
  STOP_NOTE="podStop mutation returned without GraphQL errors; pod storage may still incur charges until terminated"
elif printf '%s\n' "${resp}" | jq -e '.errors' >/dev/null 2>&1; then
  STOP_STATE="STOP_ERROR"
  STOP_NOTE="podStop response included GraphQL errors; inspect stop_response.json"
fi

runpod_cycle_registry_record \
  "${REPO_ROOT}" \
  "runpod_cycle_stop.sh" \
  "stop" \
  "${STOP_STATE}" \
  "${POD_ID}" \
  "${RUN_ID}" \
  "${POD_NAME}" \
  "" \
  "" \
  "" \
  "${STOP_NOTE}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Pod Stop" \
  "- Pod ID: \`${POD_ID}\`" \
  "- GraphQL endpoint: \`${GRAPHQL_ENDPOINT}\`" \
  "- Stop response: \`${OUT_JSON}\`" \
  "- Tracked pods registry: \`$(runpod_cycle_registry_file "${REPO_ROOT}")\`" \
  "- Note: RunPod stop halts compute but can continue storage charges; use tracked terminate script to delete pods when done." \
  ""

echo "[runpod-cycle-stop] pod_id=${POD_ID}"
