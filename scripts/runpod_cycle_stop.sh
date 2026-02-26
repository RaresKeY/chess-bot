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

resp="$(
  curl -sS "${GRAPHQL_ENDPOINT}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"query\":\"mutation StopPod(\\$input: PodStopInput!) { podStop(input: \\$input) }\",\"variables\":{\"input\":{\"podId\":\"${POD_ID}\"}}}"
)"

printf '%s\n' "${resp}" | jq . | tee "${OUT_JSON}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Pod Stop" \
  "- Pod ID: \`${POD_ID}\`" \
  "- GraphQL endpoint: \`${GRAPHQL_ENDPOINT}\`" \
  "- Stop response: \`${OUT_JSON}\`" \
  ""

echo "[runpod-cycle-stop] pod_id=${POD_ID}"
