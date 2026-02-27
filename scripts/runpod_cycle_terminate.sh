#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
RUN_ID="$(runpod_cycle_run_id)"
CYCLE_DIR="$(runpod_cycle_dir "${REPO_ROOT}" "${RUN_ID}")"
PROVISION_JSON="$(runpod_cycle_provision_json "${REPO_ROOT}" "${RUN_ID}")"
REPORT_MD="$(runpod_cycle_report_md "${REPO_ROOT}" "${RUN_ID}")"
REST_ENDPOINT="${RUNPOD_REST_ENDPOINT:-https://rest.runpod.io/v1}"

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd curl

POD_ID="${RUNPOD_POD_ID:-$(runpod_cycle_pod_id "${PROVISION_JSON}")}"
TOKEN="$(runpod_cycle_api_token "${PY_BIN}" "${REPO_ROOT}")"
[[ -n "${TOKEN}" ]] || { echo "[runpod-cycle-terminate] missing RunPod API token (checked RUNPOD_API_KEY, keyring, and .env fallback)" >&2; exit 1; }
[[ -n "${POD_ID}" ]] || { echo "[runpod-cycle-terminate] missing pod id" >&2; exit 1; }

mkdir -p "${CYCLE_DIR}"
OUT_JSON="${CYCLE_DIR}/terminate_response.json"
POD_NAME="${RUNPOD_POD_NAME:-}"
if [[ -z "${POD_NAME}" && -f "${PROVISION_JSON}" ]]; then
  POD_NAME="$(runpod_cycle_pod_name "${PROVISION_JSON}" || true)"
fi

body_file="$(mktemp /tmp/runpod_terminate_body.XXXXXX)"
http_code="$(
  curl -sS \
    -o "${body_file}" \
    -w "%{http_code}" \
    -X DELETE \
    "${REST_ENDPOINT%/}/pods/${POD_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json"
)"
resp_body="$(cat "${body_file}" || true)"
rm -f "${body_file}"

jq -nc \
  --arg pod_id "${POD_ID}" \
  --arg http_code "${http_code}" \
  --arg response_body "${resp_body}" \
  '{pod_id:$pod_id,http_code:$http_code,response_body:$response_body}' \
  | tee "${OUT_JSON}" >/dev/null

TERM_STATE="TERMINATE_ERROR"
TERM_NOTE="DELETE /pods/{id} failed; inspect terminate_response.json"
if [[ "${http_code}" == 2* ]]; then
  TERM_STATE="TERMINATED"
  TERM_NOTE="Deleted via REST DELETE /pods/{id}"
elif [[ "${http_code}" == "404" ]]; then
  body_lc="$(printf '%s' "${resp_body}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${body_lc}" == *"pod"* && "${body_lc}" == *"not found"* ]]; then
    TERM_STATE="TERMINATED"
    TERM_NOTE="DELETE returned 404 pod not found; treated as already terminated"
  fi
fi

runpod_cycle_registry_record \
  "${REPO_ROOT}" \
  "runpod_cycle_terminate.sh" \
  "terminate" \
  "${TERM_STATE}" \
  "${POD_ID}" \
  "${RUN_ID}" \
  "${POD_NAME}" \
  "" \
  "" \
  "" \
  "${TERM_NOTE}"

runpod_cycle_append_report "${REPORT_MD}" \
  "## Pod Termination" \
  "- Pod ID: \`${POD_ID}\`" \
  "- REST endpoint: \`${REST_ENDPOINT}\`" \
  "- HTTP code: \`${http_code}\`" \
  "- Terminate response: \`${OUT_JSON}\`" \
  "- Tracked pods registry: \`$(runpod_cycle_registry_file "${REPO_ROOT}")\`" \
  ""

echo "[runpod-cycle-terminate] pod_id=${POD_ID} http_code=${http_code}"
if [[ "${TERM_STATE}" != "TERMINATED" ]]; then
  exit 1
fi
