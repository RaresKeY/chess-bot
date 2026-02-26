#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
REGISTRY_FILE="$(runpod_cycle_registry_file "${REPO_ROOT}")"
REST_ENDPOINT="${RUNPOD_REST_ENDPOINT:-https://rest.runpod.io/v1}"
CONFIRM="${RUNPOD_CONFIRM_TERMINATE_ALL:-}"

if [[ "${1:-}" == "--yes" ]]; then
  CONFIRM="YES"
elif [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/runpod_cycle_terminate_all_tracked.sh [--yes]

Safely terminate (delete) all pods recorded by our RunPod cycle scripts that are
not already marked TERMINATED in the tracked pods registry.

Safety:
  - Only acts on pod IDs in config/runpod_tracked_pods.jsonl (or RUNPOD_TRACKED_PODS_FILE)
  - Requires explicit confirmation via --yes or RUNPOD_CONFIRM_TERMINATE_ALL=YES

Notes:
  - This terminates pods (REST DELETE /pods/<id>) to avoid ongoing storage charges.
  - "stop" alone can still incur storage costs on RunPod.
EOF
  exit 0
fi

runpod_cycle_require_cmd jq
runpod_cycle_require_cmd curl

if [[ ! -f "${REGISTRY_FILE}" ]]; then
  echo "[runpod-cycle-terminate-all] no tracked pods registry found: ${REGISTRY_FILE}"
  exit 0
fi

mapfile -t CANDIDATE_LINES < <(
  jq -rsc '
    [ .[] | select(type == "object" and (.pod_id? // "") != "") ] as $events
    | (reduce $events[] as $e ({}; .[$e.pod_id] = $e)) as $latest
    | ($latest | to_entries | map(.value)
       | map(select((.state // "") != "TERMINATED"))
       | sort_by(.ts_utc // "", .pod_id))
    | .[]
    | [
        (.pod_id // ""),
        (.state // ""),
        (.pod_name // ""),
        (.run_id // ""),
        (.ts_utc // "")
      ] | @tsv
  ' "${REGISTRY_FILE}"
)

if [[ "${#CANDIDATE_LINES[@]}" -eq 0 ]]; then
  echo "[runpod-cycle-terminate-all] no tracked pods require termination"
  exit 0
fi

echo "[runpod-cycle-terminate-all] registry=${REGISTRY_FILE}"
echo "[runpod-cycle-terminate-all] candidates (tracked, latest state != TERMINATED):"
for line in "${CANDIDATE_LINES[@]}"; do
  IFS=$'\t' read -r pod_id state pod_name run_id ts_utc <<<"${line}"
  echo "  - pod_id=${pod_id} state=${state} pod_name=${pod_name:-<unknown>} run_id=${run_id:-<none>} last_seen=${ts_utc:-<unknown>}"
done

if [[ "${CONFIRM}" != "YES" ]]; then
  echo "[runpod-cycle-terminate-all] refusing to terminate without explicit confirmation"
  echo "[runpod-cycle-terminate-all] re-run with '--yes' or export RUNPOD_CONFIRM_TERMINATE_ALL=YES"
  exit 2
fi

TOKEN="$(runpod_cycle_keyring_token "${PY_BIN}")"
[[ -n "${TOKEN}" ]] || { echo "[runpod-cycle-terminate-all] missing RunPod API token in keyring" >&2; exit 1; }

LOG_DIR="${REPO_ROOT}/artifacts/reports"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_JSONL="${LOG_DIR}/runpod_terminate_all_tracked_${RUN_TS}.jsonl"

failures=0
terminated=0

for line in "${CANDIDATE_LINES[@]}"; do
  IFS=$'\t' read -r pod_id state pod_name run_id ts_utc <<<"${line}"
  [[ -n "${pod_id}" ]] || continue
  echo "[runpod-cycle-terminate-all] deleting pod_id=${pod_id} (prev_state=${state})"

  body_file="$(mktemp /tmp/runpod_terminate_body.XXXXXX)"
  http_code="$(
    curl -sS \
      -o "${body_file}" \
      -w "%{http_code}" \
      -X DELETE \
      "${REST_ENDPOINT%/}/pods/${pod_id}" \
      -H "Authorization: Bearer ${TOKEN}" \
      -H "Content-Type: application/json"
  )"
  resp_body="$(cat "${body_file}" || true)"
  rm -f "${body_file}"

  jq -nc \
    --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg pod_id "${pod_id}" \
    --arg pod_name "${pod_name}" \
    --arg run_id "${run_id}" \
    --arg prev_state "${state}" \
    --arg http_code "${http_code}" \
    --arg response_body "${resp_body}" \
    '{ts_utc:$ts_utc,pod_id:$pod_id,pod_name:$pod_name,run_id:$run_id,prev_state:$prev_state,http_code:$http_code,response_body:$response_body}' >> "${LOG_JSONL}"

  if [[ "${http_code}" == 2* ]]; then
    terminated=$((terminated + 1))
    runpod_cycle_registry_record \
      "${REPO_ROOT}" \
      "runpod_cycle_terminate_all_tracked.sh" \
      "terminate" \
      "TERMINATED" \
      "${pod_id}" \
      "${run_id}" \
      "${pod_name}" \
      "" \
      "" \
      "" \
      "Deleted via REST DELETE /pods/{id}; log=${LOG_JSONL}"
  else
    failures=$((failures + 1))
    runpod_cycle_registry_record \
      "${REPO_ROOT}" \
      "runpod_cycle_terminate_all_tracked.sh" \
      "terminate_error" \
      "${state}" \
      "${pod_id}" \
      "${run_id}" \
      "${pod_name}" \
      "" \
      "" \
      "" \
      "DELETE failed http_code=${http_code}; log=${LOG_JSONL}"
    echo "[runpod-cycle-terminate-all] delete failed pod_id=${pod_id} http_code=${http_code}" >&2
  fi
done

echo "[runpod-cycle-terminate-all] terminated=${terminated} failures=${failures} log=${LOG_JSONL}"
if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
