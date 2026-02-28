#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_cycle_common.sh"

REPO_ROOT="$(runpod_cycle_repo_root)"
PY_BIN="$(runpod_cycle_py_bin "${REPO_ROOT}")"
REGISTRY_FILE="$(runpod_cycle_registry_file "${REPO_ROOT}")"
REST_ENDPOINT="${RUNPOD_REST_ENDPOINT:-https://rest.runpod.io/v1}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${REPO_ROOT}/artifacts/reports"
REPORT_JSON="${REPORT_DIR}/runpod_active_pods_full_status_${RUN_TS}.json"

INCLUDE_API=1
INCLUDE_SSH=1
WRITE_REPORT=1
RUNNING_ONLY=0
SSH_TIMEOUT_SECONDS="${RUNPOD_STATUS_SSH_TIMEOUT_SECONDS:-12}"
CONNECT_TIMEOUT_SECONDS="${RUNPOD_SSH_CONNECT_TIMEOUT_SECONDS:-10}"

usage() {
  cat <<'EOF'
Usage: bash scripts/runpod_active_pods_full_status.sh [options]

Build a full local status snapshot for tracked RunPod pods that are not locally
marked TERMINATED.

Output:
  - Prints a JSON summary to stdout
  - Writes the same JSON to artifacts/reports/ by default

Options:
  --no-api        Skip RunPod REST pod-status lookups
  --no-ssh        Skip SSH host probes
  --running-only  Include only pods with desiredStatus=RUNNING (requires API)
  --no-write      Do not write artifacts/reports JSON file
  --help          Show this help
EOF
}

while (($#)); do
  case "$1" in
    --no-api) INCLUDE_API=0 ;;
    --no-ssh) INCLUDE_SSH=0 ;;
    --running-only) RUNNING_ONLY=1 ;;
    --no-write) WRITE_REPORT=0 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "[runpod-active-status] unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

runpod_cycle_require_cmd jq
if [[ "${INCLUDE_API}" == "1" ]]; then
  runpod_cycle_require_cmd curl
fi
if [[ "${INCLUDE_SSH}" == "1" ]]; then
  runpod_cycle_require_cmd ssh
  runpod_cycle_prepare_ssh_client_files "${REPO_ROOT}"
fi

if [[ ! -f "${REGISTRY_FILE}" ]]; then
  out="$(jq -cn \
    --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg registry_file "${REGISTRY_FILE}" \
    '{ts_utc:$ts_utc, registry_file:$registry_file, pod_count:0, pods:[]}')"
  printf '%s\n' "${out}"
  exit 0
fi

mapfile -t POD_LINES < <(
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
        (.public_ip // ""),
        (.ssh_host // ""),
        (.ssh_port // ""),
        (.ts_utc // "")
      ] | @tsv
  ' "${REGISTRY_FILE}"
)

if [[ "${#POD_LINES[@]}" -eq 0 ]]; then
  out="$(jq -cn \
    --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg registry_file "${REGISTRY_FILE}" \
    '{ts_utc:$ts_utc, registry_file:$registry_file, pod_count:0, pods:[]}')"
  printf '%s\n' "${out}"
  if [[ "${WRITE_REPORT}" == "1" ]]; then
    mkdir -p "${REPORT_DIR}"
    printf '%s\n' "${out}" > "${REPORT_JSON}"
  fi
  exit 0
fi

TOKEN=""
if [[ "${INCLUDE_API}" == "1" ]]; then
  TOKEN="$(runpod_cycle_api_token "${PY_BIN}" "${REPO_ROOT}")"
  if [[ -z "${TOKEN}" ]]; then
    echo "[runpod-active-status] warning: missing RunPod API token; continuing with local/ssh only" >&2
    INCLUDE_API=0
    if [[ "${RUNNING_ONLY}" == "1" ]]; then
      echo "[runpod-active-status] --running-only requires API token/lookups" >&2
      exit 1
    fi
  fi
fi

SSH_USER="$(runpod_cycle_ssh_user)"
SSH_KEY="$(runpod_cycle_ssh_key)"
SSH_HOST_KEY_CHECKING="$(runpod_cycle_ssh_host_key_checking)"
SSH_KNOWN_HOSTS_FILE="$(runpod_cycle_ssh_known_hosts_file "${REPO_ROOT}")"

pods_json="[]"
for line in "${POD_LINES[@]}"; do
  IFS=$'\t' read -r pod_id local_state pod_name run_id local_public_ip local_ssh_host local_ssh_port last_seen_utc <<<"${line}"
  [[ -n "${pod_id}" ]] || continue

  api_json='{}'
  api_http_code=""
  api_error=""
  if [[ "${INCLUDE_API}" == "1" ]]; then
    api_body_file="$(mktemp /tmp/runpod_active_status_body.XXXXXX)"
    api_http_code="$(
      curl -sS \
        -o "${api_body_file}" \
        -w "%{http_code}" \
        -X GET \
        "${REST_ENDPOINT%/}/pods/${pod_id}" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Accept: application/json" \
        -H "User-Agent: chess-bot-runpod-active-status/1.0" || true
    )"
    api_raw="$(cat "${api_body_file}" || true)"
    rm -f "${api_body_file}"
    if [[ "${api_http_code}" =~ ^2 ]]; then
      api_json="$(jq -c '.' <<<"${api_raw}" 2>/dev/null || printf '{}')"
    else
      api_error="${api_raw}"
      api_json='{}'
    fi
  fi

  desired_status="$(jq -r '.desiredStatus // ""' <<<"${api_json}" 2>/dev/null || true)"
  if [[ "${RUNNING_ONLY}" == "1" && "${desired_status}" != "RUNNING" ]]; then
    continue
  fi

  ssh_host="$(jq -r '.publicIp // ""' <<<"${api_json}" 2>/dev/null || true)"
  [[ -n "${ssh_host}" ]] || ssh_host="${local_ssh_host:-${local_public_ip}}"
  ssh_port="$(jq -r '((.portMappings["22"] // .runtime.ports["22"] // "")|tostring)' <<<"${api_json}" 2>/dev/null || true)"
  [[ -n "${ssh_port}" && "${ssh_port}" != "null" ]] || ssh_port="${local_ssh_port:-22}"

  ssh_probe='{"attempted":false}'
  if [[ "${INCLUDE_SSH}" == "1" && -n "${ssh_host}" && -n "${ssh_port}" ]]; then
    remote_cmd="python3 - <<'PY'
import json, subprocess, socket
out={'hostname':'','gpu_lines':[],'uptime_seconds':None}
try: out['hostname']=socket.gethostname()
except Exception: pass
try:
    up=subprocess.check_output(['cat','/proc/uptime'], text=True, timeout=2).strip().split()[0]
    out['uptime_seconds']=float(up)
except Exception:
    pass
try:
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits'], text=True, timeout=4)
    out['gpu_lines']=[x.strip() for x in s.splitlines() if x.strip()]
except Exception:
    out['gpu_lines']=[]
print(json.dumps(out, ensure_ascii=True))
PY"
    ssh_out="$(timeout "${SSH_TIMEOUT_SECONDS}" ssh \
      -i "${SSH_KEY}" \
      -p "${ssh_port}" \
      -o BatchMode=yes \
      -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}" \
      -o IdentitiesOnly=yes \
      -o AddKeysToAgent=no \
      -o IdentityAgent=none \
      -o "StrictHostKeyChecking=${SSH_HOST_KEY_CHECKING}" \
      -o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}" \
      "${SSH_USER}@${ssh_host}" "${remote_cmd}" 2>&1 || true)"
    if jq -e . >/dev/null 2>&1 <<<"${ssh_out}"; then
      ssh_probe="$(jq -cn --argjson remote "${ssh_out}" '{attempted:true, reachable:true, remote:$remote}')"
    else
      ssh_probe="$(jq -cn --arg err "${ssh_out}" '{attempted:true, reachable:false, error:$err}')"
    fi
  fi

  pod_obj="$(jq -cn \
    --arg pod_id "${pod_id}" \
    --arg local_state "${local_state}" \
    --arg pod_name "${pod_name}" \
    --arg run_id "${run_id}" \
    --arg last_seen_utc "${last_seen_utc}" \
    --arg local_public_ip "${local_public_ip}" \
    --arg local_ssh_host "${local_ssh_host}" \
    --arg local_ssh_port "${local_ssh_port}" \
    --arg api_http_code "${api_http_code}" \
    --arg api_error "${api_error}" \
    --argjson api "${api_json}" \
    --argjson ssh_probe "${ssh_probe}" \
    '{
      pod_id:$pod_id,
      local:{
        state:$local_state,
        pod_name:$pod_name,
        run_id:$run_id,
        last_seen_utc:$last_seen_utc,
        public_ip:$local_public_ip,
        ssh_host:$local_ssh_host,
        ssh_port:$local_ssh_port
      },
      api:{
        http_code:$api_http_code,
        error:$api_error,
        desired_status:($api.desiredStatus // ""),
        interruptible:($api.interruptible // null),
        cost_per_hr:($api.costPerHr // null),
        image_name:($api.imageName // ""),
        public_ip:($api.publicIp // ""),
        port_mappings:($api.portMappings // {}),
        runtime:($api.runtime // null),
        machine:($api.machine // {}),
        gpu_count:($api.gpuCount // null),
        vcpu_count:($api.vcpuCount // null),
        memory_gb:($api.memoryInGb // null),
        last_status_change:($api.lastStatusChange // "")
      },
      ssh:$ssh_probe
    }'
  )"
  pods_json="$(jq -c --argjson row "${pod_obj}" '. + [$row]' <<<"${pods_json}")"
done

out="$(jq -cn \
  --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg registry_file "${REGISTRY_FILE}" \
  --arg rest_endpoint "${REST_ENDPOINT}" \
  --argjson include_api "${INCLUDE_API}" \
  --argjson include_ssh "${INCLUDE_SSH}" \
  --argjson running_only "${RUNNING_ONLY}" \
  --argjson pods "${pods_json}" \
  '{
    ts_utc:$ts_utc,
    registry_file:$registry_file,
    rest_endpoint:$rest_endpoint,
    include_api:($include_api==1),
    include_ssh:($include_ssh==1),
    running_only:($running_only==1),
    pod_count:($pods|length),
    pods:$pods
  }'
)"

printf '%s\n' "${out}"
if [[ "${WRITE_REPORT}" == "1" ]]; then
  mkdir -p "${REPORT_DIR}"
  printf '%s\n' "${out}" > "${REPORT_JSON}"
  echo "[runpod-active-status] report=${REPORT_JSON}" >&2
fi
