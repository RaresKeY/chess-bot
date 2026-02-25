#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _run_lines(cmd: list[str]) -> list[str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    if p.returncode != 0:
        return []
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def _gpu_active(util_threshold: int, mem_threshold_mb: int) -> bool:
    lines = _run_lines(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            util = int(float(parts[0]))
            mem = int(float(parts[1]))
        except ValueError:
            continue
        if util >= util_threshold or mem >= mem_threshold_mb:
            return True
    return False


def _port_has_client(port: int) -> bool:
    if port <= 0:
        return False
    lines = _run_lines(["ss", "-Htn"])
    port_str = f":{port}"
    for line in lines:
        cols = line.split()
        if len(cols) < 5:
            continue
        state = cols[0]
        local = cols[3]
        if state != "ESTAB":
            continue
        if local.endswith(port_str):
            return True
    return False


def _process_patterns_active(patterns: list[str]) -> bool:
    lines = _run_lines(["ps", "-eo", "pid=,cmd="])
    for line in lines:
        for pat in patterns:
            if pat and pat in line and "idle_watchdog.py" not in line:
                return True
    return False


def _heartbeat_recent(path_text: str, max_age: int) -> bool:
    if not path_text:
        return False
    p = Path(path_text)
    if not p.exists():
        return False
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return False
    return age <= max_age


def _stop_runpod_pod(endpoint: str, api_key: str, pod_id: str, verbose: bool) -> bool:
    attempts = [
        {
            "query": "mutation StopPod($input: PodStopInput!) { podStop(input: $input) }",
            "variables": {"input": {"podId": pod_id}},
        },
        {
            "query": "mutation StopPod($podId: String!) { podStop(input: { podId: $podId }) }",
            "variables": {"podId": pod_id},
        },
    ]
    for payload in attempts:
        req = urllib.request.Request(
            endpoint,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            if verbose:
                print({"idle_watchdog_runpod_stop_response": body[:500]})
            if "errors" not in body:
                return True
        except Exception as exc:
            if verbose:
                print({"idle_watchdog_runpod_stop_error": str(exc)})
    return False


def _perform_autostop(action: str, verbose: bool) -> None:
    if action == "runpod_api":
        endpoint = os.environ.get("RUNPOD_GRAPHQL_ENDPOINT", "https://api.runpod.io/graphql")
        api_key = os.environ.get("RUNPOD_API_KEY", "")
        pod_id = os.environ.get("RUNPOD_POD_ID", "")
        if not api_key or not pod_id:
            raise RuntimeError("RUNPOD_API_KEY and RUNPOD_POD_ID are required for AUTOSTOP_ACTION=runpod_api")
        ok = _stop_runpod_pod(endpoint, api_key, pod_id, verbose)
        if not ok:
            raise RuntimeError("RunPod stop request failed")
        return
    if action == "exit":
        if verbose:
            print({"idle_watchdog_action": "sending SIGTERM to pid1"})
        try:
            os.kill(1, signal.SIGTERM)
        except Exception:
            pass
        return
    raise RuntimeError(f"Unsupported AUTOSTOP_ACTION: {action}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Idle watchdog for RunPod cloud training pods")
    parser.add_argument("--idle-seconds", type=int, default=int(os.environ.get("IDLE_TIMEOUT_SECONDS", "3600")))
    parser.add_argument("--check-interval-seconds", type=int, default=int(os.environ.get("IDLE_CHECK_INTERVAL_SECONDS", "60")))
    parser.add_argument("--gpu-util-threshold", type=int, default=int(os.environ.get("IDLE_GPU_UTIL_THRESHOLD", "10")))
    parser.add_argument("--gpu-mem-mb-threshold", type=int, default=int(os.environ.get("IDLE_GPU_MEM_MB_THRESHOLD", "1024")))
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("SSH_PORT", "22")))
    parser.add_argument("--jupyter-port", type=int, default=int(os.environ.get("JUPYTER_PORT", "8888")))
    parser.add_argument("--api-port", type=int, default=int(os.environ.get("INFERENCE_API_PORT", "8000")))
    parser.add_argument(
        "--activity-file",
        default=os.environ.get("ACTIVITY_HEARTBEAT_FILE", "/tmp/chessbot_last_activity"),
        help="Heartbeat file touched by services like inference API",
    )
    parser.add_argument(
        "--process-patterns",
        default=os.environ.get(
            "IDLE_ACTIVE_PROCESS_PATTERNS",
            "train_baseline.py,jupyter-lab,uvicorn,inference_api.py,play_vs_model_server.py",
        ),
    )
    parser.add_argument("--autostop-action", default=os.environ.get("AUTOSTOP_ACTION", "runpod_api"))
    _bool_arg(parser, "verbose", os.environ.get("IDLE_WATCHDOG_VERBOSE", "1") == "1", "Verbose watchdog logs")
    args = parser.parse_args()

    patterns = [p.strip() for p in args.process_patterns.split(",") if p.strip()]
    last_active = time.time()
    if args.verbose:
        print(
            {
                "idle_watchdog_start": {
                    "idle_seconds": args.idle_seconds,
                    "check_interval_seconds": args.check_interval_seconds,
                    "gpu_util_threshold": args.gpu_util_threshold,
                    "gpu_mem_mb_threshold": args.gpu_mem_mb_threshold,
                    "ports": {"ssh": args.ssh_port, "jupyter": args.jupyter_port, "api": args.api_port},
                    "activity_file": args.activity_file,
                    "autostop_action": args.autostop_action,
                    "process_patterns": patterns,
                }
            }
        )

    while True:
        gpu_active = _gpu_active(args.gpu_util_threshold, args.gpu_mem_mb_threshold)
        ssh_client = _port_has_client(args.ssh_port)
        jupyter_client = _port_has_client(args.jupyter_port)
        api_client = _port_has_client(args.api_port)
        proc_active = _process_patterns_active(patterns)
        heartbeat_active = _heartbeat_recent(args.activity_file, args.check_interval_seconds * 2)
        active = any([gpu_active, ssh_client, jupyter_client, api_client, proc_active, heartbeat_active])

        now = time.time()
        if active:
            last_active = now
        idle_for = int(now - last_active)

        if args.verbose:
            print(
                {
                    "idle_watchdog_status": {
                        "gpu_active": gpu_active,
                        "ssh_client": ssh_client,
                        "jupyter_client": jupyter_client,
                        "api_client": api_client,
                        "proc_active": proc_active,
                        "heartbeat_active": heartbeat_active,
                        "idle_for_seconds": idle_for,
                    }
                }
            )

        if idle_for >= args.idle_seconds:
            print({"idle_watchdog_triggered": {"idle_for_seconds": idle_for, "action": args.autostop_action}})
            _perform_autostop(args.autostop_action, args.verbose)
            return

        time.sleep(max(5, args.check_interval_seconds))


if __name__ == "__main__":
    main()
