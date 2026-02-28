from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts_utc": "2026-02-28T00:00:00Z",
        "source_script": "test",
        "action": "start",
        "state": "RUNNING",
        "pod_id": "pod_test_123",
        "run_id": "run-test",
        "pod_name": "pod-test",
        "public_ip": "127.0.0.1",
        "ssh_host": "127.0.0.1",
        "ssh_port": "22",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_active_status_running_only_with_no_api_filters_all_rows(tmp_path: Path) -> None:
    registry = tmp_path / "runpod_tracked_pods.jsonl"
    _write_registry(registry)
    env = os.environ.copy()
    env["RUNPOD_TRACKED_PODS_FILE"] = str(registry)

    proc = subprocess.run(
        ["bash", "scripts/runpod_active_pods_full_status.sh", "--running-only", "--no-api", "--no-ssh", "--no-write"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["include_api"] is False
    assert payload["running_only"] is True
    assert payload["pod_count"] == 0


def test_active_status_no_api_mode_returns_local_registry_rows(tmp_path: Path) -> None:
    registry = tmp_path / "runpod_tracked_pods.jsonl"
    _write_registry(registry)
    env = os.environ.copy()
    env["RUNPOD_TRACKED_PODS_FILE"] = str(registry)
    env.pop("RUNPOD_API_KEY", None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

    proc = subprocess.run(
        ["bash", "scripts/runpod_active_pods_full_status.sh", "--no-api", "--no-ssh", "--no-write"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["include_api"] is False
    assert payload["include_ssh"] is False
    assert payload["pod_count"] == 1
    assert payload["pods"][0]["pod_id"] == "pod_test_123"
    assert payload["pods"][0]["local"]["state"] == "RUNNING"


def test_runpod_provision_parser_interruptible_toggle_order() -> None:
    proc = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-c",
            (
                "from scripts.runpod_provision import build_parser; "
                "p=build_parser(); "
                "a=p.parse_args(['provision','--interruptible','--no-interruptible']); "
                "print('1' if a.interruptible else '0')"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert proc.stdout.strip() == "0"
