from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _pick_file(run_artifacts: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        files = sorted(run_artifacts.glob(pat))
        if files:
            return files[-1]
    return None


def verify_full_hf_cycle_run(
    repo_root: str | Path,
    run_id: str,
    *,
    require_terminated: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    cycle_dir = root / "artifacts" / "runpod_cycles" / run_id
    run_artifacts = cycle_dir / "collected" / "run_artifacts"

    checks: dict[str, bool] = {}
    paths: dict[str, str] = {}

    required_local = {
        "provision_json": cycle_dir / "provision.json",
        "stop_response_json": cycle_dir / "stop_response.json",
        "run_artifacts_dir": run_artifacts,
    }
    for key, p in required_local.items():
        checks[key] = p.exists()
        paths[key] = str(p)

    model_path = _pick_file(run_artifacts, [f"model_{run_id}.pt", "model_*.pt", "*.pt"])
    metrics_path = _pick_file(run_artifacts, [f"metrics_{run_id}.json", "metrics_*.json"])
    train_log_path = _pick_file(run_artifacts, [f"train_stdout_{run_id}.log", "train_stdout_*.log"])
    exit_code_path = _pick_file(run_artifacts, [f"train_exit_code.txt"])
    hf_manifest_path = _pick_file(run_artifacts, [f"hf_dataset_fetch_manifest.json"])
    progress_path = _pick_file(run_artifacts, [f"train_progress_{run_id}.jsonl", "train_progress_*.jsonl"])
    context_path = _pick_file(run_artifacts, [f"context_probe_{run_id}.json", "context_probe_*.json"])
    gpu_csv_path = _pick_file(run_artifacts, [f"gpu_usage_samples_{run_id}.csv", "gpu_usage_samples_*.csv"])

    optional_map = {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "train_log_path": train_log_path,
        "exit_code_path": exit_code_path,
        "hf_manifest_path": hf_manifest_path,
        "progress_path": progress_path,
        "context_path": context_path,
        "gpu_csv_path": gpu_csv_path,
    }
    for k, p in optional_map.items():
        checks[k] = bool(p and p.exists())
        paths[k] = str(p) if p else ""

    details: dict[str, Any] = {"run_id": run_id, "cycle_dir": str(cycle_dir), "paths": paths, "checks": checks}

    if hf_manifest_path and hf_manifest_path.exists():
        m = _read_json(hf_manifest_path)
        agg_by_format = m.get("aggregate_by_format") or {}
        details["hf_manifest"] = {
            "dataset_count": (m.get("aggregate") or {}).get("dataset_count"),
            "aggregate_by_format_keys": sorted(list(agg_by_format.keys())),
            "has_game_format_bucket": "game_jsonl_runtime_splice_v1" in agg_by_format,
        }

    if exit_code_path and exit_code_path.exists():
        exit_code_txt = _read_text(exit_code_path)
        details["train_exit_code"] = exit_code_txt
        checks["train_exit_code_zero"] = exit_code_txt == "0"
    else:
        checks["train_exit_code_zero"] = False

    if metrics_path and metrics_path.exists():
        metrics = _read_json(metrics_path)
        history = metrics.get("history") or []
        last = history[-1] if history else {}
        details["metrics"] = {
            "epochs_requested": metrics.get("epochs"),
            "history_len": len(history),
            "last_epoch": last.get("epoch"),
            "last_val_loss": last.get("val_loss"),
            "last_top1": last.get("top1"),
        }

    if progress_path and progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as f:
                progress_lines = sum(1 for line in f if line.strip())
        except Exception:
            progress_lines = None
        details["progress"] = {"line_count": progress_lines}

    if gpu_csv_path and gpu_csv_path.exists():
        try:
            with gpu_csv_path.open("r", encoding="utf-8") as f:
                gpu_samples = sum(1 for line in f if line.strip())
        except Exception:
            gpu_samples = None
        details["gpu_samples"] = {"line_count": gpu_samples}

    if required_local["stop_response_json"].exists():
        stop_resp = _read_json(required_local["stop_response_json"])
        desired = (((stop_resp.get("data") or {}).get("podStop") or {}).get("desiredStatus"))
        details["stop_response"] = {"desired_status": desired}
        checks["stop_response_exitlike"] = str(desired or "").upper() in {"EXITED", "STOPPED", "TERMINATED"}
    else:
        checks["stop_response_exitlike"] = False

    term_path = cycle_dir / "terminate_response.json"
    checks["terminate_response_json"] = term_path.exists()
    paths["terminate_response_json"] = str(term_path)
    if term_path.exists():
        term = _read_json(term_path)
        http_code = str(term.get("http_code") or "")
        details["terminate_response"] = {"http_code": http_code}
        checks["terminate_http_ok"] = http_code.startswith("2") or http_code == "404"
    else:
        checks["terminate_http_ok"] = False

    required_keys = [
        "provision_json",
        "stop_response_json",
        "run_artifacts_dir",
        "model_path",
        "metrics_path",
        "train_log_path",
        "exit_code_path",
        "hf_manifest_path",
        "context_path",
        "gpu_csv_path",
        "train_exit_code_zero",
        "stop_response_exitlike",
    ]
    if require_terminated:
        required_keys.extend(["terminate_response_json", "terminate_http_ok"])

    details["ok"] = all(bool(checks.get(k)) for k in required_keys)
    details["required_checks"] = required_keys
    return details

