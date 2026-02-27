#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _find_first(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def _fmt_float(v: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return "n/a"


def _fmt_eta(seconds: Any) -> str:
    try:
        s = int(float(seconds))
    except Exception:
        return "n/a"
    if s <= 0:
        return "n/a"
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _latest_by_event(rows: List[Dict[str, Any]], event: str) -> Optional[Dict[str, Any]]:
    for row in reversed(rows):
        if row.get("event") == event:
            return row
    return None


def _parse_gpu_samples(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "latest_ts": "",
            "latest_rows": [],
            "latest_util_avg": None,
            "peak_util": None,
            "peak_mem_used_mib": None,
        }
    all_rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            ts, name, util, mem_used, mem_total = row[:5]
            try:
                util_i = int(float(util))
            except Exception:
                util_i = 0
            try:
                mem_used_i = int(float(mem_used))
            except Exception:
                mem_used_i = 0
            try:
                mem_total_i = int(float(mem_total))
            except Exception:
                mem_total_i = 0
            all_rows.append(
                {
                    "ts": ts,
                    "name": name,
                    "util": util_i,
                    "mem_used": mem_used_i,
                    "mem_total": mem_total_i,
                }
            )
    if not all_rows:
        return {
            "latest_ts": "",
            "latest_rows": [],
            "latest_util_avg": None,
            "peak_util": None,
            "peak_mem_used_mib": None,
        }
    latest_ts = all_rows[-1]["ts"]
    latest_rows = [r for r in all_rows if r["ts"] == latest_ts]
    latest_util_avg = None
    if latest_rows:
        latest_util_avg = sum(r["util"] for r in latest_rows) / len(latest_rows)
    peak_util = max((r["util"] for r in all_rows), default=None)
    peak_mem = max((r["mem_used"] for r in all_rows), default=None)
    return {
        "latest_ts": latest_ts,
        "latest_rows": latest_rows,
        "latest_util_avg": latest_util_avg,
        "peak_util": peak_util,
        "peak_mem_used_mib": peak_mem,
    }


def generate_report(repo_root: Path, run_id: str) -> Tuple[str, Dict[str, Any]]:
    cycle_dir = repo_root / "artifacts" / "runpod_cycles" / run_id
    collected_dir = cycle_dir / "collected" / "run_artifacts"
    nested_report_dir = cycle_dir / run_id / "reports"
    report_dir = cycle_dir / "reports"

    progress_path = _find_first(
        [
            collected_dir / f"train_progress_{run_id}.jsonl",
            cycle_dir / f"train_progress_{run_id}.jsonl",
        ]
    )
    metrics_path = _find_first(
        [
            collected_dir / f"metrics_{run_id}.json",
            cycle_dir / f"metrics_{run_id}.json",
        ]
    )
    model_path = _find_first(
        [
            collected_dir / f"model_{run_id}.pt",
            cycle_dir / f"model_{run_id}.pt",
        ]
    )
    gpu_path = _find_first(
        [
            collected_dir / f"gpu_usage_samples_{run_id}.csv",
            cycle_dir / f"gpu_usage_samples_{run_id}.csv",
        ]
    )
    eta_path = _find_first(
        [
            nested_report_dir / f"epoch_eta_report_{run_id}.jsonl",
            report_dir / f"epoch_eta_report_{run_id}.jsonl",
        ]
    )

    progress_rows = _read_jsonl(progress_path) if progress_path else []
    eta_rows = _read_jsonl(eta_path) if eta_path else []
    metrics = _read_json(metrics_path) if metrics_path else {}
    gpu = _parse_gpu_samples(gpu_path)

    train_setup = _latest_by_event(progress_rows, "train_setup") or {}
    epoch_end = _latest_by_event(progress_rows, "epoch_end") or {}
    script_complete = _latest_by_event(progress_rows, "script_complete") or _latest_by_event(progress_rows, "train_complete") or {}
    eta_latest = eta_rows[-1] if eta_rows else {}

    total_epochs = int(
        epoch_end.get("epochs")
        or train_setup.get("epochs")
        or metrics.get("epochs")
        or 0
    )
    completed_epoch = int(
        script_complete.get("epochs_completed")
        or epoch_end.get("epoch")
        or 0
    )
    status = "training_running"
    if script_complete:
        status = "training_finished"
    elif completed_epoch <= 0 and not progress_rows:
        status = "progress_not_found"

    metric_blob = epoch_end.get("metrics") if isinstance(epoch_end.get("metrics"), dict) else {}
    train_loss = metric_blob.get("train_loss")
    val_loss = metric_blob.get("val_loss")
    top1 = metric_blob.get("top1")
    top5 = metric_blob.get("top5")

    if train_loss is None or val_loss is None:
        hist = metrics.get("history") if isinstance(metrics.get("history"), list) else []
        if hist:
            last = hist[-1] if isinstance(hist[-1], dict) else {}
            train_loss = last.get("train_loss", train_loss)
            val_loss = last.get("val_loss", val_loss)
            top1 = last.get("top1", top1)
            top5 = last.get("top5", top5)

    cache_blob = train_setup.get("cache_load_reason_by_split")
    cache_train = ""
    cache_val = ""
    if isinstance(cache_blob, dict):
        cache_train = str(cache_blob.get("train", ""))
        cache_val = str(cache_blob.get("val", ""))

    world_size = train_setup.get("world_size")
    distributed = train_setup.get("distributed")

    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    latest_gpu_rows = gpu.get("latest_rows") or []
    gpu_snapshot_line = "n/a"
    if latest_gpu_rows:
        parts = []
        for idx, row in enumerate(latest_gpu_rows):
            parts.append(
                f"gpu{idx}:{row.get('util', 0)}% {row.get('mem_used', 0)}/{row.get('mem_total', 0)} MiB"
            )
        gpu_snapshot_line = "; ".join(parts)

    lines = [
        "# RunPod Easy Progress Report",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_utc: `{now_utc}`",
        f"- status: `{status}`",
        "",
        "## Progress",
        f"- epoch: `{completed_epoch}/{total_epochs}`",
        f"- eta_remaining: `{_fmt_eta(eta_latest.get('eta_seconds_remaining'))}`",
        f"- eta_utc: `{eta_latest.get('eta_utc') or 'n/a'}`",
        f"- metrics: `train_loss={_fmt_float(train_loss)} val_loss={_fmt_float(val_loss)} top1={_fmt_float(top1)} top5={_fmt_float(top5)}`",
        "",
        "## GPU Snapshot",
        f"- latest_sample_ts: `{gpu.get('latest_ts') or 'n/a'}`",
        f"- latest_util_avg_pct: `{_fmt_float(gpu.get('latest_util_avg'), ndigits=2)}`",
        f"- latest_per_gpu: `{gpu_snapshot_line}`",
        f"- peak_util_pct: `{gpu.get('peak_util') if gpu.get('peak_util') is not None else 'n/a'}`",
        f"- peak_mem_used_mib: `{gpu.get('peak_mem_used_mib') if gpu.get('peak_mem_used_mib') is not None else 'n/a'}`",
        "",
        "## DDP and Cache",
        f"- world_size: `{world_size if world_size is not None else 'n/a'}`",
        f"- distributed: `{distributed if distributed is not None else 'n/a'}`",
        f"- cache_train: `{cache_train or 'n/a'}`",
        f"- cache_val: `{cache_val or 'n/a'}`",
        "",
        "## Artifacts",
        f"- model_local: `{'yes' if model_path and model_path.is_file() else 'no'}`",
        f"- model_path: `{str(model_path) if model_path else 'n/a'}`",
        f"- metrics_path: `{str(metrics_path) if metrics_path else 'n/a'}`",
        f"- progress_path: `{str(progress_path) if progress_path else 'n/a'}`",
        f"- eta_report_path: `{str(eta_path) if eta_path else 'n/a'}`",
        "",
    ]
    report = "\n".join(lines)
    summary = {
        "run_id": run_id,
        "status": status,
        "epoch_completed": completed_epoch,
        "epoch_total": total_epochs,
        "world_size": world_size,
        "distributed": distributed,
        "cache_train": cache_train,
        "cache_val": cache_val,
    }
    return report, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concise progress report for a RunPod cycle run in easy-flow style."
    )
    parser.add_argument("--run-id", required=True, help="RunPod cycle run id (e.g. runpod-cycle-... ).")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root path. Default: script parent repo root.",
    )
    parser.add_argument(
        "--write-md",
        default="",
        help="Optional markdown output path. Default when unset: artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md",
    )
    parser.add_argument(
        "--write-json",
        default="",
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print markdown report to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report, summary = generate_report(repo_root=repo_root, run_id=args.run_id)
    out_md = (
        Path(args.write_md)
        if args.write_md
        else repo_root / "artifacts" / "runpod_cycles" / args.run_id / "reports" / "easy_progress_report.md"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report + "\n", encoding="utf-8")

    if args.write_json:
        out_json = Path(args.write_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(report)
        print(f"[runpod-cycle-report-style] wrote_markdown={out_md}")
        if args.write_json:
            print(f"[runpod-cycle-report-style] wrote_json={args.write_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
