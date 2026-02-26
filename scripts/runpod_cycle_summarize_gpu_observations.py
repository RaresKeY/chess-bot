#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate RunPod full-training observation artifacts by GPU SKU and suggest future override defaults"
    )
    p.add_argument(
        "--root",
        default="artifacts/runpod_cycles",
        help="Root directory containing runpod cycle runs (default: artifacts/runpod_cycles)",
    )
    p.add_argument(
        "--glob",
        default="*/spec_suggestions/gpu_full_training_observation_*.json",
        help="Glob under --root for observation JSON files",
    )
    p.add_argument("--output-json", default="", help="Optional JSON summary output path")
    p.add_argument("--output-md", default="", help="Optional Markdown summary output path")
    return p


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _recommend_batch(vram_total_mib: int | None, peak_used_mib_values: list[int]) -> int | None:
    if not vram_total_mib:
        return None
    if not peak_used_mib_values:
        return None
    peak_max = max(peak_used_mib_values)
    utilization = peak_max / max(vram_total_mib, 1)
    # Conservative coarse heuristic for next sequential runs.
    if vram_total_mib >= 70000:
        base = 8192
    elif vram_total_mib >= 44000:
        base = 4096
    elif vram_total_mib >= 22000:
        base = 2048
    elif vram_total_mib >= 15000:
        base = 1024
    else:
        base = 512
    if utilization > 0.92:
        return max(base // 2, 256)
    if utilization < 0.65:
        return base * 2
    return base


def _recommend_workers(vram_total_mib: int | None) -> int | None:
    if not vram_total_mib:
        return None
    if vram_total_mib >= 44000:
        return 8
    if vram_total_mib >= 22000:
        return 6
    if vram_total_mib >= 15000:
        return 4
    return 2


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    files = sorted(root.glob(args.glob))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        gpu_context = (data.get("gpu_context") or {})
        devices = gpu_context.get("devices") or []
        gpu_name = "unknown"
        vram_total_mib = None
        if devices and isinstance(devices[0], dict):
            gpu_name = str(devices[0].get("name") or "unknown")
            vram_total_mib = _as_int(devices[0].get("memory_total_mib"))
        data["_observation_path"] = str(path)
        data["_gpu_name"] = gpu_name
        data["_vram_total_mib"] = vram_total_mib
        grouped.setdefault(gpu_name, []).append(data)

    gpu_summaries = []
    for gpu_name in sorted(grouped):
        rows = grouped[gpu_name]
        peaks = []
        utils = []
        dataset_gib = []
        train_rows = []
        val_rows = []
        last_top1 = []
        last_val_loss = []
        epochs_ran = []
        vram_total_values = []
        run_ids = []
        for r in rows:
            run_ids.append(str(r.get("run_id") or ""))
            vram_total = _as_int(r.get("_vram_total_mib"))
            if vram_total:
                vram_total_values.append(vram_total)
            ds = r.get("dataset_summary") or {}
            gpu_peak = r.get("gpu_peak_observed") or {}
            tsummary = r.get("training_summary") or {}
            for target, source in (
                (dataset_gib, _as_float(ds.get("total_gib"))),
                (train_rows, _as_int((ds.get("total_rows") or {}).get("train"))),
                (val_rows, _as_int((ds.get("total_rows") or {}).get("val"))),
                (peaks, _as_int(gpu_peak.get("memory_used_mib_peak"))),
                (utils, _as_int(gpu_peak.get("utilization_gpu_pct_peak"))),
                (last_top1, _as_float(tsummary.get("last_top1"))),
                (last_val_loss, _as_float(tsummary.get("last_val_loss"))),
                (epochs_ran, _as_int(tsummary.get("epochs_ran"))),
            ):
                if source is not None:
                    target.append(source)

        vram_total_mib = max(vram_total_values) if vram_total_values else None
        rec_batch = _recommend_batch(vram_total_mib, [int(x) for x in peaks])
        rec_workers = _recommend_workers(vram_total_mib)
        gpu_summaries.append(
            {
                "gpu_name": gpu_name,
                "runs": len(rows),
                "run_ids": run_ids,
                "vram_total_mib": vram_total_mib,
                "dataset_total_gib_median": statistics.median(dataset_gib) if dataset_gib else None,
                "train_rows_median": int(statistics.median(train_rows)) if train_rows else None,
                "val_rows_median": int(statistics.median(val_rows)) if val_rows else None,
                "peak_memory_used_mib_max": max(peaks) if peaks else None,
                "peak_memory_used_mib_median": int(statistics.median(peaks)) if peaks else None,
                "peak_util_pct_max": max(utils) if utils else None,
                "peak_util_pct_median": int(statistics.median(utils)) if utils else None,
                "epochs_ran_median": int(statistics.median(epochs_ran)) if epochs_ran else None,
                "last_top1_median": statistics.median(last_top1) if last_top1 else None,
                "last_val_loss_median": statistics.median(last_val_loss) if last_val_loss else None,
                "recommended_overrides": {
                    "RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE": rec_batch,
                    "RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE": rec_workers,
                },
                "notes": [
                    "Recommendations are heuristic and based on observed peak GPU memory usage + VRAM tier.",
                    "Validate on the next run; adjust batch size if OOM or low utilization persists.",
                ],
            }
        )

    summary = {
        "root": str(root),
        "observation_file_count": len(files),
        "gpu_group_count": len(gpu_summaries),
        "gpus": gpu_summaries,
    }

    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_md:
        out_md = Path(args.output_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# RunPod GPU Observation Summary",
            "",
            f"- Root: `{root}`",
            f"- Observation files: `{len(files)}`",
            f"- GPU groups: `{len(gpu_summaries)}`",
            "",
        ]
        for gpu in gpu_summaries:
            rec = gpu["recommended_overrides"]
            lines.extend(
                [
                    f"## {gpu['gpu_name']}",
                    f"- Runs: `{gpu['runs']}`",
                    f"- VRAM total (MiB): `{gpu['vram_total_mib']}`",
                    f"- Dataset GiB median: `{gpu['dataset_total_gib_median']}`",
                    f"- Peak memory used max (MiB): `{gpu['peak_memory_used_mib_max']}`",
                    f"- Peak util max (%): `{gpu['peak_util_pct_max']}`",
                    f"- Last top1 median: `{gpu['last_top1_median']}`",
                    f"- Recommended `RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE`: `{rec.get('RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE')}`",
                    f"- Recommended `RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE`: `{rec.get('RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE')}`",
                    "",
                ]
            )
        out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
