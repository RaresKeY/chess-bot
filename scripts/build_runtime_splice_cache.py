#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
import time
from array import array
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import primitives from training, but we will implement the chunked loop here
from src.chessbot.training import (  # noqa: E402
    PHASE_UNKNOWN,
    RuntimeSpliceConfig,
    _moves_from_row,
    _phase_ids_by_ply_prefix,
    _runtime_index_memory_bytes,
    _runtime_splice_indices_for_moves,
    phase_to_id,
)


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


class _ProgressManager:
    """Unified progress tracker using TTY-friendly bars."""

    def __init__(self, enabled: bool, total_tasks: int) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.total = total_tasks
        self.completed_tasks = 0
        self.start_time = time.monotonic()
        self.last_draw = 0.0

    def update(self, completed_tasks: int, detail: str = "") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if (now - self.last_draw) < 0.1 and completed_tasks < self.total:
            return
        self.last_draw = now

        self.completed_tasks = completed_tasks
        frac = max(0.0, min(1.0, completed_tasks / self.total)) if self.total > 0 else 1.0
        bar_width = 30
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        
        elapsed = now - self.start_time
        rate = completed_tasks / elapsed if elapsed > 0 else 0
        
        sys.stderr.write(
            f"\r[splice-cache] [{bar}] {completed_tasks}/{self.total} chunks | {rate:.1f}chk/s | {detail[:40]:<40}"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")


def _index_file_chunk(
    path: str,
    start: int,
    end: int,
    cfg: RuntimeSpliceConfig,
    queue: Optional[multiprocessing.Queue],
    task_id: int,
) -> Tuple[int, Dict[str, Any]]:
    """Worker function to index a specific byte-range of a JSONL file."""
    path_ids = array("I")
    offsets = array("Q")
    splice_indices = array("I")
    sample_phase_ids = array("B")
    game_count = 0
    sample_count = 0

    with open(path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()  # Skip partial line

        while True:
            offset = f.tell()
            if offset >= end:
                break
            line = f.readline()
            if not line:
                break
            
            line_stripped = line.strip()
            if not line_stripped:
                continue

            try:
                row = json.loads(line_stripped.decode("utf-8"))
                moves = _moves_from_row(row)
                game_id = str(row.get("game_id", ""))
                splices = _runtime_splice_indices_for_moves(moves, cfg, game_id=game_id)
                phase_ids = _phase_ids_by_ply_prefix(moves) if splices else []
                
                game_count += 1
                for s_idx in splices:
                    sample_count += 1
                    path_ids.append(0)  # Fixed to 0 per chunk, merged later
                    offsets.append(offset)
                    splice_indices.append(s_idx)
                    ph = phase_ids[s_idx] if s_idx < len(phase_ids) else phase_to_id(PHASE_UNKNOWN)
                    sample_phase_ids.append(ph)
            except Exception:
                continue

            if queue is not None and game_count % 1000 == 0:
                queue.put(("progress", task_id, 0))

    result = {
        "path_ids": path_ids,
        "offsets": offsets,
        "splice_indices": splice_indices,
        "sample_phase_ids": sample_phase_ids,
        "game_count": game_count,
        "sample_count": sample_count,
    }
    return task_id, result


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel precompute for runtime-splice cache")
    p.add_argument("--dataset-dir", action="append", required=True, help="Dataset directories (repeatable)")
    p.add_argument("--out-subdir", default="runtime_splice_cache")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--min-context", type=int, default=8)
    p.add_argument("--min-target", type=int, default=1)
    p.add_argument("--max-samples-per-game", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    p.add_argument("--chunk-size-mb", type=int, default=4)
    _bool_arg(p, "overwrite", False, "Overwrite existing")
    _bool_arg(p, "progress-bar", True, "Show progress")
    _bool_arg(p, "verbose", True, "Print per-split summary lines")
    args = p.parse_args()

    cfg = RuntimeSpliceConfig(
        min_context=args.min_context,
        min_target=args.min_target,
        max_samples_per_game=args.max_samples_per_game,
        seed=args.seed,
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    chunk_size = args.chunk_size_mb * 1024 * 1024

    all_tasks = []
    for d_path in args.dataset_dir:
        d = Path(d_path).resolve()
        for s in splits:
            f_path = d / f"{s}.jsonl"
            if not f_path.exists():
                continue
            f_size = f_path.stat().st_size
            for start in range(0, f_size, chunk_size):
                all_tasks.append({
                    "dataset_dir": d,
                    "split": s,
                    "path": str(f_path),
                    "start": start,
                    "end": min(start + chunk_size, f_size),
                })

    if not all_tasks:
        print("No split files found to index.")
        return

    jobs = int(args.jobs)
    if jobs <= 0:
        jobs = int(multiprocessing.cpu_count() or 1)

    progress = _ProgressManager(args.progress_bar, len(all_tasks))
    
    results_by_split: Dict[Tuple[Path, str], List[Dict[str, Any]]] = {}
    completed_chunks = 0

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _index_file_chunk, t["path"], t["start"], t["end"], cfg, None, i
                ): t for i, t in enumerate(all_tasks)
            }

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                _task_id, res = future.result()
                key = (task["dataset_dir"], task["split"])
                if key not in results_by_split:
                    results_by_split[key] = []
                results_by_split[key].append(res)
                completed_chunks += 1
                progress.update(completed_chunks, f"Finished {task['split']}")
    except (PermissionError, OSError):
        if args.verbose:
            print({"cache_parallel_unavailable": True, "fallback": "single_process"})
        for i, task in enumerate(all_tasks):
            _task_id, res = _index_file_chunk(task["path"], task["start"], task["end"], cfg, None, i)
            key = (task["dataset_dir"], task["split"])
            if key not in results_by_split:
                results_by_split[key] = []
            results_by_split[key].append(res)
            completed_chunks += 1
            progress.update(completed_chunks, f"Finished {task['split']}")

    progress.close()

    # Final Merge and Write
    for (d_dir, split), chunks in results_by_split.items():
        # Sort chunks by the offset of the first record to ensure file order
        chunks.sort(key=lambda x: x["offsets"][0] if len(x["offsets"]) > 0 else 0)
        
        final_path_ids = array("I")
        final_offsets = array("Q")
        final_splice_indices = array("I")
        final_sample_phase_ids = array("B")
        total_games = 0
        total_samples = 0
        
        for c in chunks:
            final_path_ids.extend(c["path_ids"])
            final_offsets.extend(c["offsets"])
            final_splice_indices.extend(c["splice_indices"])
            final_sample_phase_ids.extend(c["sample_phase_ids"])
            total_games += c["game_count"]
            total_samples += c["sample_count"]

        out_dir = d_dir / args.out_subdir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        
        def save(name, arr):
            p = out_dir / name
            with p.open("wb") as f:
                f.write(arr.tobytes())
            return p.stat().st_size

        sizes = {
            "path_ids.u32.bin": save("path_ids.u32.bin", final_path_ids),
            "offsets.u64.bin": save("offsets.u64.bin", final_offsets),
            "splice_indices.u32.bin": save("splice_indices.u32.bin", final_splice_indices),
            "sample_phase_ids.u8.bin": save("sample_phase_ids.u8.bin", final_sample_phase_ids),
        }
        
        paths_json = out_dir / "paths.json"
        src_file = d_dir / f"{split}.jsonl"
        paths_json.write_text(json.dumps([str(src_file.resolve())], indent=2))

        # Update manifest.json per dataset
        manifest_path = d_dir / args.out_subdir / "manifest.json"
        manifest = {}
        if manifest_path.exists() and not args.overwrite:
            try: manifest = json.loads(manifest_path.read_text())
            except: pass
        
        manifest.update({
            "schema_version": 1,
            "kind": "runtime_splice_cache",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(d_dir),
            "config": asdict(cfg),
        })
        if "splits" not in manifest: manifest["splits"] = {}
        
        manifest["splits"][split] = {
            "game_rows_total": total_games,
            "sample_rows_total": total_samples,
            "total_cache_bytes": sum(sizes.values()) + paths_json.stat().st_size,
            "files": {k: {"rel_path": f"{split}/{k}", "size_bytes": v} for k, v in sizes.items()}
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if args.verbose:
            print(
                {
                    "dataset_dir": str(d_dir),
                    "split": split,
                    "game_rows_total": int(total_games),
                    "sample_rows_total": int(total_samples),
                }
            )

    print(f"Cache complete for {len(results_by_split)} split(s).")

if __name__ == "__main__":
    main()
