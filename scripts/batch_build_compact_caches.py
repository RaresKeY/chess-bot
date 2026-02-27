#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print({"run": cmd, "dry_run": dry_run})
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact game datasets and runtime caches for all validated months.")
    parser.add_argument("--validated-dir", default="data/validated", help="Base directory containing validated month folders")
    parser.add_argument("--dataset-out-dir", default="data/dataset", help="Base directory for output datasets")
    parser.add_argument("--min-context", type=int, default=8)
    parser.add_argument("--min-target", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=0, help="Workers for cache building (0 = auto / all CPU cores)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets and caches")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validated_base = Path(args.validated_dir).resolve()
    dataset_base = Path(args.dataset_out_dir).resolve()
    py = str((REPO_ROOT / ".venv" / "bin" / "python") if (REPO_ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    
    if not validated_base.is_dir():
        print(f"Validated dir not found: {validated_base}")
        return

    jobs = args.jobs if args.jobs > 0 else os.cpu_count()

    # Find all valid_games.jsonl under validated_base
    valid_game_files = sorted(validated_base.rglob("valid_games.jsonl"))
    
    if not valid_game_files:
        print(f"No valid_games.jsonl files found in {validated_base}")
        return

    for valid_file in valid_game_files:
        month_dir_name = valid_file.parent.name # e.g. elite_2025-01
        
        # Build dataset
        dataset_dir = dataset_base / f"{month_dir_name}_game"
        
        if dataset_dir.exists() and not args.overwrite:
            if (dataset_dir / "runtime_splice_cache" / "manifest.json").exists():
                print(f"Skipping {month_dir_name}, cache already exists. Use --overwrite to rebuild.")
                continue
        
        print(f"\n--- Processing {month_dir_name} ---")
        
        # 1. Build the compact game dataset
        build_ds_cmd = [
            py, "scripts/build_game_dataset.py",
            "--input", str(valid_file),
            "--output-dir", str(dataset_dir),
            "--runtime-min-context", str(args.min_context),
            "--runtime-min-target", str(args.min_target)
        ]
        
        print(f"Building compact game dataset for {month_dir_name}...")
        _run(build_ds_cmd, dry_run=args.dry_run)
        
        # 2. Build the runtime splice cache
        cache_cmd = [
            py, "scripts/build_runtime_splice_cache.py",
            "--dataset-dir", str(dataset_dir),
            "--splits", "train,val,test",
            "--jobs", str(jobs)
        ]
        if args.overwrite:
            cache_cmd.append("--overwrite")
            
        print(f"Building runtime splice cache for {month_dir_name} using {jobs} cores...")
        _run(cache_cmd, dry_run=args.dry_run)
        
    print("\nBatch compact cache building complete.")


if __name__ == "__main__":
    main()
