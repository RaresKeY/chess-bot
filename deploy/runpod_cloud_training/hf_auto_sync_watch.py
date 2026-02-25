#!/usr/bin/env python3
import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _split_patterns(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _snapshot(source_dir: Path, patterns: list[str]) -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    if not source_dir.exists():
        return snap
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(source_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns):
            st = p.stat()
            snap[rel] = (st.st_size, st.st_mtime_ns)
    return snap


def _run_sync(source_dir: Path, patterns_text: str, verbose: bool) -> int:
    script = Path(__file__).with_name("hf_sync.py")
    cmd = [sys.executable, str(script), "--source-dir", str(source_dir), "--patterns", patterns_text]
    if verbose:
        cmd.append("--verbose")
    else:
        cmd.append("--no-verbose")
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch local artifacts and sync changes to Hugging Face")
    parser.add_argument("--source-dir", default=os.environ.get("HF_SYNC_SOURCE_DIR", "./artifacts"))
    parser.add_argument("--patterns", default=os.environ.get("HF_SYNC_PATTERNS", "*.pt,*.json"))
    parser.add_argument("--interval-seconds", type=int, default=int(os.environ.get("HF_SYNC_INTERVAL_SECONDS", "120")))
    _bool_arg(parser, "verbose", os.environ.get("HF_SYNC_VERBOSE", "1") == "1", "Verbose watcher logging")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    patterns = _split_patterns(args.patterns)
    previous = _snapshot(source_dir, patterns)
    if args.verbose:
        print(
            {
                "hf_auto_sync_watch_start": {
                    "source_dir": str(source_dir),
                    "patterns": patterns,
                    "interval_seconds": args.interval_seconds,
                    "initial_matches": len(previous),
                }
            }
        )

    while True:
        time.sleep(max(5, args.interval_seconds))
        current = _snapshot(source_dir, patterns)
        if current == previous:
            continue
        changed_files = sorted(set(current.keys()) ^ set(previous.keys()))
        changed_files.extend(k for k in current.keys() if k in previous and current[k] != previous[k])
        changed_files = sorted(set(changed_files))
        if args.verbose:
            print({"hf_auto_sync_change_detected": {"changed_count": len(changed_files), "files": changed_files[:20]}})
        rc = _run_sync(source_dir, args.patterns, args.verbose)
        if args.verbose:
            print({"hf_auto_sync_result": {"return_code": rc}})
        if rc == 0:
            previous = current


if __name__ == "__main__":
    main()
