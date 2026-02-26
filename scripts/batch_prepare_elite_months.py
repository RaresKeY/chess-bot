#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
URL_MONTH_RE = re.compile(r"lichess_elite_(\d{4}-\d{2})\.zip(?:$|\?)")


def _parse_month_items(lines: Iterable[str]) -> List[str]:
    months: List[str] = []
    seen = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if MONTH_RE.fullmatch(line):
            month = line
        else:
            match = URL_MONTH_RE.search(line)
            if not match:
                raise ValueError(f"Unsupported month entry (expected YYYY-MM or elite ZIP URL): {line}")
            month = match.group(1)
        if month not in seen:
            seen.add(month)
            months.append(month)
    return months


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print({"run": cmd, "dry_run": dry_run})
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run download+validate+splice for multiple Lichess elite months from a month or URL list."
    )
    parser.add_argument(
        "--list-file",
        default="config/elite_month_validator_links.txt",
        help="Text file with one YYYY-MM or elite ZIP URL per line (comments allowed).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Pass through to monthly prep script")
    parser.add_argument("--re-download", action="store_true", help="Pass through to monthly prep script")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining months if one fails")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--min-plies", type=int, default=8)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--validation-progress-every", type=int, default=50000)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-context", type=int, default=8)
    parser.add_argument("--min-target", type=int, default=1)
    parser.add_argument("--max-samples-per-game", type=int, default=8)
    parser.add_argument("--allow-draws", action="store_true")
    parser.add_argument("--splice-workers", type=int, default=0)
    parser.add_argument("--splice-batch-size", type=int, default=256)
    parser.add_argument("--splice-progress-every", type=int, default=10000)
    args = parser.parse_args()

    list_file = Path(args.list_file)
    if not list_file.is_file():
        raise SystemExit(f"Month list file not found: {list_file}")
    months = _parse_month_items(list_file.read_text(encoding="utf-8").splitlines())
    if not months:
        raise SystemExit(f"No months found in {list_file}")

    py = str((REPO_ROOT / ".venv" / "bin" / "python") if (REPO_ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    print({"list_file": str(list_file), "months": months, "count": len(months), "max_samples_per_game": args.max_samples_per_game})

    failures: List[str] = []
    for month in months:
        cmd = [
            py,
            "scripts/acquire_and_prepare_elite_month.py",
            "--month",
            month,
            "--min-plies",
            str(args.min_plies),
            "--validation-workers",
            str(args.validation_workers),
            "--validation-progress-every",
            str(args.validation_progress_every),
            "--k",
            str(args.k),
            "--min-context",
            str(args.min_context),
            "--min-target",
            str(args.min_target),
            "--max-samples-per-game",
            str(args.max_samples_per_game),
            "--splice-workers",
            str(args.splice_workers),
            "--splice-batch-size",
            str(args.splice_batch_size),
            "--splice-progress-every",
            str(args.splice_progress_every),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.re_download:
            cmd.append("--re-download")
        if args.allow_draws:
            cmd.append("--allow-draws")

        try:
            _run(cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError:
            failures.append(month)
            print({"month_failed": month})
            if not args.continue_on_error:
                raise

    print({"complete": {"months": months, "failures": failures, "dry_run": args.dry_run}})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
