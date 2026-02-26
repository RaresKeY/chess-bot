#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> None:
    print({"run": cmd})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def _find_single_pgn(extract_dir: Path) -> Path:
    pgns = sorted(extract_dir.rglob("*.pgn"))
    if not pgns:
        raise SystemExit(f"No PGN found under {extract_dir}")
    if len(pgns) > 1:
        raise SystemExit(f"Expected one PGN under {extract_dir}, found {len(pgns)}: {[str(p) for p in pgns]}")
    return pgns[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, validate, and splice a Lichess elite monthly dataset")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format (e.g. 2025-10)")
    parser.add_argument("--validated-dir", default="", help="Output dir for validated corpus (default data/validated/elite_<month>)")
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Output dir for splice dataset (default data/dataset/elite_<month>_cap<max-samples-per-game>)",
    )
    parser.add_argument("--re-download", action="store_true", help="Force re-download ZIP even if already present")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into existing validated/dataset output dirs")
    parser.add_argument("--min-plies", type=int, default=8)
    parser.add_argument("--validation-workers", type=int, default=0, help="Validation workers (0 uses all cores)")
    parser.add_argument("--validation-progress-every", type=int, default=50000)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-context", type=int, default=8)
    parser.add_argument("--min-target", type=int, default=1)
    parser.add_argument("--max-samples-per-game", type=int, default=8)
    parser.add_argument("--allow-draws", action="store_true")
    parser.add_argument("--splice-workers", type=int, default=0, help="Splice pass-2 threads (0 uses all cores)")
    parser.add_argument("--splice-batch-size", type=int, default=256)
    parser.add_argument("--splice-progress-every", type=int, default=10000)
    args = parser.parse_args()

    month = args.month.strip()
    if len(month) != 7 or month[4] != "-":
        raise SystemExit("Expected --month in YYYY-MM format")

    validated_dir = Path(args.validated_dir) if args.validated_dir else REPO_ROOT / "data" / "validated" / f"elite_{month}"
    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else REPO_ROOT / "data" / "dataset" / f"elite_{month}_cap{args.max_samples_per_game}"
    )
    validated_dir = validated_dir.resolve()
    dataset_dir = dataset_dir.resolve()

    if not args.overwrite:
        if validated_dir.exists():
            raise SystemExit(f"Validated output dir already exists (use --overwrite): {validated_dir}")
        if dataset_dir.exists():
            raise SystemExit(f"Dataset output dir already exists (use --overwrite): {dataset_dir}")

    py = str((REPO_ROOT / ".venv" / "bin" / "python") if (REPO_ROOT / ".venv" / "bin" / "python").exists() else sys.executable)

    download_cmd = [py, "scripts/download_lichess_elite_month.py", "--month", month]
    if args.re_download:
        download_cmd.append("--re-download")
    _run(download_cmd)

    extract_dir = REPO_ROOT / "data" / "raw" / "elite" / month
    pgn_path = _find_single_pgn(extract_dir)

    valid_out = validated_dir / "valid_games.jsonl"
    invalid_out = validated_dir / "invalid_games.csv"
    summary_out = validated_dir / "summary.json"
    _run(
        [
            py,
            "scripts/validate_games.py",
            "--input",
            str(pgn_path),
            "--valid-out",
            str(valid_out),
            "--invalid-out",
            str(invalid_out),
            "--summary-out",
            str(summary_out),
            "--min-plies",
            str(args.min_plies),
            "--workers",
            str(args.validation_workers),
            "--progress-every",
            str(args.validation_progress_every),
        ]
    )

    splice_cmd = [
        py,
        "scripts/build_splice_dataset.py",
        "--input",
        str(valid_out),
        "--output-dir",
        str(dataset_dir),
        "--k",
        str(args.k),
        "--min-context",
        str(args.min_context),
        "--min-target",
        str(args.min_target),
        "--max-samples-per-game",
        str(args.max_samples_per_game),
        "--workers",
        str(args.splice_workers),
        "--batch-size",
        str(args.splice_batch_size),
        "--progress-every",
        str(args.splice_progress_every),
    ]
    if args.allow_draws:
        splice_cmd.append("--allow-draws")
    _run(splice_cmd)

    print(
        {
            "complete": {
                "month": month,
                "pgn": str(pgn_path),
                "validated_dir": str(validated_dir),
                "dataset_dir": str(dataset_dir),
                "validation_summary": str(summary_out),
                "dataset_stats": str(dataset_dir / "stats.json"),
            }
        }
    )


if __name__ == "__main__":
    main()
