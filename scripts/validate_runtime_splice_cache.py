#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_SPLIT_FILES = (
    "paths.json",
    "path_ids.u32.bin",
    "offsets.u64.bin",
    "splice_indices.u32.bin",
    "sample_phase_ids.u8.bin",
)


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _resolve_dataset_dirs(dataset_dirs: list[str], dataset_root: str, dataset_glob: str) -> list[Path]:
    out: list[Path] = []
    for d in dataset_dirs:
        p = Path(d).resolve()
        if p.is_dir():
            out.append(p)
    if dataset_root:
        root = Path(dataset_root).resolve()
        if root.is_dir():
            for p in sorted(root.glob(dataset_glob)):
                if p.is_dir():
                    out.append(p.resolve())
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _validate_split(cache_split_dir: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    sizes: dict[str, int] = {}
    for name in REQUIRED_SPLIT_FILES:
        fp = cache_split_dir / name
        if not fp.is_file():
            errors.append(f"missing_file:{fp}")
            continue
        sizes[name] = int(fp.stat().st_size)
    if errors:
        return errors, {"sizes": sizes}

    rows_offsets = sizes["offsets.u64.bin"] // 8
    rows_path_ids = sizes["path_ids.u32.bin"] // 4
    rows_splice = sizes["splice_indices.u32.bin"] // 4
    rows_phase = sizes["sample_phase_ids.u8.bin"]
    if not (rows_offsets == rows_path_ids == rows_splice == rows_phase):
        errors.append(
            f"row_count_mismatch:offsets={rows_offsets},path_ids={rows_path_ids},splice_indices={rows_splice},phase_ids={rows_phase}"
        )

    try:
        paths = json.loads((cache_split_dir / "paths.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid_paths_json:{exc}")
        paths = []
    if not isinstance(paths, list) or not paths:
        errors.append("paths_json_empty_or_invalid")
    else:
        for p in paths:
            if not Path(str(p)).is_file():
                errors.append(f"source_path_missing:{p}")

    return errors, {"sizes": sizes, "rows": int(rows_offsets)}


def _validate_dataset(dataset_dir: Path, splits: list[str]) -> dict:
    cache_dir = dataset_dir / "runtime_splice_cache"
    errors: list[str] = []
    manifest: dict = {}
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing_manifest:{manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid_manifest_json:{exc}")
            manifest = {}
    if manifest and str(manifest.get("kind", "")) != "runtime_splice_cache":
        errors.append(f"manifest_kind_unexpected:{manifest.get('kind')}")

    split_details: dict[str, dict] = {}
    for split in splits:
        cache_split_dir = cache_dir / split
        split_errors, detail = _validate_split(cache_split_dir)
        split_details[split] = detail
        for msg in split_errors:
            errors.append(f"{split}:{msg}")

    return {
        "dataset_dir": str(dataset_dir),
        "ok": len(errors) == 0,
        "errors": errors,
        "splits": split_details,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Validate runtime_splice_cache structure and binary consistency")
    p.add_argument("--dataset-dir", action="append", default=[], help="Repeatable dataset directory path")
    p.add_argument("--dataset-root", default="", help="Optional root directory to scan for dataset directories")
    p.add_argument("--dataset-glob", default="elite_*_game", help="Glob under --dataset-root")
    p.add_argument("--splits", default="train,val,test", help="Comma-separated split names")
    _bool_arg(p, "strict", True, "Exit non-zero when any dataset validation fails")
    args = p.parse_args()

    split_list = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    datasets = _resolve_dataset_dirs(args.dataset_dir, args.dataset_root, args.dataset_glob)
    if not datasets:
        raise SystemExit("No dataset directories resolved. Use --dataset-dir or --dataset-root.")

    results = [_validate_dataset(ds, split_list) for ds in datasets]
    failed = [r for r in results if not bool(r.get("ok"))]
    summary = {
        "dataset_count": len(results),
        "ok_count": len(results) - len(failed),
        "failed_count": len(failed),
        "splits": split_list,
        "results": results,
    }
    print(json.dumps(summary, indent=2))

    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
