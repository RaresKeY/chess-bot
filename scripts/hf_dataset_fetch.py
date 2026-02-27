#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
import shutil
from collections import defaultdict
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.chessbot.secrets import default_dotenv_paths, resolve_secret


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _resolve_token(explicit_token: str | None, service: str, username: str) -> str | None:
    dotenv_paths = default_dotenv_paths(
        repo_root=REPO_ROOT,
        override_var_names=("HF_DOTENV_PATH", "CHESSBOT_DOTENV_PATH"),
        fallback_filenames=(".env.hf_dataset", ".env"),
    )
    token, _ = resolve_secret(
        explicit_value=str(explicit_token or ""),
        env_var_names=("HF_TOKEN",),
        keyring_service=service,
        keyring_username=username,
        dotenv_keys=("HF_TOKEN",),
        dotenv_paths=dotenv_paths,
        order=("explicit", "env", "keyring", "dotenv"),
    )
    return token or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch a published validated dataset from a Hugging Face dataset repo")
    p.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO_ID", os.environ.get("HF_REPO_ID", "")))
    p.add_argument("--token", default=None)
    p.add_argument("--keyring-service", default=os.environ.get("HF_KEYRING_SERVICE", "huggingface"))
    p.add_argument("--keyring-username", default=os.environ.get("HF_KEYRING_USERNAME", "codex_hf_write_token"))
    p.add_argument("--dataset-name", default="")
    p.add_argument("--version", default="")
    p.add_argument("--repo-path-prefix", default=os.environ.get("HF_DATASET_PATH_PREFIX", "validated_datasets"))
    p.add_argument("--dest-dir", required=True, help="Local destination for fetched dataset")
    _bool_arg(p, "all-latest", False, "Fetch the latest version for every dataset under the repo path prefix")
    _bool_arg(p, "extract-archive", True, "Extract *.tar.gz dataset archive when present")
    _bool_arg(p, "dry-run", False, "Print the planned repo path/patterns without downloading")
    _bool_arg(p, "verbose", True, "Verbose fetch logging")
    p.add_argument("--output-manifest", default="", help="Optional local JSON file summarizing fetched datasets and train/val paths")
    return p


def _parse_repo_dataset_versions(repo_files: list[str], prefix: str) -> dict[str, list[str]]:
    base = prefix.strip("/")
    roots: dict[str, set[str]] = defaultdict(set)
    for path in repo_files:
        parts = [p for p in path.split("/") if p]
        if len(parts) < 4:
            continue
        if parts[0] != base:
            continue
        dataset_name = parts[1]
        version = parts[2]
        roots[dataset_name].add(version)
    return {k: sorted(v) for k, v in roots.items()}


def _select_versions(
    *,
    versions_by_dataset: dict[str, list[str]],
    all_latest: bool,
    dataset_name: str,
    version: str,
) -> list[tuple[str, str]]:
    if all_latest:
        selected: list[tuple[str, str]] = []
        for ds_name, versions in sorted(versions_by_dataset.items()):
            if versions:
                selected.append((ds_name, versions[-1]))
        return selected
    if version:
        return [(dataset_name, version)]
    versions = versions_by_dataset.get(dataset_name, [])
    if not versions:
        raise SystemExit(f"No datasets found for '{dataset_name}'")
    return [(dataset_name, versions[-1])]


def _copy_or_extract_dataset(local_repo_path: Path, fetched_dir: Path, extract_archive: bool) -> tuple[list[str], dict | None]:
    manifest_path = local_repo_path / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    archive_paths = sorted(local_repo_path.glob("*.tar.gz"))
    copied_files: list[str] = []
    if extract_archive and archive_paths:
        archive_path = archive_paths[-1]
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(path=fetched_dir)
        copied_files.append(str(archive_path))
    else:
        for src in local_repo_path.iterdir():
            if src.is_file():
                target = fetched_dir / src.name
                target.write_bytes(src.read_bytes())
                copied_files.append(str(src))
            elif src.is_dir():
                target_dir = fetched_dir / src.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(src, target_dir)
                copied_files.append(str(src))
    return copied_files, manifest


def main() -> None:
    args = build_parser().parse_args()
    if not args.repo_id:
        raise SystemExit("HF dataset repo id is required (--repo-id or HF_DATASET_REPO_ID)")
    if not args.all_latest and not args.dataset_name:
        raise SystemExit("--dataset-name is required unless --all-latest is used")
    single_dataset_latest = bool((not args.all_latest) and args.dataset_name and (not args.version))

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    dest_dir = Path(args.dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    repo_path_prefix = args.repo_path_prefix.strip("/ ")
    planned_repo_path = (
        f"{repo_path_prefix}/{args.dataset_name}/{(args.version or '<latest>')}"
        if not args.all_latest
        else f"{repo_path_prefix}/<dataset>/<latest>"
    )
    if args.all_latest:
        allow_patterns = [f"{repo_path_prefix}/**"]
    elif single_dataset_latest:
        allow_patterns = [f"{repo_path_prefix}/{args.dataset_name}/**"]
    else:
        allow_patterns = [f"{planned_repo_path}/**"]

    if args.verbose:
        print(
            {
                "hf_dataset_fetch": {
                    "repo_id": args.repo_id,
                    "repo_path": planned_repo_path,
                    "dest_dir": str(dest_dir),
                    "all_latest": args.all_latest,
                    "extract_archive": args.extract_archive,
                    "hf_transfer_enabled": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
                }
            }
        )

    if args.dry_run:
        print(
            {
                "hf_dataset_fetch_dry_run": {
                    "repo_id": args.repo_id,
                    "repo_type": "dataset",
                    "repo_path": planned_repo_path,
                    "allow_patterns": allow_patterns,
                    "dest_dir": str(dest_dir),
                }
            }
        )
        return

    token = _resolve_token(args.token, args.keyring_service, args.keyring_username)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it in the active Python env "
            "(for example: `.venv/bin/pip install huggingface_hub hf_transfer`). "
            f"Import error: {exc}"
        )

    selected: list[tuple[str, str]]
    if args.all_latest or single_dataset_latest:
        api = HfApi(token=token)
        repo_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
        versions_by_dataset = _parse_repo_dataset_versions(repo_files, repo_path_prefix)
        selected = _select_versions(
            versions_by_dataset=versions_by_dataset,
            all_latest=bool(args.all_latest),
            dataset_name=str(args.dataset_name or ""),
            version=str(args.version or ""),
        )
        if not selected:
            raise SystemExit(f"No datasets found under prefix '{repo_path_prefix}' in repo {args.repo_id}")
        allow_patterns = [f"{repo_path_prefix}/{dataset_name}/{version}/**" for dataset_name, version in selected]
    else:
        selected = [(args.dataset_name, args.version)]

    snapshot_root = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=allow_patterns,
        )
    )

    fetched_summaries = []
    aggregate_train_paths: list[str] = []
    aggregate_val_paths: list[str] = []
    aggregate_by_format: dict[str, dict[str, list[str] | int]] = {}
    for dataset_name, version in selected:
        repo_path = f"{repo_path_prefix}/{dataset_name}/{version}"
        local_repo_path = snapshot_root / repo_path
        if not local_repo_path.exists():
            raise SystemExit(f"Dataset path not found in snapshot: {local_repo_path}")

        fetched_dir = dest_dir / dataset_name / version
        fetched_dir.mkdir(parents=True, exist_ok=True)
        copied_files, manifest = _copy_or_extract_dataset(local_repo_path, fetched_dir, args.extract_archive)

        dataset_root = fetched_dir / "dataset"
        if not dataset_root.is_dir():
            dataset_root = fetched_dir
        train_path = dataset_root / "train.jsonl"
        val_path = dataset_root / "val.jsonl"
        if train_path.is_file():
            aggregate_train_paths.append(str(train_path))
        if val_path.is_file():
            aggregate_val_paths.append(str(val_path))
        ds_format = str((manifest or {}).get("dataset_format") or "unknown")
        slot = aggregate_by_format.setdefault(ds_format, {"train_paths": [], "val_paths": [], "dataset_count": 0})
        if train_path.is_file():
            slot["train_paths"].append(str(train_path))  # type: ignore[index]
        if val_path.is_file():
            slot["val_paths"].append(str(val_path))  # type: ignore[index]
        slot["dataset_count"] = int(slot.get("dataset_count", 0)) + 1  # type: ignore[arg-type]

        fetched_summaries.append(
            {
                "dataset_name": dataset_name,
                "version": version,
                "repo_path": repo_path,
                "dest_dir": str(fetched_dir),
                "snapshot_root": str(snapshot_root),
                "manifest_present": bool(manifest),
                "dataset_format": (manifest or {}).get("dataset_format"),
                "stats_json_present": bool((manifest or {}).get("stats_json")),
                "distribution_mode": (manifest or {}).get("distribution", {}).get("mode"),
                "copied_entries": len(copied_files),
                "train_path": str(train_path) if train_path.is_file() else "",
                "val_path": str(val_path) if val_path.is_file() else "",
            }
        )

    result = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "all_latest": args.all_latest,
        "repo_path_prefix": repo_path_prefix,
        "datasets": fetched_summaries,
        "aggregate": {
            "dataset_count": len(fetched_summaries),
            "train_paths": aggregate_train_paths,
            "val_paths": aggregate_val_paths,
            "train_count": len(aggregate_train_paths),
            "val_count": len(aggregate_val_paths),
        },
        "aggregate_by_format": {
            k: {
                "dataset_count": int(v.get("dataset_count", 0)),
                "train_paths": list(v.get("train_paths", [])),
                "val_paths": list(v.get("val_paths", [])),
                "train_count": len(list(v.get("train_paths", []))),
                "val_count": len(list(v.get("val_paths", []))),
            }
            for k, v in aggregate_by_format.items()
        },
    }
    if args.output_manifest:
        out_path = Path(args.output_manifest).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["output_manifest"] = str(out_path)

    print({"hf_dataset_fetch_complete": result})


if __name__ == "__main__":
    main()
