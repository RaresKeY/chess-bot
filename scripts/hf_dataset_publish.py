#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_token(explicit_token: str | None, service: str, username: str) -> str:
    dotenv_paths = default_dotenv_paths(
        repo_root=REPO_ROOT,
        override_var_names=("HF_DOTENV_PATH", "CHESSBOT_DOTENV_PATH"),
        fallback_filenames=(".env.hf_dataset", ".env"),
    )
    token, _ = resolve_secret(
        explicit_value=str(explicit_token or ""),
        env_var_names=("HF_WRITE_TOKEN", "HF_TOKEN"),
        keyring_service=service,
        keyring_username=username,
        dotenv_keys=("HF_WRITE_TOKEN", "HF_TOKEN"),
        dotenv_paths=dotenv_paths,
        order=("explicit", "env", "keyring", "dotenv"),
    )
    if token:
        return token
    dotenv_label = ", ".join(str(p) for p in dotenv_paths)
    raise SystemExit(
        "Missing HF token. Checked --token, HF_WRITE_TOKEN, HF_TOKEN, "
        f"keyring(service={service!r}, username={username!r}), "
        f"and dotenv paths [{dotenv_label}]."
    )


@dataclass(frozen=True)
class FileMeta:
    rel_path: str
    size_bytes: int
    sha256: str


def _iter_dataset_files(dataset_dir: Path) -> Iterable[Path]:
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file():
            yield path


def _collect_file_meta(dataset_dir: Path) -> list[FileMeta]:
    out: list[FileMeta] = []
    for path in _iter_dataset_files(dataset_dir):
        out.append(
            FileMeta(
                rel_path=path.relative_to(dataset_dir).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
    return out


def _validate_runtime_splice_cache(dataset_dir: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> None:
    cache_root = dataset_dir / "runtime_splice_cache"
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing runtime splice cache manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid runtime splice cache manifest JSON: {manifest_path} ({exc})")
    if str(manifest.get("kind", "")) != "runtime_splice_cache":
        raise SystemExit(f"Unexpected runtime splice cache kind in {manifest_path}: {manifest.get('kind')!r}")

    for split in splits:
        split_dir = cache_root / split
        required = (
            "paths.json",
            "path_ids.u32.bin",
            "offsets.u64.bin",
            "splice_indices.u32.bin",
            "sample_phase_ids.u8.bin",
        )
        for name in required:
            fp = split_dir / name
            if not fp.is_file():
                raise SystemExit(f"Missing runtime splice cache file: {fp}")


def _resolve_dataset_dirs(dataset_dir: str, dataset_root: str, dataset_glob: str) -> list[Path]:
    out: list[Path] = []
    if dataset_dir:
        p = Path(dataset_dir).resolve()
        if not p.is_dir():
            raise SystemExit(f"Dataset dir not found: {p}")
        out.append(p)
    if dataset_root:
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            raise SystemExit(f"Dataset root not found: {root}")
        out.extend(sorted(p.resolve() for p in root.glob(dataset_glob) if p.is_dir()))
    if not out:
        raise SystemExit("Provide --dataset-dir or --dataset-root.")
    # stable unique
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    if not dedup:
        raise SystemExit("No dataset dirs matched.")
    return dedup


def _read_stats_json(dataset_dir: Path) -> dict | None:
    stats_path = dataset_dir / "stats.json"
    if not stats_path.is_file():
        return None
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_runtime_splice_cache_manifest(dataset_dir: Path) -> dict | None:
    cache_manifest = dataset_dir / "runtime_splice_cache" / "manifest.json"
    if not cache_manifest.is_file():
        return None
    try:
        data = json.loads(cache_manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _detect_dataset_format(dataset_dir: Path, stats: dict | None) -> str:
    if isinstance(stats, dict):
        fmt = str(stats.get("dataset_format", "")).strip()
        if fmt:
            return fmt
    # Lightweight fallback sniff
    train_path = dataset_dir / "train.jsonl"
    if train_path.is_file():
        try:
            with train_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "moves" in row or "moves_uci" in row:
                        return "game_jsonl_runtime_splice_v1"
                    if "context" in row:
                        return "splice_rows_legacy"
                    break
        except Exception:
            pass
    return "unknown"


def _write_manifest(stage_dir: Path, *, dataset_dir: Path, dataset_name: str, version: str, metas: list[FileMeta], archive_name: str | None) -> None:
    total_bytes = sum(m.size_bytes for m in metas)
    stats_json = _read_stats_json(dataset_dir)
    runtime_splice_cache = _read_runtime_splice_cache_manifest(dataset_dir)
    dataset_format = _detect_dataset_format(dataset_dir, stats_json)
    manifest = {
        "schema_version": 1,
        "kind": "chess-bot-validated-dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "version": version,
        "source_dir": str(dataset_dir),
        "file_count": len(metas),
        "total_bytes": total_bytes,
        "dataset_format": dataset_format,
        "runtime_splice_cache": runtime_splice_cache,
        "required_files_present": {
            "train.jsonl": any(m.rel_path == "train.jsonl" for m in metas),
            "val.jsonl": any(m.rel_path == "val.jsonl" for m in metas),
        },
        "stats_json": stats_json,
        "files": [
            {"path": m.rel_path, "size_bytes": m.size_bytes, "sha256": m.sha256}
            for m in metas
        ],
        "distribution": {
            "mode": "archive" if archive_name else "files",
            "archive_name": archive_name,
        },
    }
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{m.sha256}  {m.rel_path}" for m in metas]
    (stage_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")


def _stage_archive(stage_dir: Path, dataset_dir: Path, archive_basename: str) -> str:
    archive_name = f"{archive_basename}.tar.gz"
    archive_path = stage_dir / archive_name
    with tarfile.open(archive_path, mode="w:gz") as tf:
        tf.add(dataset_dir, arcname="dataset")
    return archive_name


def _stage_files(stage_dir: Path, dataset_dir: Path) -> None:
    files_root = stage_dir / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    for src in _iter_dataset_files(dataset_dir):
        dst = files_root / src.relative_to(dataset_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish a validated dataset directory to a Hugging Face dataset repo")
    p.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO_ID", os.environ.get("HF_REPO_ID", "")))
    p.add_argument("--token", default=None)
    p.add_argument(
        "--keyring-service",
        default=os.environ.get("HF_WRITE_KEYRING_SERVICE", os.environ.get("HF_KEYRING_SERVICE", "huggingface")),
    )
    p.add_argument(
        "--keyring-username",
        default=os.environ.get("HF_WRITE_KEYRING_USERNAME", os.environ.get("HF_KEYRING_USERNAME", "codex_hf_write_token")),
    )
    p.add_argument("--dataset-dir", default="", help="Local dataset directory (typically contains train.jsonl and val.jsonl)")
    p.add_argument("--dataset-root", default="", help="Optional root directory for multi-dataset publish")
    p.add_argument("--dataset-glob", default="elite_*_game", help="Glob pattern under --dataset-root")
    p.add_argument("--dataset-name", default="", help="Logical dataset name (defaults to dataset dir basename)")
    p.add_argument("--version", default="", help="Dataset version label/path segment (default: UTC timestamp)")
    p.add_argument("--repo-path-prefix", default=os.environ.get("HF_DATASET_PATH_PREFIX", "validated_datasets"))
    p.add_argument("--archive-format", choices=["tar.gz", "none"], default=os.environ.get("HF_DATASET_ARCHIVE_FORMAT", "tar.gz"))
    p.add_argument("--commit-message", default="")
    _bool_arg(p, "create-repo", True, "Create the dataset repo if missing")
    _bool_arg(p, "private", False, "Create repo as private when creating")
    _bool_arg(p, "require-train-val", True, "Require train.jsonl and val.jsonl in dataset dir")
    _bool_arg(p, "require-runtime-cache", True, "Require runtime_splice_cache for game datasets")
    _bool_arg(p, "dry-run", False, "Prepare manifest/staging and print upload plan without network upload")
    _bool_arg(p, "verbose", True, "Verbose upload logging")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.repo_id:
        raise SystemExit("HF dataset repo id is required (--repo-id or HF_DATASET_REPO_ID)")

    version = args.version.strip() or _utc_ts()
    dataset_dirs = _resolve_dataset_dirs(args.dataset_dir, args.dataset_root, args.dataset_glob)
    repo_path_prefix = args.repo_path_prefix.strip("/ ")
    commit_message = args.commit_message or f"upload datasets ({len(dataset_dirs)}) version={version}"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    token = None
    api = None
    if not args.dry_run:
        token = _resolve_token(args.token, args.keyring_service, args.keyring_username)
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise SystemExit(
                "huggingface_hub is required. Install it in the active Python env "
                "(for example: `.venv/bin/pip install huggingface_hub hf_transfer`). "
                f"Import error: {exc}"
            )
        api = HfApi(token=token)
        if args.create_repo:
            api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    completed: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="hf_dataset_publish_") as tmp:
        for dataset_dir in dataset_dirs:
            dataset_name = args.dataset_name.strip() or dataset_dir.name
            if args.require_train_val:
                for required in ("train.jsonl", "val.jsonl"):
                    if not (dataset_dir / required).is_file():
                        raise SystemExit(f"Missing required file: {dataset_dir / required}")
            dataset_format = _detect_dataset_format(dataset_dir, _read_stats_json(dataset_dir))
            if args.require_runtime_cache and dataset_format == "game_jsonl_runtime_splice_v1":
                _validate_runtime_splice_cache(dataset_dir)

            metas = _collect_file_meta(dataset_dir)
            if not metas:
                raise SystemExit(f"No files found under dataset dir: {dataset_dir}")
            path_in_repo = "/".join(part for part in (repo_path_prefix, dataset_name, version) if part)

            stage_dir = Path(tmp) / f"stage_{dataset_name}"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
            archive_name: str | None = None
            if args.archive_format == "tar.gz":
                archive_name = _stage_archive(stage_dir, dataset_dir, f"{dataset_name}_{version}")
            else:
                _stage_files(stage_dir, dataset_dir)
            _write_manifest(
                stage_dir,
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                version=version,
                metas=metas,
                archive_name=archive_name,
            )

            if args.verbose:
                print(
                    {
                        "hf_dataset_publish": {
                            "repo_id": args.repo_id,
                            "path_in_repo": path_in_repo,
                            "dataset_dir": str(dataset_dir),
                            "archive_format": args.archive_format,
                            "file_count": len(metas),
                            "total_bytes": sum(m.size_bytes for m in metas),
                            "hf_transfer_enabled": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
                        }
                    }
                )

            if args.dry_run:
                print(
                    {
                        "hf_dataset_publish_dry_run": {
                            "repo_id": args.repo_id,
                            "repo_type": "dataset",
                            "path_in_repo": path_in_repo,
                            "archive_format": args.archive_format,
                            "dataset_dir": str(dataset_dir),
                            "stage_entries": sorted(p.relative_to(stage_dir).as_posix() for p in stage_dir.rglob("*")),
                        }
                    }
                )
            else:
                assert api is not None
                api.upload_folder(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    folder_path=str(stage_dir),
                    path_in_repo=path_in_repo,
                    commit_message=commit_message,
                )

            completed.append(
                {
                    "dataset_name": dataset_name,
                    "version": version,
                    "path_in_repo": path_in_repo,
                    "source_dir": str(dataset_dir),
                    "archive_format": args.archive_format,
                    "file_count": len(metas),
                    "total_bytes": sum(m.size_bytes for m in metas),
                }
            )

    print(
        {
            "hf_dataset_publish_complete": {
                "repo_id": args.repo_id,
                "repo_type": "dataset",
                "published_count": len(completed),
                "datasets": completed,
            }
        }
    )


if __name__ == "__main__":
    main()
