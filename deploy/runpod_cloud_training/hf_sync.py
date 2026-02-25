#!/usr/bin/env python3
import argparse
import fnmatch
import os
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import HfApi


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _split_patterns(text: str) -> List[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _iter_files(source_dir: Path, patterns: Iterable[str]) -> List[Path]:
    pats = list(patterns)
    out: List[Path] = []
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(source_dir))
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in pats):
            out.append(p)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local artifacts to a Hugging Face repo")
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", ""))
    parser.add_argument("--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"), choices=["model", "dataset", "space"])
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--source-dir", default=os.environ.get("HF_SYNC_SOURCE_DIR", "./artifacts"))
    parser.add_argument("--patterns", default=os.environ.get("HF_SYNC_PATTERNS", "*.pt,*.json"))
    parser.add_argument("--path-in-repo-prefix", default=os.environ.get("HF_PATH_IN_REPO_PREFIX", ""))
    parser.add_argument("--commit-message", default=os.environ.get("HF_COMMIT_MESSAGE", "sync chess-bot artifacts"))
    _bool_arg(parser, "create-repo", True, "Create repo if it does not exist")
    _bool_arg(parser, "verbose", os.environ.get("HF_SYNC_VERBOSE", "1") == "1", "Verbose upload logging")
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit("HF_REPO_ID / --repo-id is required")
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source dir not found: {source_dir}")

    patterns = _split_patterns(args.patterns)
    files = _iter_files(source_dir, patterns)
    if not files:
        print({"hf_sync": "no matching files", "source_dir": str(source_dir), "patterns": patterns})
        return

    api = HfApi(token=args.token)
    if args.create_repo:
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)

    uploaded = []
    for p in files:
        rel = p.relative_to(source_dir).as_posix()
        path_in_repo = f"{args.path_in_repo_prefix.rstrip('/')}/{rel}" if args.path_in_repo_prefix else rel
        if args.verbose:
            print({"hf_upload": {"local": str(p), "path_in_repo": path_in_repo}})
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
        )
        uploaded.append(path_in_repo)

    print(
        {
            "hf_sync_complete": {
                "repo_id": args.repo_id,
                "repo_type": args.repo_type,
                "uploaded_count": len(uploaded),
                "source_dir": str(source_dir),
                "patterns": patterns,
            }
        }
    )


if __name__ == "__main__":
    main()
