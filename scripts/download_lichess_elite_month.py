#!/usr/bin/env python3
import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(max(0, n))
    for unit in units:
        if x < 1024 or unit == units[-1]:
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024.0
    return f"{n}B"


def _print_bar(prefix: str, done: int, total: int) -> None:
    width = 28
    if total > 0:
        frac = min(1.0, max(0.0, done / total))
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        pct = f"{frac * 100:5.1f}%"
        msg = f"\r{prefix} [{bar}] {pct} {_fmt_bytes(done)}/{_fmt_bytes(total)}"
    else:
        msg = f"\r{prefix} {_fmt_bytes(done)}"
    sys.stdout.write(msg)
    sys.stdout.flush()


def _looks_like_zip_signature(buf: bytes) -> bool:
    return any(buf.startswith(sig) for sig in ZIP_SIGNATURES)


def _read_head(path: Path, n: int = 512) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _snippet_for_error(path: Path, limit: int = 200) -> str:
    raw = _read_head(path, limit)
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = repr(raw)
    return " ".join(text.split())


def validate_zip_file(zip_path: Path, *, context: str) -> None:
    if not zip_path.exists():
        raise RuntimeError(f"{context}: missing file: {zip_path}")
    try:
        head = _read_head(zip_path, 8)
    except OSError as exc:
        raise RuntimeError(f"{context}: unable to read {zip_path}: {exc}") from exc
    if not _looks_like_zip_signature(head):
        snippet = _snippet_for_error(zip_path)
        raise RuntimeError(
            f"{context}: file is not a ZIP (bad signature) at {zip_path}; "
            f"likely HTML/error page. head={head!r} snippet={snippet!r}"
        )
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{context}: invalid/corrupt ZIP file at {zip_path}")


def download_file(url: str, out_path: Path, chunk_size: int = 2 * 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "chess-bot-downloader/1.0"})
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    content_type = ""
    done = 0
    total = 0
    first_bytes = b""
    try:
        with urlopen(req) as resp, open(tmp_path, "wb") as f:
            content_type = (resp.headers.get("Content-Type", "") or "").lower()
            total = int(resp.headers.get("Content-Length", "0") or 0)
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                if not first_bytes:
                    first_bytes = chunk[:16]
                f.write(chunk)
                done += len(chunk)
                _print_bar("Downloading", done, total)
        if total:
            _print_bar("Downloading", total, total)
        sys.stdout.write("\n")
        # Reject obvious HTML/error responses before caching as .zip.
        if "html" in content_type and not _looks_like_zip_signature(first_bytes):
            snippet = _snippet_for_error(tmp_path)
            raise RuntimeError(
                f"downloaded response is HTML, not ZIP (content-type={content_type!r}); "
                f"likely unavailable month or upstream error. snippet={snippet!r}"
            )
        validate_zip_file(tmp_path, context="downloaded file")
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_zip_file(zip_path, context="extract input")
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        total_bytes = sum(max(0, i.file_size) for i in infos)
        done_bytes = 0
        for info in infos:
            zf.extract(info, path=out_dir)
            done_bytes += max(0, info.file_size)
            _print_bar("Extracting ", done_bytes, total_bytes)
    if total_bytes:
        _print_bar("Extracting ", total_bytes, total_bytes)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract a Lichess elite monthly dump ZIP")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format, e.g. 2025-10")
    parser.add_argument(
        "--url-template",
        default="https://database.nikonoel.fr/lichess_elite_{month}.zip",
        help="Download URL template; '{month}' will be replaced",
    )
    parser.add_argument("--zip-out", default="data/raw/elite", help="Directory for downloaded ZIP file")
    parser.add_argument("--extract-root", default="data/raw/elite", help="Root directory for extracted month folder")
    parser.add_argument("--skip-download-if-exists", action="store_true", default=True)
    parser.add_argument("--re-download", action="store_true", help="Force download even if ZIP already exists")
    args = parser.parse_args()

    month = args.month.strip()
    if len(month) != 7 or month[4] != "-":
        raise SystemExit("Expected --month in YYYY-MM format")

    url = args.url_template.format(month=month)
    zip_out_dir = Path(args.zip_out).resolve()
    extract_root = Path(args.extract_root).resolve()
    zip_path = zip_out_dir / f"lichess_elite_{month}.zip"
    extract_dir = extract_root / month

    print({"month": month, "url": url, "zip_path": str(zip_path), "extract_dir": str(extract_dir)})
    if zip_path.exists() and not args.re_download:
        try:
            validate_zip_file(zip_path, context="cached ZIP")
            print(f"ZIP already exists and is valid, skipping download: {zip_path}")
        except RuntimeError as exc:
            print(f"Cached ZIP invalid; removing and re-downloading: {zip_path}")
            print({"cached_zip_invalid": str(exc)})
            zip_path.unlink(missing_ok=True)
            download_file(url, zip_path)
    else:
        download_file(url, zip_path)

    extract_zip(zip_path, extract_dir)
    pgns = sorted(extract_dir.rglob("*.pgn"))
    print(
        {
            "download_complete": {
                "zip_path": str(zip_path),
                "extract_dir": str(extract_dir),
                "pgn_files_found": [str(p) for p in pgns],
                "pgn_count": len(pgns),
            }
        }
    )


if __name__ == "__main__":
    main()
