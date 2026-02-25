#!/usr/bin/env python3
import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


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


def download_file(url: str, out_path: Path, chunk_size: int = 2 * 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "chess-bot-downloader/1.0"})
    with urlopen(req) as resp, open(out_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            _print_bar("Downloading", done, total)
    if total:
        _print_bar("Downloading", total, total)
    sys.stdout.write("\n")


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"ZIP already exists, skipping download: {zip_path}")
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
