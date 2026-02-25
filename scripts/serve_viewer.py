#!/usr/bin/env python3
import argparse
import http.server
import os
import socketserver
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve generated viewer files over local HTTP")
    parser.add_argument("--dir", default="/work", help="Directory to serve")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address")
    parser.add_argument(
        "--example-path",
        default="",
        help="Optional path (relative to --dir) to print as a clickable example URL",
    )
    args = parser.parse_args()

    serve_dir = str(Path(args.dir).resolve())
    os.chdir(serve_dir)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {serve_dir} at http://{args.bind}:{args.port}/")
        if args.example_path:
            example = args.example_path.lstrip("/")
            print(f"Example URL: http://{args.bind}:{args.port}/{example}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
