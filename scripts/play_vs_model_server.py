#!/usr/bin/env python3
import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer

from src.chessbot.play_vs_model import (
    LoadedMoveModel,
    PlayConfig,
    move_response,
    render_play_page_html,
    state_response,
)


def handler_factory(model_runtime: LoadedMoveModel, play_cfg: PlayConfig, page_path: str, page_html: str):
    page_path = "/" + page_path.strip("/")

    class Handler(SimpleHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            if self.path in {page_path, page_path + "/"}:
                body = page_html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return super().do_GET()

        def do_POST(self):
            if self.path not in {"/api/state", "/api/move"}:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            try:
                payload = self._read_json()
                context = payload.get("context", [])
                if not isinstance(context, list) or not all(isinstance(x, str) for x in context):
                    raise ValueError("context must be a list of UCI strings")

                if self.path == "/api/state":
                    return self._send_json(HTTPStatus.OK, state_response(context))

                user_move = payload.get("user_move", "")
                if not isinstance(user_move, str) or not user_move:
                    raise ValueError("user_move is required")

                cfg = PlayConfig(
                    winner_side=str(payload.get("winner_side", play_cfg.winner_side)),
                    topk=int(payload.get("topk", play_cfg.topk)),
                    user_color=str(payload.get("user_color", play_cfg.user_color)),
                )
                result = move_response(model_runtime=model_runtime, context=context, user_move=user_move, cfg=cfg)
                return self._send_json(HTTPStatus.OK, result)
            except Exception as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local Play-vs-Model chess web app")
    parser.add_argument("--model", default="artifacts/model.pt", help="Model artifact path")
    parser.add_argument("--dir", default=".", help="Directory to serve as HTTP document root")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--page-path", default="play-vs-model", help="URL path for app page")
    parser.add_argument("--piece-base", default="assets/pieces/cburnett", help="URL path prefix for piece assets")
    parser.add_argument("--winner-side", default="B", choices=["W", "B", "D", "?"])
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    serve_dir = str(Path(args.dir).resolve())
    model_path = str(Path(args.model).resolve())
    os.chdir(serve_dir)

    model_runtime = LoadedMoveModel.from_path(model_path)
    play_cfg = PlayConfig(winner_side=args.winner_side, topk=args.topk, user_color="white")
    page_html = render_play_page_html(
        title="Play vs Chess Model",
        piece_base=args.piece_base,
        default_winner_side=args.winner_side,
        default_topk=args.topk,
    )
    handler = handler_factory(model_runtime=model_runtime, play_cfg=play_cfg, page_path=args.page_path, page_html=page_html)

    with ThreadingTCPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {serve_dir} at http://{args.bind}:{args.port}/")
        print(f"Play URL: http://{args.bind}:{args.port}/{'/'.join([args.page_path.strip('/')])}")
        print(f"Model: {model_path}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
