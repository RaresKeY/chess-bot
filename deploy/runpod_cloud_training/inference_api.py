#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _progress(step: int, total: int, label: str) -> None:
    width = 24
    frac = min(1.0, max(0.0, step / max(total, 1)))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[inference-api] load [{bar}] {step}/{total} {label}")
    sys.stderr.flush()
    if step >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _resolve_repo_imports(repo_dir: Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


class InferRequest(BaseModel):
    context: List[str] = Field(default_factory=list)
    winner_side: str = Field(default="W")
    topk: int | None = Field(default=None)


class InferenceRuntime:
    def __init__(self, repo_dir: Path, model_path: Path, device_str: str = "auto", verbose: bool = True) -> None:
        _resolve_repo_imports(repo_dir)
        from src.chessbot.inference import best_legal_from_topk
        from src.chessbot.model import NextMoveLSTM, encode_tokens, side_to_move_id_from_context_len, winner_to_id
        from src.chessbot.phase import PHASE_UNKNOWN, classify_context_phase, phase_to_id

        self.best_legal_from_topk = best_legal_from_topk
        self.encode_tokens = encode_tokens
        self.side_to_move_id_from_context_len = side_to_move_id_from_context_len
        self.winner_to_id = winner_to_id
        self.classify_context_phase = classify_context_phase
        self.phase_to_id = phase_to_id
        self.phase_unknown = PHASE_UNKNOWN
        self.model_cls = NextMoveLSTM
        self.repo_dir = repo_dir
        self.model_path = model_path
        self.verbose = verbose

        t0 = time.perf_counter()
        artifact = torch.load(str(model_path), map_location="cpu")
        self.artifact = artifact
        self.vocab = artifact["vocab"]
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        cfg = artifact["config"]

        if device_str == "auto":
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device(device_str)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

        self.model = self.model_cls(vocab_size=len(self.vocab), **cfg).to(self.device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.load_seconds = time.perf_counter() - t0
        if self.verbose:
            print(
                {
                    "inference_runtime": {
                        "model_path": str(self.model_path),
                        "device": str(self.device),
                        "vocab_size": len(self.vocab),
                        "model_config": cfg,
                        "load_seconds": round(self.load_seconds, 3),
                    }
                }
            )

    def infer(self, context: List[str], winner_side: str, topk: int) -> dict:
        if not isinstance(context, list) or not all(isinstance(x, str) for x in context):
            raise ValueError("context must be a list of UCI strings")
        original_context_len = len(context)
        context_ids = self.encode_tokens(context, self.vocab)
        if not context_ids:
            context_ids = [self.unk_id]
        tokens = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(context_ids)], dtype=torch.long, device=self.device)
        winners = torch.tensor([self.winner_to_id(winner_side)], dtype=torch.long, device=self.device)
        phase_name = str(self.classify_context_phase(context).get("phase", self.phase_unknown))
        phases = torch.tensor([self.phase_to_id(phase_name)], dtype=torch.long, device=self.device)
        side_to_moves = torch.tensor(
            [self.side_to_move_id_from_context_len(original_context_len)], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            logits = self.model(tokens, lengths, winners, phases, side_to_moves)
            k = max(1, min(int(topk), int(logits.shape[-1])))
            pred_ids = logits.topk(k, dim=1).indices[0].detach().cpu().tolist()
        topk_tokens = [self.inv_vocab.get(i, "") for i in pred_ids]
        return {
            "topk": topk_tokens,
            "best_legal": self.best_legal_from_topk(topk_tokens, context),
            "predicted_uci": topk_tokens[0] if topk_tokens else "",
            "device": str(self.device),
            "model_path": str(self.model_path),
        }


def _find_latest_model(root: Path) -> Path:
    candidates = [p for p in root.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No model artifacts found under {root}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _resolve_model_path(repo_dir: Path, model_arg: str) -> Path:
    if not model_arg or model_arg == "latest":
        return _find_latest_model(repo_dir / "artifacts")
    p = Path(model_arg)
    if not p.is_absolute():
        p = (repo_dir / p).resolve()
    return p


def _touch_activity(path_text: str | None) -> None:
    if not path_text:
        return
    try:
        p = Path(path_text)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except Exception:
        pass


def build_app(runtime: InferenceRuntime, topk_default: int, verbose: bool, activity_file: str | None) -> FastAPI:
    app = FastAPI(title="Chess Bot Inference API", version="1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        _touch_activity(activity_file)
        return {
            "ok": True,
            "device": str(runtime.device),
            "model_path": str(runtime.model_path),
            "load_seconds": runtime.load_seconds,
        }

    @app.post("/infer")
    def infer(req: InferRequest) -> dict:
        t0 = time.perf_counter()
        try:
            out = runtime.infer(req.context, req.winner_side, req.topk or topk_default)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        out["latency_ms"] = round(latency_ms, 3)
        _touch_activity(activity_file)
        if verbose:
            print(
                {
                    "infer_request": {
                        "context_len": len(req.context),
                        "winner_side": req.winner_side,
                        "topk": int(req.topk or topk_default),
                        "latency_ms": round(latency_ms, 3),
                        "predicted_uci": out.get("predicted_uci", ""),
                        "best_legal": out.get("best_legal", ""),
                    }
                }
            )
        return out

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a simple chess move inference HTTP API")
    parser.add_argument("--repo-dir", default=os.environ.get("REPO_DIR", "/workspace/chess-bot"))
    parser.add_argument("--model", default=os.environ.get("INFERENCE_API_MODEL_PATH", "latest"))
    parser.add_argument("--host", default=os.environ.get("INFERENCE_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INFERENCE_API_PORT", "8000")))
    parser.add_argument("--device", default=os.environ.get("INFERENCE_API_DEVICE", "auto"))
    parser.add_argument("--topk-default", type=int, default=int(os.environ.get("INFERENCE_API_TOPK_DEFAULT", "10")))
    parser.add_argument(
        "--activity-file",
        default=os.environ.get("ACTIVITY_HEARTBEAT_FILE", "/tmp/chessbot_last_activity"),
        help="Heartbeat file updated on API requests for idle watchdog coordination",
    )
    _bool_arg(parser, "verbose", os.environ.get("INFERENCE_API_VERBOSE", "1") == "1", "Verbose request/runtime logs")
    _bool_arg(parser, "load-progress", True, "Show startup model loading progress bar")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    total_steps = 4
    if args.load_progress:
        _progress(1, total_steps, "resolve repo")
    _resolve_repo_imports(repo_dir)
    if args.load_progress:
        _progress(2, total_steps, "resolve model path")
    model_path = _resolve_model_path(repo_dir, args.model)
    if args.load_progress:
        _progress(3, total_steps, "load model")
    runtime = InferenceRuntime(repo_dir=repo_dir, model_path=model_path, device_str=args.device, verbose=args.verbose)
    if args.load_progress:
        _progress(4, total_steps, "ready")

    app = build_app(runtime, topk_default=args.topk_default, verbose=args.verbose, activity_file=args.activity_file)
    print(
        {
            "inference_api_start": {
                "host": args.host,
                "port": args.port,
                "device": str(runtime.device),
                "model_path": str(model_path),
                "topk_default": args.topk_default,
                "activity_file": args.activity_file,
            }
        }
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
