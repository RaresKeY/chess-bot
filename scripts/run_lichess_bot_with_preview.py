#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_bin() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or "python3"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Lichess bot and local live-preview server together")
    p.add_argument("--model", default="latest", help="Model artifact path or 'latest'")
    p.add_argument("--preview-dir", default="artifacts/lichess_live_preview", help="Directory for live preview files")
    p.add_argument("--preview-port", type=int, default=8010, help="Local preview server port")
    p.add_argument("--preview-bind", default="127.0.0.1", help="Local preview server bind address")
    p.add_argument("--log-jsonl", default="artifacts/lichess_bot/bot.jsonl", help="Append bot logs to this JSONL file")
    p.add_argument("--min-request-interval-ms", type=int, default=1500, help="Minimum delay between Lichess API requests")
    p.add_argument("--winner-side", default="W", choices=["W", "B", "D", "?"], help="Model conditioning token")
    p.add_argument("--topk", type=int, default=10, help="Top-k predictions for legal move selection")
    p.add_argument("--accept-rated", action=argparse.BooleanOptionalAction, default=False, help="Allow rated challenges")
    p.add_argument("--variants", default="standard", help="Comma-separated allowed variants")
    p.add_argument("--min-initial-seconds", type=int, default=0, help="Decline games faster than this initial clock")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Do not post moves to Lichess")
    p.add_argument("--token", default="", help="Optional token override (otherwise keyring/env used by bot script)")
    p.add_argument("--keyring-service", default="lichess", help="Keyring service for token lookup")
    p.add_argument("--keyring-username", default="lichess_api_token", help="Keyring username for token lookup")
    return p


def _child_preexec() -> None:
    # New session/process-group so wrapper can kill descendants as a unit.
    os.setsid()
    # Linux-only safety net: if wrapper dies unexpectedly, child receives SIGTERM.
    if sys.platform.startswith("linux"):
        try:
            import ctypes  # local import to avoid unnecessary dependency at module import time

            libc = ctypes.CDLL(None)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        except Exception:
            pass


def _start_process(cmd: List[str], name: str) -> subprocess.Popen:
    # Isolate child process groups and set a parent-death signal (Linux) so
    # detached children do not survive wrapper termination in host/TTY edge cases.
    if os.name == "posix":
        return subprocess.Popen(cmd, cwd=str(REPO_ROOT), preexec_fn=_child_preexec)
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), start_new_session=True)


def main() -> int:
    args = _build_parser().parse_args()
    py = _python_bin()

    preview_dir = str(Path(args.preview_dir))
    os.makedirs(preview_dir, exist_ok=True)

    preview_cmd = [
        py,
        str(REPO_ROOT / "scripts" / "serve_lichess_preview.py"),
        "--dir",
        preview_dir,
        "--port",
        str(args.preview_port),
        "--bind",
        str(args.preview_bind),
        "--keyring-service",
        args.keyring_service,
        "--keyring-username",
        args.keyring_username,
        "--min-request-interval-ms",
        str(args.min_request_interval_ms),
    ]
    bot_cmd = [
        py,
        str(REPO_ROOT / "scripts" / "lichess_bot.py"),
        "--model",
        args.model,
        "--preview-live-dir",
        preview_dir,
        "--log-jsonl",
        args.log_jsonl,
        "--min-request-interval-ms",
        str(args.min_request_interval_ms),
        "--winner-side",
        args.winner_side,
        "--topk",
        str(args.topk),
        "--variants",
        args.variants,
        "--min-initial-seconds",
        str(args.min_initial_seconds),
        "--keyring-service",
        args.keyring_service,
        "--keyring-username",
        args.keyring_username,
    ]
    bot_cmd.append("--accept-rated" if args.accept_rated else "--no-accept-rated")
    bot_cmd.append("--dry-run" if args.dry_run else "--no-dry-run")
    if args.token:
        bot_cmd.extend(["--token", args.token])
        preview_cmd.extend(["--token", args.token])

    print({"preview_url": f"http://{args.preview_bind}:{args.preview_port}/index.html", "preview_dir": str(Path(preview_dir).resolve())})
    print({"preview_cmd": preview_cmd})
    print({"bot_cmd": bot_cmd})
    sys.stdout.flush()

    preview_proc = _start_process(preview_cmd, "preview")
    time.sleep(0.25)
    bot_proc = _start_process(bot_cmd, "bot")

    children = [bot_proc, preview_proc]
    shutdown_done = False

    def _terminate_proc(proc: subprocess.Popen, sig: int) -> None:
        if proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except ProcessLookupError:
            return
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                return

    def _shutdown(sig_name: str) -> None:
        nonlocal shutdown_done
        if shutdown_done:
            return
        shutdown_done = True
        print({"event": "wrapper_shutdown", "signal": sig_name})
        sys.stdout.flush()
        for proc in children:
            _terminate_proc(proc, signal.SIGTERM)
        deadline = time.time() + 5.0
        while time.time() < deadline and any(p.poll() is None for p in children):
            time.sleep(0.1)
        for proc in children:
            _terminate_proc(proc, signal.SIGKILL)
        for proc in children:
            try:
                proc.wait(timeout=0.2)
            except Exception:
                continue

    def _handle_signal(signum, _frame):  # type: ignore[no-untyped-def]
        _shutdown(signal.Signals(signum).name)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = 0
    try:
        while True:
            if preview_proc.poll() is not None:
                print({"event": "wrapper_child_exit", "child": "preview", "returncode": preview_proc.returncode})
                exit_code = preview_proc.returncode or 0
                break
            if bot_proc.poll() is not None:
                print({"event": "wrapper_child_exit", "child": "bot", "returncode": bot_proc.returncode})
                exit_code = bot_proc.returncode or 0
                break
            time.sleep(0.25)
    finally:
        _shutdown("FINALIZE")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
