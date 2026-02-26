#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.runpod_cycle_verify import verify_full_hf_cycle_run


def main() -> None:
    p = argparse.ArgumentParser(description="Verify a RunPod full-HF cycle local artifact set")
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--require-terminated", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-json", default="")
    args = p.parse_args()

    result = verify_full_hf_cycle_run(args.repo_root, args.run_id, require_terminated=bool(args.require_terminated))
    out = json.dumps(result, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(out + "\n", encoding="utf-8")
    print(out)
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
