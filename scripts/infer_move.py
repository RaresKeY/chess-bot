#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.inference import (
    infer_first_move_dual_artifacts_on_device,
    infer_first_move_auto_from_artifact_on_device,
    infer_from_artifact_on_device,
    infer_rollout_from_artifact_on_device,
    parse_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer best legal move from a context")
    parser.add_argument("--model", default="", help="Single-model artifact path (legacy next-move or single dual-sequence artifact)")
    parser.add_argument("--white-model", default="", help="White-side dual-sequence artifact path")
    parser.add_argument("--black-model", default="", help="Black-side dual-sequence artifact path")
    parser.add_argument("--context", required=True, help="Space-separated UCI moves")
    parser.add_argument("--winner-side", default="W", choices=["W", "B", "D", "?"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--policy-mode",
        choices=["auto", "next", "rollout"],
        default="auto",
        help="Inference policy mode: auto prefers rollout for multistep artifacts; next forces legacy next-move inference",
    )
    parser.add_argument("--rollout-plies", type=int, default=0, help="If >0, generate a continuation rollout of N plies and return first move")
    parser.add_argument(
        "--rollout-fallback-legal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="During rollout generation, use legal fallback moves when no legal model prediction exists",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for inference (default: cpu)")
    args = parser.parse_args()

    print(
        {
            "torch_version": torch.__version__,
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_device": args.device,
        }
    )
    context = parse_context(args.context)

    use_dual_pair = bool(str(args.white_model).strip() and str(args.black_model).strip())
    use_single_model = bool(str(args.model).strip())
    if not use_dual_pair and not use_single_model:
        raise SystemExit("Provide --model or both --white-model and --black-model")

    if use_dual_pair:
        white_artifact = torch.load(args.white_model, map_location="cpu")
        black_artifact = torch.load(args.black_model, map_location="cpu")
        out = infer_first_move_dual_artifacts_on_device(
            white_artifact=white_artifact,
            black_artifact=black_artifact,
            context=context,
            topk=args.topk,
            device_str=args.device,
        )
    else:
        artifact = torch.load(args.model, map_location="cpu")
        if args.policy_mode == "auto" or args.policy_mode == "next" or args.policy_mode == "rollout":
            out = infer_first_move_auto_from_artifact_on_device(
                artifact=artifact,
                context=context,
                winner_side=args.winner_side,
                topk=args.topk,
                device_str=args.device,
                policy_mode=args.policy_mode,
                rollout_plies=int(args.rollout_plies),
                rollout_fallback_legal=bool(args.rollout_fallback_legal),
            )
        elif int(args.rollout_plies) > 0:
            out = infer_rollout_from_artifact_on_device(
                artifact=artifact,
                context=context,
                winner_side=args.winner_side,
                topk=args.topk,
                rollout_plies=int(args.rollout_plies),
                device_str=args.device,
                fallback_legal=bool(args.rollout_fallback_legal),
            )
        else:
            out = infer_from_artifact_on_device(
                artifact=artifact,
                context=context,
                winner_side=args.winner_side,
                topk=args.topk,
                device_str=args.device,
            )
    print(out)


if __name__ == "__main__":
    main()
