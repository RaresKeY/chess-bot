#!/usr/bin/env python3
import argparse
import os

import torch

from src.chessbot.inference import infer_from_artifact_on_device, parse_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer best legal move from a context")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True, help="Space-separated UCI moves")
    parser.add_argument("--winner-side", default="W", choices=["W", "B", "D", "?"])
    parser.add_argument("--topk", type=int, default=10)
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
    artifact = torch.load(args.model, map_location="cpu")
    context = parse_context(args.context)
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
