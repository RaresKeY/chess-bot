#!/usr/bin/env python3
import argparse
import os

import torch

from src.chessbot.evaluation import evaluate_artifact
from src.chessbot.io_utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained baseline model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/eval_metrics.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation (default: cpu, e.g. cuda:0)",
    )
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
    rows = list(read_jsonl(args.data))
    out = evaluate_artifact(artifact=artifact, rows=rows, batch_size=args.batch_size, device_str=args.device)
    write_json(args.output, out)
    print(out)


if __name__ == "__main__":
    main()
