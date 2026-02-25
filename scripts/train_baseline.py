#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.chessbot.io_utils import ensure_parent, read_jsonl, write_json
from src.chessbot.training import train_next_move_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline next-move predictor")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", default="artifacts/model.pt")
    parser.add_argument("--metrics-out", default="artifacts/train_metrics.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1, help="LSTM layer count")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for embedding/head (and LSTM inter-layer when num_layers>1)")
    parser.add_argument("--winner-weight", type=float, default=1.2)
    parser.add_argument("--no-winner-feature", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use (default: auto, e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DataLoader pin_memory (auto-disabled on CPU)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA mixed precision when training on GPU",
    )
    parser.add_argument(
        "--restore-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore best validation checkpoint (lowest val_loss) before saving when validation rows exist",
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

    train_rows = list(read_jsonl(args.train))
    val_rows = list(read_jsonl(args.val))
    if not train_rows:
        raise SystemExit("No training rows found")

    artifact, history = train_next_move_model(
        train_rows=train_rows,
        val_rows=val_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        winner_weight=args.winner_weight,
        use_winner=not args.no_winner_feature,
        device_str=args.device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
        restore_best=args.restore_best,
    )

    for row in history:
        print(row)

    ensure_parent(args.output)
    torch.save(artifact, args.output)
    summary = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "epochs": args.epochs,
        "history": history,
        "model_path": args.output,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "device_requested": args.device,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp": args.amp,
        "restore_best": args.restore_best,
        "best_checkpoint": artifact.get("runtime", {}).get("best_checkpoint"),
    }
    write_json(args.metrics_out, summary)
    print(f"Saved model: {args.output}")
    print(f"Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
