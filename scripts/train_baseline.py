#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.training import train_next_move_model_from_jsonl_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline next-move predictor")
    parser.add_argument(
        "--train",
        action="append",
        required=True,
        help="Training JSONL path. Repeat flag to combine multiple train datasets.",
    )
    parser.add_argument(
        "--val",
        action="append",
        required=True,
        help="Validation JSONL path. Repeat flag to combine multiple val datasets.",
    )
    parser.add_argument("--output", default="artifacts/model.pt")
    parser.add_argument("--metrics-out", default="artifacts/train_metrics.json")
    parser.add_argument("--progress-jsonl-out", default="", help="Optional JSONL progress event stream path (epoch-level updates)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layer count")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate for embedding/head (and LSTM inter-layer when num_layers>1)")
    parser.add_argument("--phase-feature", action=argparse.BooleanOptionalAction, default=True, help="Concat phase embedding to classifier head")
    parser.add_argument("--phase-embed-dim", type=int, default=8, help="Phase embedding dim when --phase-feature is enabled")
    parser.add_argument("--side-to-move-feature", action=argparse.BooleanOptionalAction, default=True, help="Concat side-to-move embedding to classifier head")
    parser.add_argument("--side-to-move-embed-dim", type=int, default=4, help="Side-to-move embedding dim when --side-to-move-feature is enabled")
    parser.add_argument("--winner-weight", type=float, default=1.2)
    parser.add_argument("--phase-weight-opening", type=float, default=1.0)
    parser.add_argument("--phase-weight-middlegame", type=float, default=1.0)
    parser.add_argument("--phase-weight-endgame", type=float, default=1.0)
    parser.add_argument("--phase-weight-unknown", type=float, default=1.0)
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
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="plateau",
        help="Learning-rate scheduler strategy (plateau watches validation metric)",
    )
    parser.add_argument(
        "--lr-scheduler-metric",
        choices=["val_loss", "top1"],
        default="val_loss",
        help="Validation metric to drive ReduceLROnPlateau",
    )
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--lr-plateau-patience", type=int, default=3, help="Epochs without improvement before LR reduction")
    parser.add_argument("--lr-plateau-threshold", type=float, default=1e-4, help="Absolute improvement threshold for plateau detection")
    parser.add_argument("--lr-plateau-min-lr", type=float, default=0.0, help="Lower bound for scheduler-adjusted LR")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N non-improving validation epochs (0 disables)",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=["val_loss", "top1"],
        default="val_loss",
        help="Validation metric used for early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum metric improvement required to reset early-stopping patience",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose startup/epoch/checkpoint logging",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-epoch batch progress bar (requires --verbose)",
    )
    args = parser.parse_args()
    train_paths = [Path(p).resolve() for p in args.train]
    val_paths = [Path(p).resolve() for p in args.val]
    train_path = train_paths[0]
    val_path = val_paths[0]
    output_path = Path(args.output).resolve()
    metrics_path = Path(args.metrics_out).resolve()
    progress_jsonl_path = Path(args.progress_jsonl_out).resolve() if args.progress_jsonl_out else None

    progress_fh = None

    def emit_progress(event: dict) -> None:
        nonlocal progress_fh
        if progress_jsonl_path is None:
            return
        if progress_fh is None:
            progress_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            progress_fh = progress_jsonl_path.open("a", encoding="utf-8")
        row = {"ts_epoch_ms": int(time.time() * 1000), **event}
        progress_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        progress_fh.flush()

    if args.verbose:
        print(
            {
                "torch_version": torch.__version__,
                "cuda_is_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "requested_device": args.device,
            }
        )
        print(
            {
                "train_start": {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "output_path": str(output_path),
                "metrics_out": str(metrics_path),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "phase_feature": args.phase_feature,
                    "phase_embed_dim": args.phase_embed_dim,
                    "side_to_move_feature": args.side_to_move_feature,
                    "side_to_move_embed_dim": args.side_to_move_embed_dim,
                    "winner_weight": args.winner_weight,
                    "phase_weights": {
                        "unknown": args.phase_weight_unknown,
                        "opening": args.phase_weight_opening,
                        "middlegame": args.phase_weight_middlegame,
                        "endgame": args.phase_weight_endgame,
                    },
                    "use_winner": not args.no_winner_feature,
                    "device_requested": args.device,
                    "num_workers": args.num_workers,
                    "pin_memory_requested": args.pin_memory,
                    "amp_requested": args.amp,
                    "restore_best": args.restore_best,
                    "lr_scheduler": args.lr_scheduler,
                    "lr_scheduler_metric": args.lr_scheduler_metric,
                    "lr_plateau_factor": args.lr_plateau_factor,
                    "lr_plateau_patience": args.lr_plateau_patience,
                    "lr_plateau_threshold": args.lr_plateau_threshold,
                    "lr_plateau_min_lr": args.lr_plateau_min_lr,
                    "early_stopping_patience": args.early_stopping_patience,
                    "early_stopping_metric": args.early_stopping_metric,
                    "early_stopping_min_delta": args.early_stopping_min_delta,
                    "verbose": args.verbose,
                    "progress": args.progress,
                }
            }
        )
        # Preserve prior single-path keys for compatibility while exposing full path lists.
        print(
            {
                "train_inputs": {
                    "train_paths": [str(p) for p in train_paths],
                    "val_paths": [str(p) for p in val_paths],
                    "train_file_count": len(train_paths),
                    "val_file_count": len(val_paths),
                }
            }
        )
    emit_progress(
        {
            "event": "script_start",
            "epochs_requested": int(args.epochs),
            "train_paths": [str(p) for p in train_paths],
            "val_paths": [str(p) for p in val_paths],
            "output_path": str(output_path),
            "metrics_out": str(metrics_path),
            "device_requested": str(args.device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "amp_requested": bool(args.amp),
        }
    )

    try:
        artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
            train_paths=[str(p) for p in train_paths],
            val_paths=[str(p) for p in val_paths],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            winner_weight=args.winner_weight,
            use_phase_feature=args.phase_feature,
            phase_embed_dim=args.phase_embed_dim,
            use_side_to_move_feature=args.side_to_move_feature,
            side_to_move_embed_dim=args.side_to_move_embed_dim,
            phase_weights={
                "unknown": args.phase_weight_unknown,
                "opening": args.phase_weight_opening,
                "middlegame": args.phase_weight_middlegame,
                "endgame": args.phase_weight_endgame,
            },
            use_winner=not args.no_winner_feature,
            device_str=args.device,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            amp=args.amp,
            restore_best=args.restore_best,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_metric=args.lr_scheduler_metric,
            lr_plateau_factor=args.lr_plateau_factor,
            lr_plateau_patience=args.lr_plateau_patience,
            lr_plateau_threshold=args.lr_plateau_threshold,
            lr_plateau_min_lr=args.lr_plateau_min_lr,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_metric=args.early_stopping_metric,
            early_stopping_min_delta=args.early_stopping_min_delta,
            verbose=args.verbose,
            show_progress=args.progress,
            progress_callback=emit_progress,
        )
    except Exception as exc:
        emit_progress({"event": "script_error", "error_type": type(exc).__name__, "message": str(exc)})
        raise

    train_rows_by_file = dataset_info["train_rows_by_file"]
    val_rows_by_file = dataset_info["val_rows_by_file"]
    if args.verbose:
        print(
            {
                "dataset_loaded": {
                    "train_rows": dataset_info["train_rows"],
                    "val_rows": dataset_info["val_rows"],
                    "train_has_rows": bool(dataset_info["train_rows"]),
                    "val_has_rows": bool(dataset_info["val_rows"]),
                    "train_rows_by_file": train_rows_by_file,
                    "val_rows_by_file": val_rows_by_file,
                    "data_loading": dataset_info.get("data_loading"),
                }
            }
        )

    if args.verbose:
        print({"training_complete": {"epochs_ran": len(history), "last_epoch": (history[-1]["epoch"] if history else None)}})

    ensure_parent(args.output)
    torch.save(artifact, args.output)
    summary = {
        "train_rows": dataset_info["train_rows"],
        "val_rows": dataset_info["val_rows"],
        "train_inputs": [str(p) for p in train_paths],
        "val_inputs": [str(p) for p in val_paths],
        "train_rows_by_file": train_rows_by_file,
        "val_rows_by_file": val_rows_by_file,
        "data_loading": dataset_info.get("data_loading"),
        "epochs": args.epochs,
        "history": history,
        "model_path": args.output,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "phase_feature": args.phase_feature,
        "phase_embed_dim": args.phase_embed_dim,
        "side_to_move_feature": args.side_to_move_feature,
        "side_to_move_embed_dim": args.side_to_move_embed_dim,
        "device_requested": args.device,
        "phase_weights": {
            "unknown": args.phase_weight_unknown,
            "opening": args.phase_weight_opening,
            "middlegame": args.phase_weight_middlegame,
            "endgame": args.phase_weight_endgame,
        },
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp": args.amp,
        "restore_best": args.restore_best,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_metric": args.lr_scheduler_metric,
        "lr_plateau_factor": args.lr_plateau_factor,
        "lr_plateau_patience": args.lr_plateau_patience,
        "lr_plateau_threshold": args.lr_plateau_threshold,
        "lr_plateau_min_lr": args.lr_plateau_min_lr,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "verbose": args.verbose,
        "progress": args.progress,
        "best_checkpoint": artifact.get("runtime", {}).get("best_checkpoint"),
        "early_stopping": artifact.get("runtime", {}).get("early_stopping"),
        "lr_scheduler_runtime": artifact.get("runtime", {}).get("lr_scheduler"),
    }
    write_json(args.metrics_out, summary)
    emit_progress(
        {
            "event": "script_complete",
            "epochs_completed": int(len(history)),
            "history_last_epoch": int(history[-1]["epoch"]) if history else None,
            "model_path": str(output_path),
            "metrics_out": str(metrics_path),
            "best_checkpoint": summary.get("best_checkpoint"),
            "early_stopping": summary.get("early_stopping"),
        }
    )
    if args.verbose:
        print({"best_checkpoint": summary.get("best_checkpoint")})
    print(f"Saved model: {args.output}")
    print(f"Saved metrics: {args.metrics_out}")
    if progress_fh is not None:
        progress_fh.close()


if __name__ == "__main__":
    main()
