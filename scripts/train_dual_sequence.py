#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.training_dual_sequence import (
    TRAINING_ARCHITECTURE_ALLPLAY_BOOTSTRAP,
    TRAINING_ARCHITECTURE_WINNER_SPLIT,
    MODEL_FAMILY_DUAL_SEQUENCE,
    MODEL_FAMILY_DUAL_SEQUENCE_BOARD,
    MODEL_SIDE_BLACK,
    MODEL_SIDE_WHITE,
    train_bootstrap_dual_head_curriculum_from_jsonl_paths,
    train_dual_sequence_model_from_jsonl_paths,
)


def _resolve_device(device_arg: str) -> str:
    requested = str(device_arg or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return requested


def _summarize_side_result(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics", {})
    history = list(metrics.get("history", []) or [])
    final_row = history[-1] if history else {}
    return {
        "model_side": metrics.get("model_side"),
        "train_rows_total": metrics.get("train_rows_total"),
        "val_rows_total": metrics.get("val_rows_total"),
        "train_dropped_rows_total": metrics.get("train_dropped_rows_total"),
        "val_dropped_rows_total": metrics.get("val_dropped_rows_total"),
        "best_val_loss": metrics.get("best_val_loss"),
        "last_epoch": final_row,
        "model_path": metrics.get("model_path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dual one-shot sequence chess models (legacy winner-split or all-play bootstrap + side fine-tune)"
    )
    parser.add_argument("--train", action="append", required=True, help="Train JSONL path (repeatable)")
    parser.add_argument("--val", action="append", required=True, help="Validation JSONL path (repeatable)")
    parser.add_argument("--side-mode", choices=["white", "black", "both"], default="both")
    parser.add_argument(
        "--architecture",
        choices=[TRAINING_ARCHITECTURE_WINNER_SPLIT, TRAINING_ARCHITECTURE_ALLPLAY_BOOTSTRAP],
        default=TRAINING_ARCHITECTURE_WINNER_SPLIT,
        help="Training architecture: winner-split dual tracks or all-play bootstrap then side fine-tune",
    )
    parser.add_argument(
        "--model-family",
        choices=[MODEL_FAMILY_DUAL_SEQUENCE, MODEL_FAMILY_DUAL_SEQUENCE_BOARD],
        default=MODEL_FAMILY_DUAL_SEQUENCE,
        help="Backbone family: move-only or board-conditioned",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bootstrap-epochs", type=int, default=0, help="All-play bootstrap epochs (0 => max(1, epochs//3))")
    parser.add_argument("--finetune-epochs", type=int, default=0, help="Per-side fine-tune epochs (0 => epochs)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--step-loss-decay", type=float, default=0.9)
    parser.add_argument("--step1-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--curriculum-start-horizon", type=int, default=0)
    parser.add_argument("--curriculum-end-horizon", type=int, default=0)
    parser.add_argument("--curriculum-ramp-epochs", type=int, default=0)
    parser.add_argument("--side-to-move-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--side-to-move-embed-dim", type=int, default=4)
    parser.add_argument("--runtime-min-context", type=int, default=8)
    parser.add_argument("--runtime-min-target", type=int, default=1)
    parser.add_argument("--runtime-max-samples-per-game", type=int, default=0)
    parser.add_argument("--mate-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mate-in-x", type=int, default=3)
    parser.add_argument("--mate-weight", type=float, default=1.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Torch device: auto/cpu/cuda/cuda:N")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-model-shared", default="artifacts/model_shared_bootstrap.pt")
    parser.add_argument("--out-model-white", default="artifacts/model_white.pt")
    parser.add_argument("--out-model-black", default="artifacts/model_black.pt")
    parser.add_argument("--out-metrics", default="artifacts/train_metrics_dual_sequence.json")
    args = parser.parse_args()

    train_paths = [os.fspath(Path(p).resolve()) for p in args.train]
    val_paths = [os.fspath(Path(p).resolve()) for p in args.val]
    device = _resolve_device(args.device)
    print(
        {
            "torch_version": torch.__version__,
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_device": args.device,
            "resolved_device": device,
            "architecture": args.architecture,
            "side_mode": args.side_mode,
            "model_family": args.model_family,
            "horizon": int(args.horizon),
        }
    )

    side_order: List[str] = []
    bootstrap_result: Dict[str, Any] = {}
    results_by_side: Dict[str, Dict[str, Any]] = {}
    output_model_family = str(args.model_family)
    training_mode = "dual_side_sequence"
    if str(args.architecture) == TRAINING_ARCHITECTURE_ALLPLAY_BOOTSTRAP:
        if str(args.side_mode) != "both":
            raise SystemExit("--architecture=allplay_bootstrap_side_finetune_curriculum requires --side-mode both")
        side_order = [MODEL_SIDE_WHITE, MODEL_SIDE_BLACK]
        ensure_parent(args.out_model_shared)
        ensure_parent(args.out_model_white)
        ensure_parent(args.out_model_black)
        bootstrap_epochs = int(args.bootstrap_epochs) if int(args.bootstrap_epochs) > 0 else max(1, int(args.epochs) // 3)
        finetune_epochs = int(args.finetune_epochs) if int(args.finetune_epochs) > 0 else int(args.epochs)
        orchestrated = train_bootstrap_dual_head_curriculum_from_jsonl_paths(
            train_paths=train_paths,
            val_paths=val_paths,
            out_model_shared_path=str(args.out_model_shared),
            out_model_white_path=str(args.out_model_white),
            out_model_black_path=str(args.out_model_black),
            out_metrics_path=None,
            seed=int(args.seed),
            horizon=int(args.horizon),
            bootstrap_epochs=int(bootstrap_epochs),
            finetune_epochs=int(finetune_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            embed_dim=int(args.embed_dim),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            use_side_to_move_feature=bool(args.side_to_move_feature),
            side_to_move_embed_dim=int(args.side_to_move_embed_dim),
            step_loss_decay=float(args.step_loss_decay),
            step1_loss_multiplier=float(args.step1_loss_multiplier),
            runtime_min_context=int(args.runtime_min_context),
            runtime_min_target=int(args.runtime_min_target),
            runtime_max_samples_per_game=int(args.runtime_max_samples_per_game),
            num_workers=int(args.num_workers),
            device_str=device,
            mate_bias_enabled=bool(args.mate_bias),
            mate_in_x=int(args.mate_in_x),
            mate_weight=float(args.mate_weight),
            model_family=str(args.model_family),
            curriculum_start_horizon=int(args.curriculum_start_horizon),
            curriculum_end_horizon=int(args.curriculum_end_horizon),
            curriculum_ramp_epochs=int(args.curriculum_ramp_epochs),
            verbose=bool(args.verbose),
        )
        bootstrap_result = dict(orchestrated.get("bootstrap", {}))
        results_by_side = dict(orchestrated.get("sides", {}))
        merged_metrics = dict(orchestrated.get("metrics", {}))
        output_model_family = str(merged_metrics.get("output_model_family") or output_model_family)
        training_mode = str(merged_metrics.get("training_mode") or "dual_bootstrap_curriculum")
        print({"event": "trained_bootstrap", "stage": "bootstrap_allplay", "model_path": args.out_model_shared})
        for side in side_order:
            print({"event": "trained_side", **_summarize_side_result(results_by_side.get(side, {}))})
    else:
        if args.side_mode == "white":
            side_order = [MODEL_SIDE_WHITE]
        elif args.side_mode == "black":
            side_order = [MODEL_SIDE_BLACK]
        else:
            side_order = [MODEL_SIDE_WHITE, MODEL_SIDE_BLACK]
        for side in side_order:
            out_model = args.out_model_white if side == MODEL_SIDE_WHITE else args.out_model_black
            ensure_parent(out_model)
            result = train_dual_sequence_model_from_jsonl_paths(
                train_paths=train_paths,
                val_paths=val_paths,
                model_side=side,
                out_model_path=out_model,
                out_metrics_path=None,
                seed=int(args.seed),
                horizon=int(args.horizon),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                embed_dim=int(args.embed_dim),
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                dropout=float(args.dropout),
                use_side_to_move_feature=bool(args.side_to_move_feature),
                side_to_move_embed_dim=int(args.side_to_move_embed_dim),
                step_loss_decay=float(args.step_loss_decay),
                step1_loss_multiplier=float(args.step1_loss_multiplier),
                runtime_min_context=int(args.runtime_min_context),
                runtime_min_target=int(args.runtime_min_target),
                runtime_max_samples_per_game=int(args.runtime_max_samples_per_game),
                num_workers=int(args.num_workers),
                device_str=device,
                mate_bias_enabled=bool(args.mate_bias),
                mate_in_x=int(args.mate_in_x),
                mate_weight=float(args.mate_weight),
                model_family=str(args.model_family),
                curriculum_start_horizon=int(args.curriculum_start_horizon),
                curriculum_end_horizon=int(args.curriculum_end_horizon),
                curriculum_ramp_epochs=int(args.curriculum_ramp_epochs),
                stage_name=f"train_{side}",
                training_architecture=str(args.architecture),
                verbose=bool(args.verbose),
            )
            results_by_side[side] = result
            print({"event": "trained_side", **_summarize_side_result(result)})

    merged = {
        "training_mode": str(training_mode),
        "architecture": str(args.architecture),
        "side_mode": args.side_mode,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "runtime": {
            "device_requested": str(args.device),
            "device_resolved": str(device),
            "horizon": int(args.horizon),
            "epochs": int(args.epochs),
            "bootstrap_epochs": int(args.bootstrap_epochs) if int(args.bootstrap_epochs) > 0 else max(1, int(args.epochs) // 3),
            "finetune_epochs": int(args.finetune_epochs) if int(args.finetune_epochs) > 0 else int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "embed_dim": int(args.embed_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "step_loss_decay": float(args.step_loss_decay),
            "step1_loss_multiplier": float(args.step1_loss_multiplier),
            "curriculum_start_horizon": int(args.curriculum_start_horizon),
            "curriculum_end_horizon": int(args.curriculum_end_horizon),
            "curriculum_ramp_epochs": int(args.curriculum_ramp_epochs),
            "side_to_move_feature": bool(args.side_to_move_feature),
            "side_to_move_embed_dim": int(args.side_to_move_embed_dim),
            "model_family_backbone": str(args.model_family),
            "model_family": str(output_model_family),
            "runtime_min_context": int(args.runtime_min_context),
            "runtime_min_target": int(args.runtime_min_target),
            "runtime_max_samples_per_game": int(args.runtime_max_samples_per_game),
            "mate_bias": bool(args.mate_bias),
            "mate_in_x": int(args.mate_in_x),
            "mate_weight": float(args.mate_weight),
            "num_workers": int(args.num_workers),
        },
        "bootstrap": bootstrap_result.get("metrics", {}),
        "sides": {k: v.get("metrics", {}) for k, v in results_by_side.items()},
        "side_summaries": {k: _summarize_side_result(v) for k, v in results_by_side.items()},
    }
    ensure_parent(args.out_metrics)
    write_json(args.out_metrics, merged)
    print(json.dumps({"event": "training_complete", "out_metrics": args.out_metrics, "sides": side_order}))


if __name__ == "__main__":
    main()
