#!/usr/bin/env python3
import argparse
from datetime import timedelta
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow direct script execution without requiring PYTHONPATH=. from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.training import train_next_move_model_from_jsonl_paths


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.upper() in {"N/A", "NA"}:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.upper() in {"N/A", "NA"}:
            return None
        return int(float(s))
    except Exception:
        return None


def _parse_nvidia_smi_csv_line(line: str) -> Dict[str, Any]:
    parts = [p.strip() for p in line.split(",")]
    # index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit
    while len(parts) < 8:
        parts.append("")
    return {
        "gpu_index": _safe_int(parts[0]),
        "gpu_name": parts[1] or None,
        "gpu_util_percent": _safe_float(parts[2]),
        "vram_used_mib": _safe_float(parts[3]),
        "vram_total_mib": _safe_float(parts[4]),
        "gpu_temp_c": _safe_float(parts[5]),
        "gpu_power_w": _safe_float(parts[6]),
        "gpu_power_limit_w": _safe_float(parts[7]),
    }


def _read_proc_status_kv() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    except Exception:
        return out
    return out


def _summarize_telemetry_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = [
        "gpu_util_percent",
        "vram_used_mib",
        "vram_total_mib",
        "gpu_temp_c",
        "gpu_power_w",
        "gpu_power_limit_w",
        "proc_rss_kib",
        "epoch",
        "last_train_loss",
        "last_val_loss",
        "last_top1",
        "last_top5",
        "last_samples_per_sec_epoch",
        "last_samples_per_sec_interval",
        "last_epoch_eta_sec",
    ]
    metrics: Dict[str, Any] = {}
    for key in numeric_keys:
        vals: List[float] = []
        for s in samples:
            v = s.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            continue
        metrics[key] = {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
        }
    first_ts = samples[0].get("ts_ms") if samples else None
    last_ts = samples[-1].get("ts_ms") if samples else None
    return {
        "sample_count": len(samples),
        "first_ts_ms": first_ts,
        "last_ts_ms": last_ts,
        "duration_seconds": ((last_ts - first_ts) / 1000.0) if (isinstance(first_ts, int) and isinstance(last_ts, int)) else None,
        "metrics": metrics,
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)


def _configure_tf32(mode: str) -> Dict[str, Any]:
    requested = str(mode or "auto").strip().lower()
    if requested not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported tf32 mode: {mode}")
    matmul_before = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
    cudnn_before = bool(getattr(torch.backends.cudnn, "allow_tf32", False))
    if requested == "on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif requested == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "requested": requested,
        "matmul_allow_tf32_before": matmul_before,
        "cudnn_allow_tf32_before": cudnn_before,
        "matmul_allow_tf32_after": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32_after": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


def _runtime_precision_context(*, amp_requested: bool, amp_dtype_requested: str, tf32_state: Dict[str, Any]) -> Dict[str, Any]:
    matmul_precision = None
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            matmul_precision = str(torch.get_float32_matmul_precision())
        except Exception:
            matmul_precision = None
    autocast_gpu_dtype_default = None
    try:
        if hasattr(torch, "get_autocast_dtype"):
            autocast_gpu_dtype_default = str(torch.get_autocast_dtype("cuda"))
        elif hasattr(torch, "get_autocast_gpu_dtype"):
            autocast_gpu_dtype_default = str(torch.get_autocast_gpu_dtype())
    except Exception:
        autocast_gpu_dtype_default = None
    amp_dtype_mode = str(amp_dtype_requested or "auto").strip().lower()
    amp_dtype_effective = "none"
    if amp_requested:
        if amp_dtype_mode == "fp16":
            amp_dtype_effective = str(torch.float16)
        elif amp_dtype_mode == "bf16":
            amp_dtype_effective = str(torch.bfloat16)
        elif amp_dtype_mode == "auto":
            amp_dtype_effective = autocast_gpu_dtype_default or "auto"
        else:
            amp_dtype_effective = amp_dtype_mode
    return {
        "torch_float32_matmul_precision": matmul_precision,
        "autocast_gpu_dtype_default": autocast_gpu_dtype_default,
        "amp_requested": bool(amp_requested),
        "amp_dtype_requested": str(amp_dtype_requested),
        "amp_dtype_effective": amp_dtype_effective,
        "tf32": tf32_state,
    }


def _resolve_distributed_context(mode: str) -> Dict[str, int | bool]:
    normalized = str(mode or "auto").strip().lower()
    if normalized not in {"auto", "off", "on"}:
        raise ValueError(f"Unsupported distributed mode: {mode}")
    world_size = max(1, _env_int("WORLD_SIZE", 1))
    rank = max(0, _env_int("RANK", 0))
    local_rank = max(0, _env_int("LOCAL_RANK", 0))
    if normalized == "off":
        enabled = False
    elif normalized == "on":
        enabled = world_size > 1
        if not enabled:
            raise RuntimeError("Distributed mode 'on' requires torchrun-style WORLD_SIZE>1 environment")
    else:
        enabled = world_size > 1
    return {
        "enabled": bool(enabled),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
    }


def _collect_cuda_startup_diagnostics(dist_ctx: Dict[str, int | bool]) -> Dict[str, Any]:
    env_keys = [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    ]
    env_info = {k: os.environ.get(k, "") for k in env_keys}
    diag: Dict[str, Any] = {
        "distributed": {
            "enabled": bool(dist_ctx.get("enabled", False)),
            "world_size": int(dist_ctx.get("world_size", 1)),
            "rank": int(dist_ctx.get("rank", 0)),
            "local_rank": int(dist_ctx.get("local_rank", 0)),
        },
        "torch": {
            "version": str(getattr(torch, "__version__", "")),
            "cuda_version_compiled": str(getattr(torch.version, "cuda", "")),
            "cuda_is_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "env": env_info,
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=8.0,
            ).strip()
            diag["nvidia_smi_l"] = out
        except Exception as exc:
            diag["nvidia_smi_l_error"] = repr(exc)
    else:
        diag["nvidia_smi_l_error"] = "nvidia-smi not found"
    return diag


def _format_cuda_unavailable_distributed_error(diag: Dict[str, Any]) -> str:
    hints = [
        "PyTorch CUDA runtime failed to initialize on this node.",
        "This can happen when node drivers/runtime are mismatched even if `nvidia-smi` shows GPUs.",
        "Try a fresh pod/host, verify image+driver compatibility, then rerun the same torch CUDA self-check.",
    ]
    return (
        "Distributed GPU training requires CUDA, but torch.cuda.is_available() is False.\n"
        f"startup_diagnostics={json.dumps(diag, ensure_ascii=True)}\n"
        "hints:\n- "
        + "\n- ".join(hints)
    )


class _TrainingTelemetryLogger:
    def __init__(
        self,
        enabled: bool,
        run_dir: Optional[Path],
        interval_sec: float,
        requested_device: str,
    ) -> None:
        self.enabled = bool(enabled and run_dir is not None)
        self.run_dir = run_dir
        self.interval_sec = max(0.5, float(interval_sec))
        self.requested_device = str(requested_device)
        self._state_lock = threading.Lock()
        self._latest_state: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples_fh = None
        self._samples: List[Dict[str, Any]] = []
        self._gpu_index_hint = self._infer_gpu_index_hint(self.requested_device)
        self._nvidia_smi_available = shutil.which("nvidia-smi") is not None
        self._meta: Dict[str, Any] = {}

    @staticmethod
    def _infer_gpu_index_hint(device: str) -> Optional[int]:
        dev = str(device or "").strip().lower()
        if dev.startswith("cuda:"):
            try:
                return int(dev.split(":", 1)[1])
            except Exception:
                return 0
        if dev in {"cuda", "auto"}:
            return 0
        return None

    def init_run(self, meta: Dict[str, Any]) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._meta = dict(meta)
        write_json(str(self.run_dir / "run_meta.json"), self._meta)
        self._samples_fh = (self.run_dir / "samples.jsonl").open("a", encoding="utf-8")

    def update_meta(self, extra: Dict[str, Any]) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._meta.update(extra)
        write_json(str(self.run_dir / "run_meta.json"), self._meta)

    def update_from_progress_event(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        evt = dict(event or {})
        state_updates: Dict[str, Any] = {}
        event_name = str(evt.get("event", ""))
        state_updates["last_event"] = event_name
        if "epoch" in evt:
            state_updates["epoch"] = _safe_int(evt.get("epoch"))
        if "epochs" in evt:
            state_updates["epochs"] = _safe_int(evt.get("epochs"))
        if event_name == "train_setup":
            for key in [
                "train_rows",
                "val_rows",
                "device_selected",
                "amp_enabled",
                "batch_size",
                "data_loading",
                "dataset_schema",
                "training_objective",
                "rollout_horizon",
                "closeness_horizon",
            ]:
                if key in evt:
                    state_updates[key] = evt.get(key)
        if event_name == "epoch_end":
            metrics = evt.get("metrics") or {}
            if isinstance(metrics, dict):
                state_updates["last_train_loss"] = _safe_float(metrics.get("train_loss"))
                state_updates["last_val_loss"] = _safe_float(metrics.get("val_loss"))
                state_updates["last_top1"] = _safe_float(metrics.get("top1"))
                state_updates["last_top5"] = _safe_float(metrics.get("top5"))
                state_updates["last_lr"] = _safe_float(metrics.get("lr"))
        if event_name == "batch_progress":
            state_updates["batch_idx"] = _safe_int(evt.get("batch_idx"))
            state_updates["train_batches_total"] = _safe_int(evt.get("train_batches_total"))
            state_updates["samples_seen_epoch"] = _safe_int(evt.get("samples_seen_epoch"))
            state_updates["last_samples_per_sec_epoch"] = _safe_float(evt.get("samples_per_sec_epoch"))
            state_updates["last_samples_per_sec_interval"] = _safe_float(evt.get("samples_per_sec_interval"))
            state_updates["last_epoch_eta_sec"] = _safe_float(evt.get("epoch_eta_sec"))
            state_updates["last_train_loss"] = _safe_float(evt.get("running_train_loss"))
            state_updates["last_train_l1_penalty"] = _safe_float(evt.get("running_train_l1_penalty"))
        with self._state_lock:
            self._latest_state.update({k: v for k, v in state_updates.items() if v is not None or isinstance(v, str)})

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="train-telemetry-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_sec * 2))
        if self._samples_fh is not None:
            self._samples_fh.close()
            self._samples_fh = None

    def write_final_summary(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self.run_dir is None:
            return
        summary = _summarize_telemetry_samples(self._samples)
        if extra:
            summary["training_summary"] = extra
        write_json(str(self.run_dir / "summary.json"), summary)

    def _query_gpu_row(self) -> Dict[str, Any]:
        if not self._nvidia_smi_available or self._gpu_index_hint is None:
            return {}
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception:
            return {}
        if proc.returncode != 0:
            return {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_nvidia_smi_csv_line(line)
            if parsed.get("gpu_index") == self._gpu_index_hint:
                return parsed
        return {}

    def _sample(self) -> Dict[str, Any]:
        ts_ms = int(time.time() * 1000)
        row: Dict[str, Any] = {"ts_ms": ts_ms, "pid": int(os.getpid())}
        proc_status = _read_proc_status_kv()
        vmrss = proc_status.get("VmRSS", "")
        if vmrss:
            parts = vmrss.split()
            if parts:
                row["proc_rss_kib"] = _safe_int(parts[0])
        gpu = self._query_gpu_row()
        row.update(gpu)
        with self._state_lock:
            row.update(self._latest_state)
        return row

    def _run(self) -> None:
        while not self._stop.is_set():
            row = self._sample()
            self._samples.append(row)
            if self._samples_fh is not None:
                self._samples_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
                self._samples_fh.flush()
            self._stop.wait(self.interval_sec)


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
    parser.add_argument("--rollout-horizon", type=int, default=1, help="Predict N future plies with teacher-forced recursive loss (1 preserves baseline)")
    parser.add_argument("--closeness-horizon", type=int, default=4, help="Evaluate continuation closeness on first N rollout plies (clamped to rollout horizon)")
    parser.add_argument("--rollout-loss-decay", type=float, default=0.7, help="Decay factor for multistep rollout loss weights")
    parser.add_argument("--runtime-min-context", type=int, default=8, help="Runtime splicing min context plies for game-level datasets")
    parser.add_argument("--runtime-min-target", type=int, default=1, help="Runtime splicing min target plies for game-level datasets")
    parser.add_argument("--runtime-max-samples-per-game", type=int, default=0, help="Runtime splicing sample cap per game (0=no cap)")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional cap on effective training rows after indexing/cache load")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Optional cap on effective validation rows after indexing/cache load")
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=0,
        help="Optional cap on effective train+val rows (auto-split by source ratio when per-split caps unset)",
    )
    parser.add_argument(
        "--require-runtime-splice-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail for game datasets when runtime splice cache cannot be used (no runtime index fallback)",
    )
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
    parser.add_argument(
        "--distributed",
        choices=["auto", "off", "on"],
        default="auto",
        help="Distributed mode: auto (enable under torchrun), off, or on (requires WORLD_SIZE>1).",
    )
    parser.add_argument(
        "--distributed-backend",
        default="nccl",
        help="torch.distributed backend when distributed mode is enabled",
    )
    parser.add_argument(
        "--distributed-timeout-sec",
        type=int,
        default=1800,
        help="Process-group init timeout in seconds when distributed mode is enabled",
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
        "--amp-dtype",
        choices=["auto", "fp16", "bf16"],
        default="auto",
        help="CUDA autocast dtype when AMP is enabled (auto keeps PyTorch default)",
    )
    parser.add_argument(
        "--tf32",
        choices=["auto", "on", "off"],
        default="auto",
        help="Toggle CUDA TF32 matmul/cuDNN math mode (auto keeps runtime defaults)",
    )
    parser.add_argument(
        "--sparsity-mode",
        choices=["off", "l1"],
        default="off",
        help="Optional sparsity regularization mode (l1 adds L1 penalty during training).",
    )
    parser.add_argument(
        "--sparsity-l1-lambda",
        type=float,
        default=0.0,
        help="L1 penalty multiplier when --sparsity-mode=l1 (0 disables).",
    )
    parser.add_argument(
        "--sparsity-include-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include bias terms in sparsity/L1 tracking.",
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
    parser.add_argument("--best-checkpoint-out", default="", help="Optional path to persist current best checkpoint at each improvement")
    parser.add_argument("--epoch-checkpoint-dir", default="", help="Optional directory to write epoch-end checkpoints")
    parser.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample local GPU/process/training telemetry to local_logs/training_telemetry/ during training",
    )
    parser.add_argument(
        "--telemetry-interval-sec",
        type=float,
        default=2.0,
        help="Telemetry sampling interval in seconds",
    )
    parser.add_argument(
        "--batch-progress-interval-sec",
        type=float,
        default=15.0,
        help="Emit batch-level progress events every N seconds while inside an epoch (0 disables periodic batch events).",
    )
    parser.add_argument(
        "--telemetry-dir",
        default="local_logs/training_telemetry",
        help="Base directory for per-run telemetry logs (gitignored local folder)",
    )
    args = parser.parse_args()
    tf32_state = _configure_tf32(args.tf32)
    precision_runtime = _runtime_precision_context(
        amp_requested=bool(args.amp),
        amp_dtype_requested=str(args.amp_dtype),
        tf32_state=tf32_state,
    )
    dist_ctx = _resolve_distributed_context(args.distributed)
    distributed_enabled = bool(dist_ctx["enabled"])
    distributed_rank = int(dist_ctx["rank"])
    distributed_world_size = int(dist_ctx["world_size"])
    distributed_local_rank = int(dist_ctx["local_rank"])
    is_primary_rank = (not distributed_enabled) or distributed_rank == 0
    distributed_initialized = False
    device_request = str(args.device)
    if distributed_enabled:
        if not torch.cuda.is_available():
            diag = _collect_cuda_startup_diagnostics(dist_ctx)
            raise RuntimeError(_format_cuda_unavailable_distributed_error(diag))
        torch.cuda.set_device(distributed_local_rank)
        if device_request in {"auto", "cuda"}:
            device_request = f"cuda:{distributed_local_rank}"
        if not dist.is_initialized():
            timeout = timedelta(seconds=max(1, int(args.distributed_timeout_sec)))
            dist.init_process_group(
                backend=str(args.distributed_backend),
                init_method="env://",
                timeout=timeout,
            )
            distributed_initialized = True
    train_paths = [Path(p).resolve() for p in args.train]
    val_paths = [Path(p).resolve() for p in args.val]
    train_path = train_paths[0]
    val_path = val_paths[0]
    output_path = Path(args.output).resolve()
    metrics_path = Path(args.metrics_out).resolve()
    progress_jsonl_path = Path(args.progress_jsonl_out).resolve() if args.progress_jsonl_out else None
    telemetry_run_dir = (
        Path(args.telemetry_dir).resolve()
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        / f"{output_path.stem}"
    )

    progress_fh = None
    telemetry = _TrainingTelemetryLogger(
        enabled=bool(args.telemetry and is_primary_rank),
        run_dir=telemetry_run_dir,
        interval_sec=float(args.telemetry_interval_sec),
        requested_device=str(device_request),
    )

    telemetry.init_run(
        {
            "schema_version": 1,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cwd": str(Path.cwd()),
            "pid": int(os.getpid()),
            "command": " ".join(sys.argv),
            "output_path": str(output_path),
            "metrics_out": str(metrics_path),
            "train_paths": [str(p) for p in train_paths],
            "val_paths": [str(p) for p in val_paths],
            "requested_device": str(device_request),
            "distributed": {
                "enabled": bool(distributed_enabled),
                "rank": int(distributed_rank),
                "world_size": int(distributed_world_size),
                "local_rank": int(distributed_local_rank),
                "backend": str(args.distributed_backend),
            },
            "telemetry_interval_sec": float(args.telemetry_interval_sec),
            "batch_progress_interval_sec": float(args.batch_progress_interval_sec),
            "model_request": {
                "embed_dim": int(args.embed_dim),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "phase_feature": bool(args.phase_feature),
                "phase_embed_dim": int(args.phase_embed_dim),
                "side_to_move_feature": bool(args.side_to_move_feature),
                "side_to_move_embed_dim": int(args.side_to_move_embed_dim),
                "rollout_horizon": int(args.rollout_horizon),
                "closeness_horizon": int(args.closeness_horizon),
                "rollout_loss_decay": float(args.rollout_loss_decay),
            },
            "train_request": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "num_workers": int(args.num_workers),
                "pin_memory": bool(args.pin_memory),
                "amp": bool(args.amp),
                "runtime_min_context": int(args.runtime_min_context),
                "runtime_min_target": int(args.runtime_min_target),
                "runtime_max_samples_per_game": int(args.runtime_max_samples_per_game),
                "require_runtime_splice_cache": bool(args.require_runtime_splice_cache),
                "sparsity_mode": str(args.sparsity_mode),
                "sparsity_l1_lambda": float(args.sparsity_l1_lambda),
                "sparsity_include_bias": bool(args.sparsity_include_bias),
            },
            "precision_runtime": precision_runtime,
        }
    )
    telemetry.start()

    def emit_progress(event: dict) -> None:
        nonlocal progress_fh
        if not is_primary_rank:
            return
        telemetry.update_from_progress_event(event)
        if progress_jsonl_path is None:
            return
        if progress_fh is None:
            progress_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            progress_fh = progress_jsonl_path.open("a", encoding="utf-8")
        row = {"ts_epoch_ms": int(time.time() * 1000), **event}
        progress_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        progress_fh.flush()

    if args.verbose and is_primary_rank:
        print(
            {
                "torch_version": torch.__version__,
                "cuda_is_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "requested_device": device_request,
                "distributed": {
                    "enabled": bool(distributed_enabled),
                    "rank": int(distributed_rank),
                    "world_size": int(distributed_world_size),
                    "local_rank": int(distributed_local_rank),
                    "backend": str(args.distributed_backend),
                },
                "precision_runtime": precision_runtime,
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
            "device_requested": device_request,
                    "num_workers": args.num_workers,
                    "pin_memory_requested": args.pin_memory,
                    "amp_requested": args.amp,
                    "amp_dtype": str(args.amp_dtype),
                    "tf32": tf32_state,
                    "precision_runtime": precision_runtime,
                    "rollout_horizon": args.rollout_horizon,
                    "closeness_horizon": args.closeness_horizon,
                    "rollout_loss_decay": args.rollout_loss_decay,
                    "runtime_min_context": args.runtime_min_context,
                    "runtime_min_target": args.runtime_min_target,
                    "runtime_max_samples_per_game": args.runtime_max_samples_per_game,
                    "require_runtime_splice_cache": args.require_runtime_splice_cache,
                    "sparsity_mode": args.sparsity_mode,
                    "sparsity_l1_lambda": args.sparsity_l1_lambda,
                    "sparsity_include_bias": args.sparsity_include_bias,
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
            "device_requested": str(device_request),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "batch_progress_interval_sec": float(args.batch_progress_interval_sec),
            "amp_requested": bool(args.amp),
            "amp_dtype": str(args.amp_dtype),
            "tf32": tf32_state,
            "precision_runtime": precision_runtime,
            "rollout_horizon": int(args.rollout_horizon),
            "closeness_horizon": int(args.closeness_horizon),
            "rollout_loss_decay": float(args.rollout_loss_decay),
            "runtime_min_context": int(args.runtime_min_context),
            "runtime_min_target": int(args.runtime_min_target),
            "runtime_max_samples_per_game": int(args.runtime_max_samples_per_game),
            "require_runtime_splice_cache": bool(args.require_runtime_splice_cache),
            "distributed": {
                "enabled": bool(distributed_enabled),
                "rank": int(distributed_rank),
                "world_size": int(distributed_world_size),
                "local_rank": int(distributed_local_rank),
            },
        }
    )

    artifact = None
    history = []
    dataset_info = {}
    train_started_monotonic = time.monotonic()
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
            device_str=device_request,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
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
            batch_progress_interval_sec=args.batch_progress_interval_sec,
            rollout_horizon=args.rollout_horizon,
            closeness_horizon=args.closeness_horizon,
            rollout_loss_decay=args.rollout_loss_decay,
            runtime_min_context=args.runtime_min_context,
            runtime_min_target=args.runtime_min_target,
            runtime_max_samples_per_game=args.runtime_max_samples_per_game,
            require_runtime_splice_cache=args.require_runtime_splice_cache,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            max_total_rows=args.max_total_rows,
            best_checkpoint_out=args.best_checkpoint_out,
            epoch_checkpoint_dir=args.epoch_checkpoint_dir,
            distributed_enabled=distributed_enabled,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            sparsity_mode=args.sparsity_mode,
            sparsity_l1_lambda=args.sparsity_l1_lambda,
            sparsity_include_bias=args.sparsity_include_bias,
        )
    except Exception as exc:
        emit_progress({"event": "script_error", "error_type": type(exc).__name__, "message": str(exc)})
        telemetry.update_meta(
            {
                "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        telemetry.stop()
        telemetry.write_final_summary(
            {
                "status": "error",
                "runtime_seconds": float(time.monotonic() - train_started_monotonic),
            }
        )
        if distributed_initialized and dist.is_initialized():
            dist.destroy_process_group()
        raise

    train_rows_by_file = dataset_info["train_rows_by_file"]
    val_rows_by_file = dataset_info["val_rows_by_file"]
    if args.verbose and is_primary_rank:
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
                    "dataset_schema": dataset_info.get("dataset_schema"),
                    "train_games": dataset_info.get("train_games"),
                    "val_games": dataset_info.get("val_games"),
                }
            }
        )

    if args.verbose and is_primary_rank:
        print({"training_complete": {"epochs_ran": len(history), "last_epoch": (history[-1]["epoch"] if history else None)}})

    if not is_primary_rank:
        telemetry.stop()
        telemetry.write_final_summary(
            {
                "status": "ok",
                "runtime_seconds": float(time.monotonic() - train_started_monotonic),
                "epochs_completed": int(len(history)),
                "dataset_schema": dataset_info.get("dataset_schema"),
                "data_loading": dataset_info.get("data_loading"),
                "distributed_rank": int(distributed_rank),
                "distributed_world_size": int(distributed_world_size),
            }
        )
        if distributed_initialized and dist.is_initialized():
            dist.destroy_process_group()
        return

    ensure_parent(args.output)
    torch.save(artifact, args.output)
    param_count = sum(int(v.numel()) for v in artifact.get("state_dict", {}).values())
    summary = {
        "train_rows": dataset_info["train_rows"],
        "val_rows": dataset_info["val_rows"],
        "train_inputs": [str(p) for p in train_paths],
        "val_inputs": [str(p) for p in val_paths],
        "train_rows_by_file": train_rows_by_file,
        "val_rows_by_file": val_rows_by_file,
        "data_loading": dataset_info.get("data_loading"),
        "dataset_schema": dataset_info.get("dataset_schema"),
        "train_games": dataset_info.get("train_games"),
        "val_games": dataset_info.get("val_games"),
        "train_games_by_file": dataset_info.get("train_games_by_file"),
        "val_games_by_file": dataset_info.get("val_games_by_file"),
        "runtime_splice": dataset_info.get("runtime_splice"),
        "runtime_splice_index_bytes_train": dataset_info.get("runtime_splice_index_bytes_train"),
        "runtime_splice_index_bytes_val": dataset_info.get("runtime_splice_index_bytes_val"),
        "train_rows_source": dataset_info.get("train_rows_source"),
        "val_rows_source": dataset_info.get("val_rows_source"),
        "subset_sampling": dataset_info.get("subset_sampling"),
        "epochs": args.epochs,
        "history": history,
        "model_path": args.output,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "phase_feature": args.phase_feature,
        "phase_embed_dim": args.phase_embed_dim,
        "side_to_move_feature": args.side_to_move_feature,
        "side_to_move_embed_dim": args.side_to_move_embed_dim,
        "device_requested": device_request,
        "distributed": {
            "enabled": bool(distributed_enabled),
            "rank": int(distributed_rank),
            "world_size": int(distributed_world_size),
            "local_rank": int(distributed_local_rank),
            "backend": str(args.distributed_backend),
        },
        "phase_weights": {
            "unknown": args.phase_weight_unknown,
            "opening": args.phase_weight_opening,
            "middlegame": args.phase_weight_middlegame,
            "endgame": args.phase_weight_endgame,
        },
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "tf32": tf32_state,
        "precision_runtime": precision_runtime,
        "sparsity_mode": args.sparsity_mode,
        "sparsity_l1_lambda": args.sparsity_l1_lambda,
        "sparsity_include_bias": args.sparsity_include_bias,
        "sparsity_runtime": artifact.get("runtime", {}).get("sparsity"),
        "rollout_horizon": args.rollout_horizon,
        "closeness_horizon": args.closeness_horizon,
        "rollout_loss_decay": args.rollout_loss_decay,
        "runtime_min_context": args.runtime_min_context,
        "runtime_min_target": args.runtime_min_target,
        "runtime_max_samples_per_game": args.runtime_max_samples_per_game,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_total_rows": args.max_total_rows,
        "require_runtime_splice_cache": args.require_runtime_splice_cache,
        "best_checkpoint_out": args.best_checkpoint_out,
        "epoch_checkpoint_dir": args.epoch_checkpoint_dir,
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
        "training_objective": artifact.get("runtime", {}).get("training_objective", "single_step_next_move"),
        "rollout_loss_weights": artifact.get("runtime", {}).get("rollout_loss_weights"),
        "param_count": param_count,
    }
    write_json(args.metrics_out, summary)
    telemetry.update_meta(
        {
            "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "ok",
            "dataset_loaded": {
                "data_loading": dataset_info.get("data_loading"),
                "dataset_schema": dataset_info.get("dataset_schema"),
                "train_rows": dataset_info.get("train_rows"),
                "val_rows": dataset_info.get("val_rows"),
                "train_games": dataset_info.get("train_games"),
                "val_games": dataset_info.get("val_games"),
                "runtime_splice": dataset_info.get("runtime_splice"),
                "runtime_splice_index_bytes_train": dataset_info.get("runtime_splice_index_bytes_train"),
                "runtime_splice_index_bytes_val": dataset_info.get("runtime_splice_index_bytes_val"),
                "sparsity": dataset_info.get("sparsity"),
            },
            "precision_runtime": precision_runtime,
            "model_final": {
                "param_count": int(param_count),
                "artifact_training_objective": artifact.get("runtime", {}).get("training_objective"),
                "config": artifact.get("config"),
            },
        }
    )
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
    if args.verbose and is_primary_rank:
        print({"best_checkpoint": summary.get("best_checkpoint")})
    print(f"Saved model: {args.output}")
    print(f"Saved metrics: {args.metrics_out}")
    if args.telemetry and is_primary_rank:
        print(f"Saved telemetry logs: {telemetry_run_dir}")
    telemetry.stop()
    telemetry.write_final_summary(
        {
            "status": "ok",
            "runtime_seconds": float(time.monotonic() - train_started_monotonic),
            "epochs_completed": int(len(history)),
            "best_checkpoint": summary.get("best_checkpoint"),
            "param_count": int(param_count),
            "dataset_schema": dataset_info.get("dataset_schema"),
            "data_loading": dataset_info.get("data_loading"),
        }
    )
    if progress_fh is not None:
        progress_fh.close()
    if distributed_initialized and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
