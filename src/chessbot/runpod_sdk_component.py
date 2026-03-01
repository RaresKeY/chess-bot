#!/usr/bin/env python3
"""RunPod SDK-backed component kept separate from raw REST/GraphQL helpers.

This module intentionally does not replace the existing raw-call path.
It provides a side-by-side implementation so teams can choose either path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.chessbot.secrets import default_dotenv_paths, resolve_secret


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEYRING_SERVICE = "runpod"
DEFAULT_KEYRING_USERNAME = "RUNPOD_API_KEY"


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=True))


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _resolve_api_key(args: argparse.Namespace) -> str:
    dotenv_paths = default_dotenv_paths(
        repo_root=REPO_ROOT,
        override_var_names=("RUNPOD_SDK_DOTENV_PATH", "RUNPOD_DOTENV_PATH", "CHESSBOT_DOTENV_PATH"),
        fallback_filenames=(".env.runpod", ".env"),
    )
    value, _ = resolve_secret(
        explicit_value=str(getattr(args, "api_key", "") or ""),
        env_var_names=("RUNPOD_API_KEY",),
        keyring_service=str(args.keyring_service),
        keyring_username=str(args.keyring_username),
        dotenv_keys=("RUNPOD_API_KEY",),
        dotenv_paths=dotenv_paths,
        order=("explicit", "env", "keyring", "dotenv"),
    )
    return value


def _resolve_attr(root: Any, path: str) -> Optional[Any]:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _first_callable(root: Any, names: Sequence[str]) -> Optional[Callable[..., Any]]:
    for name in names:
        fn = _resolve_attr(root, name)
        if callable(fn):
            return fn
    return None


def _load_runpod_sdk() -> Any:
    try:
        import runpod  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "RunPod SDK component requires the `runpod` package. "
            "Install it in the active environment, e.g. `.venv/bin/python -m pip install runpod`."
        ) from exc
    return runpod


def _set_sdk_api_key(runpod_mod: Any, api_key: str) -> None:
    os.environ["RUNPOD_API_KEY"] = api_key
    # Cover common SDK layouts.
    if hasattr(runpod_mod, "api_key"):
        try:
            setattr(runpod_mod, "api_key", api_key)
        except Exception:
            pass
    config = _resolve_attr(runpod_mod, "config")
    if config is not None and hasattr(config, "api_key"):
        try:
            setattr(config, "api_key", api_key)
        except Exception:
            pass


def _invoke_with_payload(fn: Callable[..., Any], payload: Dict[str, Any]) -> Any:
    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
        ((), payload),
        ((payload,), {}),
        ((), {"payload": payload}),
        ((), {"body": payload}),
        ((), {"input": payload}),
        ((), {"data": payload}),
    ]
    last_err: Optional[Exception] = None
    for args, kwargs in attempts:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("unreachable")


def _invoke_with_id(fn: Callable[..., Any], pod_id: str) -> Any:
    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
        ((pod_id,), {}),
        ((), {"pod_id": pod_id}),
        ((), {"id": pod_id}),
        ((), {"podId": pod_id}),
        ((), {"pod": pod_id}),
    ]
    last_err: Optional[Exception] = None
    for args, kwargs in attempts:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("unreachable")


def _normalize_rows(payload: Any, common_list_keys: Iterable[str]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in common_list_keys:
            rows = payload.get(key)
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
        data = payload.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rank_gpu_rows(rows: List[Dict[str, Any]], *, cloud_type: str, min_memory_gb: int, max_hourly_price: float) -> List[Dict[str, Any]]:
    want_secure = str(cloud_type).upper() == "SECURE"
    out: List[Dict[str, Any]] = []
    for row in rows:
        mem = int(row.get("memoryInGb") or row.get("memory_gb") or row.get("memory") or 0)
        if mem < min_memory_gb:
            continue
        price = _as_float(
            row.get("securePrice" if want_secure else "communityPrice")
            or row.get("price_per_hr")
            or row.get("price")
            or 0.0,
            default=0.0,
        )
        if max_hourly_price > 0 and price > max_hourly_price:
            continue
        max_count = int(row.get("maxGpuCount") or row.get("max_gpu_count") or 0)
        out.append(
            {
                "id": str(row.get("id", "")),
                "display_name": row.get("displayName") or row.get("display_name") or "",
                "memory_gb": mem,
                "cloud_type": "SECURE" if want_secure else "COMMUNITY",
                "max_gpu_count": max_count,
                "price_per_hr": price,
                "raw": row,
                "_sort": (price if price > 0 else 1e9, -mem, -max_count),
            }
        )
    out.sort(key=lambda x: x["_sort"])
    for row in out:
        row.pop("_sort", None)
    return out


def _choose_template(templates: List[Dict[str, Any]], template_id: str = "", template_name: str = "") -> Dict[str, Any]:
    if template_id:
        for t in templates:
            if str(t.get("id", "")) == template_id:
                return t
        raise SystemExit(f"Template id not found: {template_id}")
    if template_name:
        exact = [t for t in templates if str(t.get("name", "")) == template_name]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise SystemExit(f"Multiple templates matched name exactly: {template_name}")
        partial = [t for t in templates if template_name.lower() in str(t.get("name", "")).lower()]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise SystemExit(f"No template matched name: {template_name}")
        raise SystemExit(f"Multiple templates matched name substring: {template_name}")
    raise SystemExit("Specify --template-id or --template-name")


def _sdk_gpu_types(runpod_mod: Any) -> List[Dict[str, Any]]:
    fn = _first_callable(
        runpod_mod,
        (
            "get_gpu_types",
            "gpu_types",
            "list_gpu_types",
            "get_gpus",
            "api.get_gpu_types",
            "api.gpu_types",
            "api.get_gpus",
            "pods.get_gpu_types",
        ),
    )
    if fn is None:
        raise SystemExit("RunPod SDK component cannot find a GPU listing method in the installed SDK.")
    payload = fn()
    return _normalize_rows(payload, ("gpus", "gpuTypes", "items", "results"))


def _sdk_templates(runpod_mod: Any, *, include_runpod_templates: bool, include_public_templates: bool, include_serverless: bool) -> List[Dict[str, Any]]:
    fn = _first_callable(
        runpod_mod,
        (
            "get_templates",
            "templates",
            "list_templates",
            "api.get_templates",
            "api.templates",
        ),
    )
    if fn is None:
        raise SystemExit("RunPod SDK component cannot find a template listing method in the installed SDK.")
    try:
        payload = fn(
            include_runpod_templates=include_runpod_templates,
            include_public_templates=include_public_templates,
            include_serverless=include_serverless,
        )
    except TypeError:
        payload = fn()
    return _normalize_rows(payload, ("templates", "items", "results", "data"))


def _sdk_create_pod(runpod_mod: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _first_callable(
        runpod_mod,
        (
            "create_pod",
            "pods.create",
            "pod.create",
            "api.create_pod",
        ),
    )
    if fn is None:
        raise SystemExit("RunPod SDK component cannot find a pod creation method in the installed SDK.")
    out = _invoke_with_payload(fn, payload)
    return out if isinstance(out, dict) else {"response": out}


def _sdk_get_pod(runpod_mod: Any, pod_id: str) -> Dict[str, Any]:
    fn = _first_callable(
        runpod_mod,
        (
            "get_pod",
            "pod",
            "pods.get",
            "api.get_pod",
        ),
    )
    if fn is None:
        raise SystemExit("RunPod SDK component cannot find a pod status method in the installed SDK.")
    out = _invoke_with_id(fn, pod_id)
    return out if isinstance(out, dict) else {"response": out}


def _sdk_stop_pod(runpod_mod: Any, pod_id: str) -> Dict[str, Any]:
    fn = _first_callable(
        runpod_mod,
        (
            "stop_pod",
            "pod_stop",
            "pods.stop",
            "api.stop_pod",
        ),
    )
    if fn is None:
        raise SystemExit("RunPod SDK component cannot find a pod stop method in the installed SDK.")
    out = _invoke_with_id(fn, pod_id)
    return out if isinstance(out, dict) else {"response": out}


def _sdk_terminate_pod(runpod_mod: Any, pod_id: str) -> Dict[str, Any]:
    fn = _first_callable(
        runpod_mod,
        (
            "terminate_pod",
            "delete_pod",
            "pods.terminate",
            "pods.delete",
            "api.terminate_pod",
        ),
    )
    if fn is None:
        raise SystemExit("RunPod SDK component cannot find a pod terminate/delete method in the installed SDK.")
    out = _invoke_with_id(fn, pod_id)
    return out if isinstance(out, dict) else {"response": out}


def _extract_pod_state(pod: Dict[str, Any]) -> str:
    for key in ("desiredStatus", "desired_status", "status", "machineStatus", "state"):
        value = pod.get(key)
        if isinstance(value, str) and value:
            return value
    nested = pod.get("pod")
    if isinstance(nested, dict):
        return _extract_pod_state(nested)
    return ""


def cmd_gpu_search(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    runpod_mod = _load_runpod_sdk()
    _set_sdk_api_key(runpod_mod, api_key)
    rows = _sdk_gpu_types(runpod_mod)
    ranked = _rank_gpu_rows(
        rows,
        cloud_type=args.cloud_type,
        min_memory_gb=int(args.min_memory_gb),
        max_hourly_price=float(args.max_hourly_price),
    )
    if args.limit > 0:
        ranked = ranked[: args.limit]
    _print_json({"gpus": ranked, "count": len(ranked), "cloud_type": args.cloud_type, "component": "runpod_sdk"})
    return 0


def cmd_template_list(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    runpod_mod = _load_runpod_sdk()
    _set_sdk_api_key(runpod_mod, api_key)
    rows = _sdk_templates(
        runpod_mod,
        include_runpod_templates=args.include_runpod_templates,
        include_public_templates=args.include_public_templates,
        include_serverless=args.include_serverless,
    )
    if args.template_name:
        rows = [r for r in rows if args.template_name.lower() in str(r.get("name", "")).lower()]
    if args.pods_only:
        rows = [r for r in rows if str(r.get("templateType", "pod")).lower() != "serverless"]
    _print_json({"templates": rows, "count": len(rows), "component": "runpod_sdk"})
    return 0


def cmd_provision(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    runpod_mod = _load_runpod_sdk()
    _set_sdk_api_key(runpod_mod, api_key)

    templates = _sdk_templates(
        runpod_mod,
        include_runpod_templates=args.include_runpod_templates,
        include_public_templates=args.include_public_templates,
        include_serverless=False,
    )
    template = _choose_template(templates, template_id=args.template_id, template_name=args.template_name)

    gpu_ids: List[str] = []
    if args.gpu_type_id:
        gpu_ids = [args.gpu_type_id]
    else:
        ranked = _rank_gpu_rows(
            _sdk_gpu_types(runpod_mod),
            cloud_type=args.cloud_type,
            min_memory_gb=int(args.min_memory_gb),
            max_hourly_price=float(args.max_hourly_price),
        )
        if not ranked:
            raise SystemExit("No GPU types matched requested constraints.")
        gpu_ids = [str(ranked[0].get("id", ""))]

    env_map: Dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"Invalid --env item (expected KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"Invalid --env item (empty key): {item}")
        env_map[k] = v

    payload: Dict[str, Any] = {
        "name": args.name,
        "imageName": template.get("imageName") or template.get("image") or "",
        "gpuTypeIds": gpu_ids,
        "cloudType": args.cloud_type,
        "gpuCount": int(args.gpu_count),
        "volumeInGb": int(args.volume_in_gb),
        "containerDiskInGb": int(args.container_disk_in_gb),
        "volumeMountPath": args.volume_mount_path,
        "ports": args.ports,
        "env": env_map,
        "interruptible": bool(args.interruptible),
        "supportPublicIp": bool(args.support_public_ip),
        "templateId": template.get("id", ""),
    }
    create_out = _sdk_create_pod(runpod_mod, payload)

    pod_id = str(create_out.get("id") or create_out.get("podId") or create_out.get("pod_id") or "")
    if args.wait_ready and pod_id:
        deadline = time.time() + int(args.wait_timeout_seconds)
        last_status = ""
        while True:
            pod = _sdk_get_pod(runpod_mod, pod_id)
            last_status = _extract_pod_state(pod)
            if last_status.upper() in {"RUNNING", "READY"}:
                create_out["pod_status"] = pod
                break
            if time.time() >= deadline:
                create_out["wait_timeout"] = {
                    "seconds": int(args.wait_timeout_seconds),
                    "last_status": last_status,
                }
                break
            time.sleep(max(1, int(args.wait_poll_seconds)))

    _print_json(
        {
            "component": "runpod_sdk",
            "pod_id": pod_id,
            "pod_status": create_out.get("pod_status", {}),
            "create_response": create_out,
            "selected_gpu_type_id": gpu_ids[0],
            "selected_template": {"id": template.get("id"), "name": template.get("name")},
            "request": {
                "name": args.name,
                "cloud_type": args.cloud_type,
                "gpu_count": int(args.gpu_count),
                "interruptible": bool(args.interruptible),
            },
        }
    )
    return 0


def cmd_pod_status(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    runpod_mod = _load_runpod_sdk()
    _set_sdk_api_key(runpod_mod, api_key)
    pod = _sdk_get_pod(runpod_mod, args.pod_id)
    _print_json({"component": "runpod_sdk", "pod_id": args.pod_id, "pod_status": pod})
    return 0


def cmd_pod_stop(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    runpod_mod = _load_runpod_sdk()
    _set_sdk_api_key(runpod_mod, api_key)
    out = _sdk_stop_pod(runpod_mod, args.pod_id)
    _print_json({"component": "runpod_sdk", "pod_id": args.pod_id, "stop_response": out})
    return 0


def cmd_pod_terminate(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    runpod_mod = _load_runpod_sdk()
    _set_sdk_api_key(runpod_mod, api_key)
    out = _sdk_terminate_pod(runpod_mod, args.pod_id)
    _print_json({"component": "runpod_sdk", "pod_id": args.pod_id, "terminate_response": out})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "RunPod SDK component (modular, side-by-side with raw API scripts). "
            "This command family intentionally does not replace scripts/runpod_provision.py."
        )
    )
    p.add_argument("--api-key", default="", help="RunPod API key (highest precedence)")
    p.add_argument("--keyring-service", default=DEFAULT_KEYRING_SERVICE)
    p.add_argument("--keyring-username", default=DEFAULT_KEYRING_USERNAME)

    sub = p.add_subparsers(dest="cmd", required=True)

    p_gpu = sub.add_parser("gpu-search", help="List/rank GPU types through RunPod SDK")
    p_gpu.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    p_gpu.add_argument("--min-memory-gb", type=int, default=24)
    p_gpu.add_argument("--max-hourly-price", type=float, default=0.0)
    p_gpu.add_argument("--limit", type=int, default=20)
    p_gpu.set_defaults(func=cmd_gpu_search)

    p_tpl = sub.add_parser("template-list", help="List templates through RunPod SDK")
    _bool_arg(p_tpl, "include-runpod-templates", True, "Include RunPod templates")
    _bool_arg(p_tpl, "include-public-templates", True, "Include public templates")
    _bool_arg(p_tpl, "include-serverless", False, "Include serverless templates")
    _bool_arg(p_tpl, "pods-only", True, "Filter out serverless templates")
    p_tpl.add_argument("--template-name", default="", help="Optional case-insensitive name filter")
    p_tpl.set_defaults(func=cmd_template_list)

    p_prov = sub.add_parser("provision", help="Provision pod through RunPod SDK")
    p_prov.add_argument("--name", default="chess-bot-sdk-pod")
    p_prov.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    p_prov.add_argument("--gpu-count", type=int, default=1)
    p_prov.add_argument("--gpu-type-id", default="")
    p_prov.add_argument("--min-memory-gb", type=int, default=24)
    p_prov.add_argument("--max-hourly-price", type=float, default=0.0)
    p_prov.add_argument("--template-id", default="")
    p_prov.add_argument("--template-name", default="chess-bot-training")
    _bool_arg(p_prov, "include-runpod-templates", True, "Include RunPod templates")
    _bool_arg(p_prov, "include-public-templates", True, "Include public templates")
    p_prov.add_argument("--ports", nargs="*", default=["22/tcp", "8888/http", "8000/http"])
    p_prov.add_argument("--volume-mount-path", default="/workspace")
    p_prov.add_argument("--volume-in-gb", type=int, default=40)
    p_prov.add_argument("--container-disk-in-gb", type=int, default=15)
    p_prov.add_argument("--env", action="append", default=[], help="KEY=VALUE (repeatable)")
    _bool_arg(p_prov, "interruptible", False, "Request spot/interruptible instance")
    _bool_arg(p_prov, "support-public-ip", True, "Enable public IP support")
    _bool_arg(p_prov, "wait-ready", True, "Poll until pod reaches running/ready")
    p_prov.add_argument("--wait-timeout-seconds", type=int, default=900)
    p_prov.add_argument("--wait-poll-seconds", type=int, default=10)
    p_prov.set_defaults(func=cmd_provision)

    p_status = sub.add_parser("pod-status", help="Fetch one pod status via RunPod SDK")
    p_status.add_argument("--pod-id", required=True)
    p_status.set_defaults(func=cmd_pod_status)

    p_stop = sub.add_parser("pod-stop", help="Stop one pod via RunPod SDK")
    p_stop.add_argument("--pod-id", required=True)
    p_stop.set_defaults(func=cmd_pod_stop)

    p_term = sub.add_parser("pod-terminate", help="Terminate one pod via RunPod SDK")
    p_term.add_argument("--pod-id", required=True)
    p_term.set_defaults(func=cmd_pod_terminate)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
