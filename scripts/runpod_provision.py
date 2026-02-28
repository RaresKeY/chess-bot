#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.chessbot.secrets import default_dotenv_paths, resolve_secret


REST_BASE = "https://rest.runpod.io/v1"
GRAPHQL_BASE = "https://api.runpod.io/graphql"
DEFAULT_KEYRING_SERVICE = "runpod"
DEFAULT_KEYRING_USERNAME = "RUNPOD_API_KEY"
HTTP_USER_AGENT = "chess-bot-runpod-cli/1.0 (+local automation)"


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return body.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _raise_graphql_access_error(exc: urllib.error.HTTPError, *, operation: str) -> None:
    body = _read_http_error_body(exc).strip()
    if exc.code == 403:
        msg = (
            f"RunPod GraphQL request was denied (HTTP 403) during {operation}. "
            "This helper uses GraphQL for GPU discovery (`gpuTypes`) and your API key appears to lack GraphQL access/scopes, "
            "or GraphQL access is restricted for this key/account. "
            "REST auth may still work (for example `template-list`). "
            "Fix options: use a RunPod API key with GraphQL access, launch from the RunPod UI template, "
            "or use `provision --gpu-type-id <gpuTypeId>` to bypass GPU discovery in this helper."
        )
        if body:
            msg += f" Response body: {body}"
        raise SystemExit(msg)
    detail = f"RunPod GraphQL request failed (HTTP {exc.code}) during {operation}"
    if body:
        detail += f": {body}"
    raise SystemExit(detail)


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
        override_var_names=("RUNPOD_DOTENV_PATH", "CHESSBOT_DOTENV_PATH"),
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


def _http_json(
    method: str,
    url: str,
    *,
    bearer_token: str,
    body: Optional[Dict[str, Any]] = None,
    timeout_s: int = 60,
) -> Any:
    data = None
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
        "User-Agent": HTTP_USER_AGENT,
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = _read_http_error_body(exc).strip()
        detail = f"HTTP {exc.code} for {method} {url}"
        if body:
            detail += f": {body}"
        raise SystemExit(detail)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}


def _graphql_json(
    endpoint: str,
    *,
    api_key: str,
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.query:
        keep = []
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            if k.lower() == "api_key":
                continue
            keep.append((k, v))
        endpoint = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(keep), parsed.fragment)
        )
    req = urllib.request.Request(
        endpoint,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": HTTP_USER_AGENT,
        },
        data=json.dumps({"query": query, "variables": variables or {}}).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(body)
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=True))
    return payload


def _list_templates(
    api_key: str,
    *,
    include_runpod_templates: bool = True,
    include_public_templates: bool = True,
    include_serverless: bool = False,
    rest_base: str = REST_BASE,
) -> List[Dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "includeRunpodTemplates": str(bool(include_runpod_templates)).lower(),
            "includePublicTemplates": str(bool(include_public_templates)).lower(),
            "includeServerless": str(bool(include_serverless)).lower(),
        }
    )
    url = f"{rest_base.rstrip('/')}/templates?{query}"
    out = _http_json("GET", url, bearer_token=api_key)
    if isinstance(out, list):
        return [x for x in out if isinstance(x, dict)]
    if isinstance(out, dict):
        items = out.get("items")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
        data = out.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


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


def _gpu_types(api_key: str, *, graphql_endpoint: str = GRAPHQL_BASE) -> List[Dict[str, Any]]:
    query = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        maxGpuCount
        maxGpuCountSecureCloud
        maxGpuCountCommunityCloud
        securePrice
        communityPrice
        secureSpotPrice
        communitySpotPrice
      }
    }
    """
    try:
        payload = _graphql_json(graphql_endpoint, api_key=api_key, query=query)
    except urllib.error.HTTPError as exc:
        _raise_graphql_access_error(exc, operation="gpu-search")
    items = ((payload.get("data") or {}).get("gpuTypes") or [])
    return [x for x in items if isinstance(x, dict)]


def _rank_gpu_rows(
    rows: List[Dict[str, Any]],
    *,
    cloud_type: str,
    min_memory_gb: int = 0,
    max_hourly_price: float = 0.0,
) -> List[Dict[str, Any]]:
    want_secure = str(cloud_type).upper() == "SECURE"
    out: List[Dict[str, Any]] = []
    for row in rows:
        mem = int(row.get("memoryInGb") or 0)
        if mem < int(min_memory_gb):
            continue
        price_key = "securePrice" if want_secure else "communityPrice"
        spot_key = "secureSpotPrice" if want_secure else "communitySpotPrice"
        max_count_key = "maxGpuCountSecureCloud" if want_secure else "maxGpuCountCommunityCloud"
        price = row.get(price_key)
        try:
            price_f = float(price) if price is not None else 0.0
        except Exception:
            price_f = 0.0
        if max_hourly_price and price_f > float(max_hourly_price):
            continue
        max_count = int(row.get(max_count_key) or row.get("maxGpuCount") or 0)
        out.append(
            {
                "id": str(row.get("id", "")),
                "display_name": str(row.get("displayName", "")),
                "memory_gb": mem,
                "cloud_type": "SECURE" if want_secure else "COMMUNITY",
                "max_gpu_count": max_count,
                "price_per_hr": price_f,
                "spot_price_per_hr": row.get(spot_key),
                "_sort_price": price_f if price_f > 0 else 1e9,
            }
        )
    out.sort(key=lambda x: (0 if x["max_gpu_count"] > 0 else 1, x["_sort_price"], -x["memory_gb"], x["display_name"]))
    for r in out:
        r.pop("_sort_price", None)
    return out


def _create_pod(
    api_key: str,
    *,
    rest_base: str,
    name: str,
    template_id: str,
    cloud_type: str,
    gpu_type_ids: Optional[List[str]] = None,
    gpu_count: int = 1,
    ports: Optional[List[str]] = None,
    volume_mount_path: str = "/workspace",
    volume_in_gb: Optional[int] = None,
    container_disk_in_gb: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None,
    support_public_ip: Optional[bool] = None,
    interruptible: Optional[bool] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "name": name,
        "templateId": template_id,
        "computeType": "GPU",
        "cloudType": str(cloud_type).upper(),
        "gpuCount": int(gpu_count),
        "volumeMountPath": volume_mount_path,
    }
    if gpu_type_ids:
        body["gpuTypeIds"] = gpu_type_ids
        body["gpuTypePriority"] = "availability"
    if ports:
        body["ports"] = ports
    if volume_in_gb is not None:
        body["volumeInGb"] = int(volume_in_gb)
    if container_disk_in_gb is not None:
        body["containerDiskInGb"] = int(container_disk_in_gb)
    if env_vars:
        body["env"] = env_vars
    if support_public_ip is not None:
        body["supportPublicIp"] = bool(support_public_ip)
    if interruptible is not None:
        body["interruptible"] = bool(interruptible)
    return _http_json("POST", f"{rest_base.rstrip('/')}/pods", bearer_token=api_key, body=body)


def _get_pod(api_key: str, *, rest_base: str, pod_id: str) -> Dict[str, Any]:
    out = _http_json("GET", f"{rest_base.rstrip('/')}/pods/{urllib.parse.quote(pod_id)}", bearer_token=api_key)
    return out if isinstance(out, dict) else {"response": out}


def _parse_env_items(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --env item (expected KEY=VALUE): {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --env item (empty key): {item}")
        out[key] = value
    return out


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=True))


def cmd_gpu_search(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing RunPod API key (checked --api-key, RUNPOD_API_KEY, keyring, and .env fallback)")
    rows = _gpu_types(api_key, graphql_endpoint=args.graphql_endpoint)
    ranked = _rank_gpu_rows(
        rows,
        cloud_type=args.cloud_type,
        min_memory_gb=args.min_memory_gb,
        max_hourly_price=args.max_hourly_price,
    )
    if args.limit > 0:
        ranked = ranked[: args.limit]
    _print_json({"gpus": ranked, "count": len(ranked), "cloud_type": args.cloud_type})
    return 0


def cmd_template_list(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing RunPod API key (checked --api-key, RUNPOD_API_KEY, keyring, and .env fallback)")
    templates = _list_templates(
        api_key,
        include_runpod_templates=args.include_runpod_templates,
        include_public_templates=args.include_public_templates,
        include_serverless=args.include_serverless,
        rest_base=args.rest_base,
    )
    if args.template_name:
        templates = [t for t in templates if args.template_name.lower() in str(t.get("name", "")).lower()]
    if args.pods_only:
        templates = [t for t in templates if not bool(t.get("isServerless"))]
    if args.limit > 0:
        templates = templates[: args.limit]
    _print_json({"templates": templates, "count": len(templates)})
    return 0


def cmd_provision(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing RunPod API key (checked --api-key, RUNPOD_API_KEY, keyring, and .env fallback)")

    chosen_gpu: Optional[Dict[str, Any]] = None
    if args.gpu_type_id:
        # Allow provisioning with an explicit GPU type even when GraphQL gpuTypes
        # discovery is unavailable to the API key (common for scoped keys).
        chosen_gpu = {
            "id": str(args.gpu_type_id),
            "display_name": str(args.gpu_type_id),
            "memory_gb": None,
            "cloud_type": str(args.cloud_type).upper(),
            "max_gpu_count": None,
            "price_per_hr": None,
            "spot_price_per_hr": None,
            "graphql_lookup_skipped": True,
        }
    else:
        gpu_rows = _gpu_types(api_key, graphql_endpoint=args.graphql_endpoint)
        ranked_gpus = _rank_gpu_rows(
            gpu_rows,
            cloud_type=args.cloud_type,
            min_memory_gb=args.min_memory_gb,
            max_hourly_price=args.max_hourly_price,
        )
        chosen_gpu = next((g for g in ranked_gpus if g.get("max_gpu_count", 0) >= int(args.gpu_count)), None)
        if chosen_gpu is None:
            raise SystemExit("No GPU candidates available after filters")

    templates = _list_templates(
        api_key,
        include_runpod_templates=args.include_runpod_templates,
        include_public_templates=args.include_public_templates,
        include_serverless=False,
        rest_base=args.rest_base,
    )
    chosen_template = _choose_template(templates, template_id=args.template_id, template_name=args.template_name)

    env_vars = _parse_env_items(args.env or [])
    if args.use_runpod_training_preset_env:
        env_vars.setdefault("START_IDLE_WATCHDOG", "1")
        env_vars.setdefault("START_INFERENCE_API", "1")
        env_vars.setdefault("START_JUPYTER", "1")

    create_resp = _create_pod(
        api_key,
        rest_base=args.rest_base,
        name=args.name,
        template_id=str(chosen_template.get("id", "")),
        cloud_type=args.cloud_type,
        gpu_type_ids=[chosen_gpu["id"]],
        gpu_count=args.gpu_count,
        ports=args.ports,
        volume_mount_path=args.volume_mount_path,
        volume_in_gb=args.volume_in_gb,
        container_disk_in_gb=args.container_disk_in_gb,
        env_vars=env_vars or None,
        support_public_ip=(args.cloud_type.upper() == "COMMUNITY") if args.support_public_ip_auto else None,
        interruptible=args.interruptible,
    )

    pod_id = str((create_resp or {}).get("id") or (create_resp or {}).get("podId") or "")
    out: Dict[str, Any] = {
        "chosen_gpu": chosen_gpu,
        "chosen_template": {
            "id": chosen_template.get("id"),
            "name": chosen_template.get("name"),
            "imageName": chosen_template.get("imageName"),
            "ports": chosen_template.get("ports"),
            "isRunpod": chosen_template.get("isRunpod"),
            "isPublic": chosen_template.get("isPublic"),
        },
        "create_response": create_resp,
        "pod_id": pod_id,
    }

    if args.wait_ready and pod_id:
        deadline = time.time() + max(5, int(args.wait_timeout_seconds))
        last_status: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                last_status = _get_pod(api_key, rest_base=args.rest_base, pod_id=pod_id)
            except urllib.error.URLError:
                last_status = {}
            public_ip = str(last_status.get("publicIp") or "")
            desired_status = str(last_status.get("desiredStatus") or "")
            runtime = last_status.get("runtime") if isinstance(last_status.get("runtime"), dict) else {}
            runtime_status = str((runtime or {}).get("uptimeSeconds", ""))
            if public_ip:
                out["pod_status"] = last_status
                break
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "wait_status": {
                                "pod_id": pod_id,
                                "desiredStatus": desired_status,
                                "publicIp": public_ip,
                                "runtime_uptime": runtime_status,
                            }
                        },
                        ensure_ascii=True,
                    ),
                    file=sys.stderr,
                )
            time.sleep(max(2, int(args.wait_poll_seconds)))
        else:
            out["pod_status"] = last_status
            out["wait_ready_timeout"] = True

    _print_json(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RunPod API helper (GPU search, template selection, pod provisioning)")
    p.add_argument(
        "--api-key",
        default="",
        help="RunPod API key (falls back to env RUNPOD_API_KEY, keyring, then .env files)",
    )
    p.add_argument("--keyring-service", default=DEFAULT_KEYRING_SERVICE, help="Keyring service name for API key lookup")
    p.add_argument("--keyring-username", default=DEFAULT_KEYRING_USERNAME, help="Keyring username/key for API key lookup")
    p.add_argument("--rest-base", default=REST_BASE, help="RunPod REST API base URL")
    p.add_argument("--graphql-endpoint", default=GRAPHQL_BASE, help="RunPod GraphQL endpoint (without api_key query param)")
    _bool_arg(p, "verbose", True, "Verbose provisioning wait logs")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_gpu = sub.add_parser("gpu-search", help="List/rank GPU types by cloud and price/memory filters")
    p_gpu.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    p_gpu.add_argument("--min-memory-gb", type=int, default=0)
    p_gpu.add_argument("--max-hourly-price", type=float, default=0.0)
    p_gpu.add_argument("--limit", type=int, default=20)
    p_gpu.set_defaults(func=cmd_gpu_search)

    p_tpl = sub.add_parser("template-list", help="List Pod templates")
    _bool_arg(p_tpl, "include-runpod-templates", True, "Include official RunPod templates")
    _bool_arg(p_tpl, "include-public-templates", True, "Include public templates")
    _bool_arg(p_tpl, "include-serverless", False, "Include serverless templates in results")
    _bool_arg(p_tpl, "pods-only", True, "Filter out serverless templates")
    p_tpl.add_argument("--template-name", default="", help="Optional case-insensitive name substring filter")
    p_tpl.add_argument("--limit", type=int, default=50)
    p_tpl.set_defaults(func=cmd_template_list)

    p_prov = sub.add_parser("provision", help="Search GPUs, choose template, create Pod")
    p_prov.add_argument("--name", default="chessbot-train", help="New pod name")
    p_prov.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    p_prov.add_argument("--gpu-count", type=int, default=1)
    p_prov.add_argument("--gpu-type-id", default="", help="Optional explicit GPU id or display name to force selection")
    p_prov.add_argument("--min-memory-gb", type=int, default=0)
    p_prov.add_argument("--max-hourly-price", type=float, default=0.0)
    p_prov.add_argument("--template-id", default="", help="Template ID to use")
    p_prov.add_argument("--template-name", default="", help="Template name or substring to use")
    _bool_arg(p_prov, "include-runpod-templates", True, "Include official RunPod templates")
    _bool_arg(p_prov, "include-public-templates", True, "Include public templates")
    p_prov.add_argument("--ports", action="append", default=["22/tcp", "8888/http", "8000/http"])
    p_prov.add_argument("--volume-mount-path", default="/workspace")
    p_prov.add_argument("--volume-in-gb", type=int, default=None)
    p_prov.add_argument("--container-disk-in-gb", type=int, default=None)
    p_prov.add_argument("--env", action="append", default=[], help="Repeat KEY=VALUE env vars for pod creation")
    _bool_arg(
        p_prov,
        "use-runpod-training-preset-env",
        False,
        "Inject common defaults for this repo's RunPod training/inference services",
    )
    _bool_arg(
        p_prov,
        "support-public-ip-auto",
        True,
        "Set supportPublicIp=true automatically for COMMUNITY cloud",
    )
    _bool_arg(
        p_prov,
        "interruptible",
        False,
        "Request spot/interruptible pod capacity",
    )
    _bool_arg(p_prov, "wait-ready", True, "Poll pod until public IP appears")
    p_prov.add_argument("--wait-timeout-seconds", type=int, default=600)
    p_prov.add_argument("--wait-poll-seconds", type=int, default=10)
    p_prov.set_defaults(func=cmd_provision)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
