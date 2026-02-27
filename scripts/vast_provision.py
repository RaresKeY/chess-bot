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


API_BASE = "https://console.vast.ai/api/v0"
DEFAULT_KEYRING_SERVICE = "vast"
DEFAULT_KEYRING_USERNAME = "VAST_API_KEY"
HTTP_USER_AGENT = "chess-bot-vast-cli/1.0 (+local automation)"


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


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


def _resolve_api_key(args: argparse.Namespace) -> str:
    dotenv_paths = default_dotenv_paths(
        repo_root=REPO_ROOT,
        override_var_names=("VAST_DOTENV_PATH", "CHESSBOT_DOTENV_PATH"),
        fallback_filenames=(".env.vast", ".env"),
    )
    value, _ = resolve_secret(
        explicit_value=str(getattr(args, "api_key", "") or ""),
        env_var_names=("VAST_API_KEY",),
        keyring_service=str(args.keyring_service),
        keyring_username=str(args.keyring_username),
        dotenv_keys=("VAST_API_KEY",),
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
        detail = f"HTTP {exc.code} for {method} {url}"
        body_text = _read_http_error_body(exc).strip()
        if body_text:
            detail += f": {body_text}"
        raise SystemExit(detail)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}


def _normalize_offers(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("offers", "results", "rows", "instances"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
    return []


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _rank_offers(
    offers: List[Dict[str, Any]],
    *,
    max_dph_total: float = 0.0,
    min_reliability: float = 0.0,
    min_gpu_ram_gb: float = 0.0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in offers:
        dph = _as_float(row.get("dph_total"), default=1e9)
        reliability = _as_float(row.get("reliability"), default=0.0)
        gpu_ram = _as_float(row.get("gpu_ram"), default=0.0)
        gpu_ram_gb = gpu_ram / 1024.0 if gpu_ram > 128 else gpu_ram
        if max_dph_total > 0 and dph > max_dph_total:
            continue
        if min_reliability > 0 and reliability < min_reliability:
            continue
        if min_gpu_ram_gb > 0 and gpu_ram_gb < min_gpu_ram_gb:
            continue
        out.append(
            {
                "ask_id": int(row.get("id") or 0),
                "machine_id": row.get("machine_id"),
                "gpu_name": row.get("gpu_name"),
                "num_gpus": row.get("num_gpus"),
                "dph_total": dph,
                "reliability": reliability,
                "gpu_ram_gb": gpu_ram_gb,
                "inet_up": row.get("inet_up"),
                "inet_down": row.get("inet_down"),
                "raw": row,
                "_sort": (dph if dph > 0 else 1e9, -reliability, -gpu_ram_gb),
            }
        )
    out.sort(key=lambda x: x["_sort"])
    for row in out:
        row.pop("_sort", None)
    return out


def _search_offers(
    api_key: str,
    *,
    api_base: str,
    gpu_count: int,
    gpu_name: str,
    verified_only: bool,
    rentable_only: bool,
    order: str,
) -> List[Dict[str, Any]]:
    body: Dict[str, Any] = {"order": order}
    if verified_only:
        body["verified"] = {"eq": True}
    if rentable_only:
        body["rentable"] = {"eq": True}
    if gpu_count > 0:
        body["num_gpus"] = {"eq": int(gpu_count)}
    if gpu_name:
        body["gpu_name"] = {"eq": gpu_name}
    payload = _http_json("POST", f"{api_base.rstrip('/')}/bundles/", bearer_token=api_key, body=body)
    return _normalize_offers(payload)


def _show_instances(api_key: str, *, api_base: str) -> Dict[str, Any]:
    out = _http_json("GET", f"{api_base.rstrip('/')}/instances/", bearer_token=api_key)
    return out if isinstance(out, dict) else {"instances": out}


def _show_instance(api_key: str, *, api_base: str, instance_id: int) -> Dict[str, Any]:
    out = _http_json("GET", f"{api_base.rstrip('/')}/instances/{int(instance_id)}/", bearer_token=api_key)
    return out if isinstance(out, dict) else {"instance": out}


def _create_instance(
    api_key: str,
    *,
    api_base: str,
    ask_id: int,
    image: str,
    disk_gb: float,
    label: str,
    runtype: str,
    env: str,
    onstart_cmd: str,
    template_hash_id: str,
    target_state: str,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if image:
        body["image"] = image
    if disk_gb > 0:
        body["disk"] = float(disk_gb)
    if label:
        body["label"] = label
    if runtype:
        body["runtype"] = runtype
    if env:
        body["env"] = env
    if onstart_cmd:
        body["onstart_cmd"] = onstart_cmd
    if template_hash_id:
        body["template_hash_id"] = template_hash_id
    if target_state:
        body["target_state"] = target_state
    out = _http_json("PUT", f"{api_base.rstrip('/')}/asks/{int(ask_id)}/", bearer_token=api_key, body=body)
    return out if isinstance(out, dict) else {"response": out}


def _manage_instance(api_key: str, *, api_base: str, instance_id: int, state: str = "", label: str = "") -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if state:
        body["state"] = state
    if label:
        body["label"] = label
    if not body:
        raise SystemExit("manage-instance requires --state and/or --label")
    out = _http_json(
        "PUT",
        f"{api_base.rstrip('/')}/instances/{int(instance_id)}/",
        bearer_token=api_key,
        body=body,
    )
    return out if isinstance(out, dict) else {"response": out}


def _destroy_instance(api_key: str, *, api_base: str, instance_id: int) -> Dict[str, Any]:
    out = _http_json("DELETE", f"{api_base.rstrip('/')}/instances/{int(instance_id)}/", bearer_token=api_key)
    return out if isinstance(out, dict) else {"response": out}


def _parse_env_items(items: List[str]) -> str:
    if not items:
        return ""
    parts: List[str] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --env item (expected KEY=VALUE): {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --env item (empty key): {item}")
        parts.extend(["-e", f"{key}={value}"])
    return " ".join(parts)


def _extract_instance_id(create_resp: Dict[str, Any]) -> int:
    for key in ("new_contract", "instance_id", "id"):
        value = create_resp.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return 0


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=True))


def cmd_offer_search(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing Vast API key (checked --api-key, VAST_API_KEY, keyring, and .env fallback)")
    offers = _search_offers(
        api_key,
        api_base=args.api_base,
        gpu_count=args.gpu_count,
        gpu_name=args.gpu_name,
        verified_only=args.verified_only,
        rentable_only=args.rentable_only,
        order=args.order,
    )
    ranked = _rank_offers(
        offers,
        max_dph_total=args.max_dph_total,
        min_reliability=args.min_reliability,
        min_gpu_ram_gb=args.min_gpu_ram_gb,
    )
    if args.limit > 0:
        ranked = ranked[: args.limit]
    _print_json({"offers": ranked, "count": len(ranked)})
    return 0


def cmd_instance_list(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing Vast API key (checked --api-key, VAST_API_KEY, keyring, and .env fallback)")
    _print_json(_show_instances(api_key, api_base=args.api_base))
    return 0


def cmd_provision(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing Vast API key (checked --api-key, VAST_API_KEY, keyring, and .env fallback)")

    chosen_offer: Optional[Dict[str, Any]] = None
    if args.offer_id > 0:
        chosen_offer = {"ask_id": int(args.offer_id), "raw": None, "source": "explicit"}
    else:
        offers = _search_offers(
            api_key,
            api_base=args.api_base,
            gpu_count=args.gpu_count,
            gpu_name=args.gpu_name,
            verified_only=args.verified_only,
            rentable_only=args.rentable_only,
            order=args.order,
        )
        ranked = _rank_offers(
            offers,
            max_dph_total=args.max_dph_total,
            min_reliability=args.min_reliability,
            min_gpu_ram_gb=args.min_gpu_ram_gb,
        )
        if not ranked:
            raise SystemExit("No Vast offers matched requested filters")
        chosen_offer = ranked[0]

    env_str = _parse_env_items(args.env or [])
    create_resp = _create_instance(
        api_key,
        api_base=args.api_base,
        ask_id=int(chosen_offer["ask_id"]),
        image=args.image,
        disk_gb=args.disk,
        label=args.label,
        runtype=args.runtype,
        env=env_str,
        onstart_cmd=args.onstart_cmd,
        template_hash_id=args.template_hash_id,
        target_state=args.target_state,
    )

    instance_id = _extract_instance_id(create_resp)
    out: Dict[str, Any] = {
        "chosen_offer": chosen_offer,
        "create_response": create_resp,
        "instance_id": instance_id,
    }

    if args.wait_ready and instance_id > 0:
        deadline = time.time() + max(5, int(args.wait_timeout_seconds))
        last_status: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                last_status = _show_instance(api_key, api_base=args.api_base, instance_id=instance_id)
            except SystemExit:
                last_status = {}
            inst = last_status.get("instances") if isinstance(last_status.get("instances"), dict) else {}
            actual_status = str(inst.get("actual_status") or inst.get("cur_state") or "")
            ssh_host = str(inst.get("ssh_host") or inst.get("public_ipaddr") or inst.get("public_ip") or "")
            if actual_status.lower() == "running" and ssh_host:
                out["show_response"] = last_status
                break
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "wait_status": {
                                "instance_id": instance_id,
                                "actual_status": actual_status,
                                "ssh_host": ssh_host,
                            }
                        },
                        ensure_ascii=True,
                    ),
                    file=sys.stderr,
                )
            time.sleep(max(2, int(args.wait_poll_seconds)))
        else:
            out["show_response"] = last_status
            out["wait_ready_timeout"] = True

    _print_json(out)
    return 0


def cmd_manage_instance(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing Vast API key (checked --api-key, VAST_API_KEY, keyring, and .env fallback)")
    out = _manage_instance(
        api_key,
        api_base=args.api_base,
        instance_id=args.instance_id,
        state=args.state,
        label=args.label,
    )
    _print_json({"instance_id": args.instance_id, "response": out})
    return 0


def cmd_destroy_instance(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing Vast API key (checked --api-key, VAST_API_KEY, keyring, and .env fallback)")
    out = _destroy_instance(api_key, api_base=args.api_base, instance_id=args.instance_id)
    _print_json({"instance_id": args.instance_id, "response": out})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast.ai API helper (offer search, provision, instance lifecycle)")
    p.add_argument(
        "--api-key",
        default="",
        help="Vast API key (falls back to env VAST_API_KEY, keyring, then .env files)",
    )
    p.add_argument("--keyring-service", default=DEFAULT_KEYRING_SERVICE, help="Keyring service name for API key lookup")
    p.add_argument("--keyring-username", default=DEFAULT_KEYRING_USERNAME, help="Keyring username/key for API key lookup")
    p.add_argument("--api-base", default=API_BASE, help="Vast API base URL")
    _bool_arg(p, "verbose", True, "Verbose provisioning wait logs")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("offer-search", help="Search and rank Vast offers")
    p_search.add_argument("--gpu-count", type=int, default=1)
    p_search.add_argument("--gpu-name", default="")
    p_search.add_argument("--max-dph-total", type=float, default=0.0)
    p_search.add_argument("--min-reliability", type=float, default=0.0)
    p_search.add_argument("--min-gpu-ram-gb", type=float, default=0.0)
    p_search.add_argument("--order", default="dph_total")
    _bool_arg(p_search, "verified-only", True, "Filter to verified hosts")
    _bool_arg(p_search, "rentable-only", True, "Filter to currently rentable offers")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.set_defaults(func=cmd_offer_search)

    p_list = sub.add_parser("instance-list", help="List instances for authenticated user")
    p_list.set_defaults(func=cmd_instance_list)

    p_prov = sub.add_parser("provision", help="Search offers (or use explicit offer) then create instance")
    p_prov.add_argument("--offer-id", type=int, default=0, help="Optional explicit ask/offer id")
    p_prov.add_argument("--gpu-count", type=int, default=1)
    p_prov.add_argument("--gpu-name", default="")
    p_prov.add_argument("--max-dph-total", type=float, default=0.0)
    p_prov.add_argument("--min-reliability", type=float, default=0.0)
    p_prov.add_argument("--min-gpu-ram-gb", type=float, default=0.0)
    p_prov.add_argument("--order", default="dph_total")
    _bool_arg(p_prov, "verified-only", True, "Filter to verified hosts")
    _bool_arg(p_prov, "rentable-only", True, "Filter to currently rentable offers")
    p_prov.add_argument("--image", default="vastai/base-image:@vastai-automatic-tag")
    p_prov.add_argument("--disk", type=float, default=40)
    p_prov.add_argument("--label", default="")
    p_prov.add_argument("--runtype", default="ssh")
    p_prov.add_argument("--template-hash-id", default="")
    p_prov.add_argument("--target-state", default="running", choices=["running", "stopped"])
    p_prov.add_argument("--onstart-cmd", default="")
    p_prov.add_argument("--env", action="append", default=[], help="Repeat KEY=VALUE to append container env flags")
    _bool_arg(p_prov, "wait-ready", True, "Poll instance until it appears in running status with SSH host")
    p_prov.add_argument("--wait-timeout-seconds", type=int, default=600)
    p_prov.add_argument("--wait-poll-seconds", type=int, default=10)
    p_prov.set_defaults(func=cmd_provision)

    p_manage = sub.add_parser("manage-instance", help="Manage instance state and label")
    p_manage.add_argument("--instance-id", type=int, required=True)
    p_manage.add_argument("--state", default="", choices=["", "running", "stopped"])
    p_manage.add_argument("--label", default="")
    p_manage.set_defaults(func=cmd_manage_instance)

    p_destroy = sub.add_parser("destroy-instance", help="Destroy instance permanently")
    p_destroy.add_argument("--instance-id", type=int, required=True)
    p_destroy.set_defaults(func=cmd_destroy_instance)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
