from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Sequence


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_inline_comment(raw: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            if idx == 0 or raw[idx - 1].isspace():
                return raw[:idx]
    return raw


def _decode_quoted(value: str) -> str:
    if len(value) < 2:
        return value
    if value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    if value[0] == '"' and value[-1] == '"':
        inner = value[1:-1]
        return bytes(inner, "utf-8").decode("unicode_escape")
    return value


def parse_dotenv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.match(key):
            continue
        value = _strip_inline_comment(raw_value).strip()
        value = _decode_quoted(value)
        out[key] = value.strip()
    return out


def default_dotenv_paths(
    *,
    repo_root: str | Path,
    override_var_names: Sequence[str],
    fallback_filenames: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    if env is None:
        env = os.environ
    root = Path(repo_root).resolve()
    out: list[Path] = []
    seen: set[str] = set()

    def _append(raw_path: str) -> None:
        if not raw_path:
            return
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        out.append(candidate)

    for name in override_var_names:
        raw_value = str(env.get(name, "") or "").strip()
        if not raw_value:
            continue
        for item in raw_value.split(os.pathsep):
            _append(item.strip())

    for filename in fallback_filenames:
        _append(filename)
    return out


def lookup_dotenv_value(keys: Sequence[str], paths: Sequence[Path]) -> str:
    for path in paths:
        if not path.is_file():
            continue
        parsed = parse_dotenv_file(path)
        for key in keys:
            value = str(parsed.get(key, "") or "").strip()
            if value:
                return value
    return ""


def token_from_keyring(service: str, username: str) -> str:
    try:
        import keyring  # type: ignore
    except Exception:
        return ""
    try:
        token = keyring.get_password(service, username)
    except Exception:
        return ""
    return str(token or "").strip()


def resolve_secret(
    *,
    explicit_value: str = "",
    env_var_names: Sequence[str] = (),
    keyring_service: str = "",
    keyring_username: str = "",
    dotenv_keys: Sequence[str] = (),
    dotenv_paths: Sequence[Path] = (),
    order: Sequence[str] = ("explicit", "env", "keyring", "dotenv"),
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    if env is None:
        env = os.environ
    effective_dotenv_keys = tuple(dotenv_keys) if dotenv_keys else tuple(env_var_names)

    for source in order:
        if source == "explicit":
            v = str(explicit_value or "").strip()
            if v:
                return v, "explicit"
        elif source == "env":
            for name in env_var_names:
                v = str(env.get(name, "") or "").strip()
                if v:
                    return v, "env"
        elif source == "keyring":
            if keyring_service and keyring_username:
                v = token_from_keyring(keyring_service, keyring_username)
                if v:
                    return v, "keyring"
        elif source == "dotenv":
            if effective_dotenv_keys and dotenv_paths:
                v = lookup_dotenv_value(effective_dotenv_keys, dotenv_paths)
                if v:
                    return v, "dotenv"
    return "", ""
