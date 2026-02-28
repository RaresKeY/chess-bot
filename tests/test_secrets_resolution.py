from __future__ import annotations

from pathlib import Path

from src.chessbot.secrets import default_dotenv_paths, parse_dotenv_file, resolve_secret


def test_parse_dotenv_file_supports_export_quotes_and_comments(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "# comment",
                "export RUNPOD_API_KEY = \"abc123\" # trailing comment",
                "HF_TOKEN='  keep-space  '",
                "PLAIN=value",
                "BAD-KEY=ignore",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_dotenv_file(dotenv)
    assert parsed["RUNPOD_API_KEY"] == "abc123"
    assert parsed["HF_TOKEN"] == "keep-space"
    assert parsed["PLAIN"] == "value"
    assert "BAD-KEY" not in parsed


def test_resolve_secret_uses_requested_order(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("LICHESS_BOT_TOKEN=dotenv-token\n", encoding="utf-8")
    monkeypatch.setattr("src.chessbot.secrets.token_from_keyring", lambda *_: "keyring-token")
    token, source = resolve_secret(
        explicit_value="",
        env_var_names=("LICHESS_BOT_TOKEN",),
        keyring_service="lichess",
        keyring_username="lichess_api_token",
        dotenv_paths=(dotenv,),
        order=("keyring", "env", "dotenv"),
        env={"LICHESS_BOT_TOKEN": "env-token"},
    )
    assert token == "keyring-token"
    assert source == "keyring"


def test_resolve_secret_precedence_matrix(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("APP_TOKEN=dotenv-token\n", encoding="utf-8")

    monkeypatch.setattr("src.chessbot.secrets.token_from_keyring", lambda *_: "keyring-token")

    token, source = resolve_secret(
        explicit_value="explicit-token",
        env_var_names=("APP_TOKEN",),
        keyring_service="svc",
        keyring_username="user",
        dotenv_paths=(dotenv,),
        dotenv_keys=("APP_TOKEN",),
        order=("explicit", "env", "keyring", "dotenv"),
        env={"APP_TOKEN": "env-token"},
    )
    assert token == "explicit-token"
    assert source == "explicit"

    token, source = resolve_secret(
        explicit_value="",
        env_var_names=("APP_TOKEN",),
        keyring_service="svc",
        keyring_username="user",
        dotenv_paths=(dotenv,),
        dotenv_keys=("APP_TOKEN",),
        order=("explicit", "env", "keyring", "dotenv"),
        env={"APP_TOKEN": "env-token"},
    )
    assert token == "env-token"
    assert source == "env"

    token, source = resolve_secret(
        explicit_value="",
        env_var_names=("APP_TOKEN",),
        keyring_service="svc",
        keyring_username="user",
        dotenv_paths=(dotenv,),
        dotenv_keys=("APP_TOKEN",),
        order=("explicit", "env", "keyring", "dotenv"),
        env={},
    )
    assert token == "keyring-token"
    assert source == "keyring"


def test_default_dotenv_paths_override_then_fallback_order(tmp_path: Path) -> None:
    override1 = tmp_path / "custom1.env"
    override2 = tmp_path / "custom2.env"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    paths = default_dotenv_paths(
        repo_root=repo_root,
        override_var_names=("X_DOTENV",),
        fallback_filenames=(".env.runpod", ".env"),
        env={"X_DOTENV": f"{override1}:{override2}:{override1}"},
    )
    got = [str(p) for p in paths]
    assert got[0] == str(override1)
    assert got[1] == str(override2)
    assert got[2] == str((repo_root / ".env.runpod").resolve())
    assert got[3] == str((repo_root / ".env").resolve())
