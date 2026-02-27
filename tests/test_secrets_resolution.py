from __future__ import annotations

from pathlib import Path

from src.chessbot.secrets import parse_dotenv_file, resolve_secret


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
