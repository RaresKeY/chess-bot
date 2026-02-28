from __future__ import annotations

from pathlib import Path

from scripts import populate_env_from_keyring as mod


def test_main_writes_env_file(tmp_path: Path, monkeypatch, capsys) -> None:
    out_path = tmp_path / ".env"
    values = {
        ("runpod", "RUNPOD_API_KEY"): "rp123",
        ("huggingface", "codex_hf_read_token"): "hf-read-456",
        ("huggingface", "codex_hf_write_token"): "hf456",
        ("lichess", "lichess_api_token"): "li789",
    }

    def fake_keyring(service: str, username: str) -> str:
        return values.get((service, username), "")

    monkeypatch.setattr(mod, "token_from_keyring", fake_keyring)
    monkeypatch.setattr("sys.argv", ["populate_env_from_keyring.py", "--output", str(out_path)])

    rc = mod.main()

    assert rc == 0
    text = out_path.read_text(encoding="utf-8")
    assert 'RUNPOD_API_KEY="rp123"' in text
    assert 'HF_READ_TOKEN="hf-read-456"' in text
    assert 'HF_WRITE_TOKEN="hf456"' in text
    assert 'LICHESS_BOT_TOKEN="li789"' in text
    stdout = capsys.readouterr().out
    assert "Wrote" in stdout
    assert "rp123" not in stdout
    assert "hf-read-456" not in stdout
    assert "hf456" not in stdout
    assert "li789" not in stdout


def test_main_fails_when_missing_entries_without_allow_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(mod, "token_from_keyring", lambda *_: "")
    monkeypatch.setattr("sys.argv", ["populate_env_from_keyring.py"])

    rc = mod.main()

    assert rc == 2
    stdout = capsys.readouterr().out
    assert "Missing required keyring entries" in stdout
    assert "runpod/RUNPOD_API_KEY" in stdout


def test_main_allows_missing_entries_with_flag(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / ".env"
    monkeypatch.setattr(mod, "token_from_keyring", lambda *_: "")
    monkeypatch.setattr(
        "sys.argv",
        ["populate_env_from_keyring.py", "--allow-missing", "--output", str(out_path)],
    )

    rc = mod.main()

    assert rc == 0
    text = out_path.read_text(encoding="utf-8")
    assert 'RUNPOD_API_KEY=""' in text
    assert 'HF_READ_TOKEN=""' in text
    assert 'HF_WRITE_TOKEN=""' in text
    assert 'LICHESS_BOT_TOKEN=""' in text


def test_quote_env_value_escapes_double_quotes_and_backslashes() -> None:
    quoted = mod._quote_env_value('a"b\\c')
    assert quoted == '"a\\"b\\\\c"'
