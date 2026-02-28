from __future__ import annotations

import pytest

from scripts.hf_dataset_fetch import _latest_version, _parse_repo_dataset_versions, _resolve_token, _select_versions


def test_select_versions_single_dataset_latest_without_version() -> None:
    repo_files = [
        "validated_datasets/elite_2025-11_game/validated-20260226T100000Z/manifest.json",
        "validated_datasets/elite_2025-11_game/validated-20260226T151127Z/manifest.json",
        "validated_datasets/elite_2025-10_game/validated-20260226T151127Z/manifest.json",
    ]
    versions = _parse_repo_dataset_versions(repo_files, "validated_datasets")
    selected = _select_versions(
        versions_by_dataset=versions,
        all_latest=False,
        dataset_name="elite_2025-11_game",
        version="",
    )
    assert selected == [("elite_2025-11_game", "validated-20260226T151127Z")]


def test_select_versions_single_dataset_missing_raises() -> None:
    with pytest.raises(SystemExit):
        _select_versions(
            versions_by_dataset={},
            all_latest=False,
            dataset_name="missing",
            version="",
        )


def test_resolve_token_falls_back_to_dotenv(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env.hf_dataset"
    dotenv.write_text("HF_READ_TOKEN=dotenv-hf-read-token\n", encoding="utf-8")
    monkeypatch.delenv("HF_READ_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("src.chessbot.secrets.token_from_keyring", lambda *_: "")
    monkeypatch.setattr("scripts.hf_dataset_fetch.default_dotenv_paths", lambda **_: [dotenv])
    assert _resolve_token(None, "huggingface", "codex_hf_read_token") == "dotenv-hf-read-token"


def test_latest_version_prefers_newer_timestamp_even_with_mixed_prefixes() -> None:
    versions = [
        "validated-20260226T151127Z",
        "20260227T044455Z",
        "validated-20260225T235959Z",
    ]
    assert _latest_version(versions) == "20260227T044455Z"


def test_select_versions_all_latest_uses_timestamp_aware_ordering() -> None:
    versions_by_dataset = {
        "elite_2025-11_game": ["validated-20260226T151127Z", "20260227T044455Z"],
        "elite_2025-10_game": ["validated-20260226T151127Z", "20260227T044455Z"],
    }
    selected = _select_versions(
        versions_by_dataset=versions_by_dataset,
        all_latest=True,
        dataset_name="",
        version="",
    )
    assert selected == [
        ("elite_2025-10_game", "20260227T044455Z"),
        ("elite_2025-11_game", "20260227T044455Z"),
    ]
