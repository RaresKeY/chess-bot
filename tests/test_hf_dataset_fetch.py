from __future__ import annotations

import pytest

from scripts.hf_dataset_fetch import _parse_repo_dataset_versions, _select_versions


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
