import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.vast_provision import (
    _http_json,
    _parse_env_items,
    _rank_offers,
    _resolve_api_key,
    cmd_provision,
)


class _BytesResponse:
    def __init__(self, payload: bytes):
        self._bio = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._bio.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class VastApiHelperTests(unittest.TestCase):
    def test_http_json_uses_bearer_header(self):
        with mock.patch("scripts.vast_provision.urllib.request.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _BytesResponse(b'{"ok":true}')
            _http_json("GET", "https://console.vast.ai/api/v0/instances/", bearer_token="SECRET")
        req = urlopen_mock.call_args.args[0]
        self.assertEqual(req.headers.get("Authorization"), "Bearer SECRET")
        self.assertTrue((req.headers.get("User-agent") or "").startswith("chess-bot-vast-cli/"))

    def test_rank_offers_filters_and_sorts(self):
        offers = [
            {"id": 11, "gpu_name": "A", "num_gpus": 1, "dph_total": 1.2, "reliability": 0.98, "gpu_ram": 24576},
            {"id": 22, "gpu_name": "B", "num_gpus": 1, "dph_total": 0.8, "reliability": 0.95, "gpu_ram": 24576},
            {"id": 33, "gpu_name": "C", "num_gpus": 1, "dph_total": 0.7, "reliability": 0.70, "gpu_ram": 8192},
        ]
        ranked = _rank_offers(offers, max_dph_total=1.5, min_reliability=0.9, min_gpu_ram_gb=20)
        self.assertEqual([row["ask_id"] for row in ranked], [22, 11])

    def test_parse_env_items_builds_cli_env_string(self):
        env_str = _parse_env_items(["A=1", "B=two"])
        self.assertEqual(env_str, "-e A=1 -e B=two")

    def test_resolve_api_key_uses_dotenv_fallback(self):
        args = argparse.Namespace(api_key="", keyring_service="vast", keyring_username="VAST_API_KEY")
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("src.chessbot.secrets.token_from_keyring", return_value=""):
                with tempfile.TemporaryDirectory() as td:
                    dotenv = Path(td) / ".env.vast"
                    dotenv.write_text("VAST_API_KEY=dotenv-vast-token\n", encoding="utf-8")
                    with mock.patch("scripts.vast_provision.default_dotenv_paths", return_value=[dotenv]):
                        token = _resolve_api_key(args)
        self.assertEqual(token, "dotenv-vast-token")

    def test_provision_with_explicit_offer_id_skips_search(self):
        args = argparse.Namespace(
            api_key="SECRET",
            keyring_service="vast",
            keyring_username="VAST_API_KEY",
            api_base="https://console.vast.ai/api/v0",
            verbose=False,
            offer_id=987,
            gpu_count=1,
            gpu_name="",
            max_dph_total=0.0,
            min_reliability=0.0,
            min_gpu_ram_gb=0.0,
            order="dph_total",
            verified_only=True,
            rentable_only=True,
            image="vastai/base-image:@vastai-automatic-tag",
            disk=40,
            label="test",
            runtype="ssh",
            template_hash_id="",
            target_state="running",
            onstart_cmd="",
            env=[],
            wait_ready=False,
            wait_timeout_seconds=30,
            wait_poll_seconds=3,
        )
        with mock.patch("scripts.vast_provision._resolve_api_key", return_value="SECRET"), mock.patch(
            "scripts.vast_provision._search_offers"
        ) as search_mock, mock.patch(
            "scripts.vast_provision._create_instance",
            return_value={"new_contract": 12345},
        ) as create_mock, mock.patch(
            "scripts.vast_provision._print_json"
        ):
            rc = cmd_provision(args)

        self.assertEqual(rc, 0)
        search_mock.assert_not_called()
        self.assertEqual(create_mock.call_args.kwargs["ask_id"], 987)


if __name__ == "__main__":
    unittest.main()
