import io
import json
import unittest
from unittest import mock

from deploy.runpod_cloud_training.idle_watchdog import _stop_runpod_pod
from scripts.runpod_provision import _choose_template, _graphql_json, _rank_gpu_rows


class _BytesResponse:
    def __init__(self, payload: bytes):
        self._bio = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._bio.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RunpodApiHelperTests(unittest.TestCase):
    def test_graphql_uses_bearer_header_and_strips_api_key_query_param(self):
        with mock.patch("scripts.runpod_provision.urllib.request.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _BytesResponse(b'{"data":{"gpuTypes":[]}}')
            _graphql_json(
                "https://api.runpod.io/graphql?api_key=SHOULD_NOT_LEAK&foo=bar",
                api_key="SECRET_TOKEN",
                query="query { gpuTypes { id } }",
            )

        req = urlopen_mock.call_args.args[0]
        self.assertNotIn("api_key=", req.full_url.lower())
        self.assertIn("foo=bar", req.full_url)
        self.assertEqual(req.headers.get("Authorization"), "Bearer SECRET_TOKEN")

    def test_idle_watchdog_stop_uses_bearer_header_and_strips_api_key_query_param(self):
        with mock.patch("deploy.runpod_cloud_training.idle_watchdog.urllib.request.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _BytesResponse(b'{"data":{"podStop":true}}')
            ok = _stop_runpod_pod(
                endpoint="https://api.runpod.io/graphql?api_key=SHOULD_NOT_LEAK",
                api_key="SECRET_TOKEN",
                pod_id="pod_123",
                verbose=False,
            )
        self.assertTrue(ok)
        req = urlopen_mock.call_args.args[0]
        self.assertNotIn("api_key=", req.full_url.lower())
        self.assertEqual(req.headers.get("Authorization"), "Bearer SECRET_TOKEN")

    def test_rank_gpu_rows_filters_and_sorts_by_price(self):
        rows = [
            {"id": "g1", "displayName": "GPU A", "memoryInGb": 24, "maxGpuCountSecureCloud": 2, "securePrice": 1.2},
            {"id": "g2", "displayName": "GPU B", "memoryInGb": 16, "maxGpuCountSecureCloud": 4, "securePrice": 0.8},
            {"id": "g3", "displayName": "GPU C", "memoryInGb": 48, "maxGpuCountSecureCloud": 0, "securePrice": 1.0},
        ]
        ranked = _rank_gpu_rows(rows, cloud_type="SECURE", min_memory_gb=20, max_hourly_price=1.5)
        self.assertEqual([r["id"] for r in ranked], ["g1", "g3"])
        self.assertEqual(ranked[0]["price_per_hr"], 1.2)

    def test_choose_template_by_partial_name(self):
        templates = [
            {"id": "t1", "name": "Base CUDA"},
            {"id": "t2", "name": "Chess Bot RunPod"},
        ]
        chosen = _choose_template(templates, template_name="chess bot")
        self.assertEqual(chosen["id"], "t2")


if __name__ == "__main__":
    unittest.main()
