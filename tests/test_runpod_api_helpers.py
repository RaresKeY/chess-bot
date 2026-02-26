import io
import json
import argparse
import urllib.error
import unittest
from unittest import mock

from deploy.runpod_cloud_training.idle_watchdog import _stop_runpod_pod
from scripts.runpod_provision import _choose_template, _gpu_types, _graphql_json, _rank_gpu_rows, build_parser, cmd_provision


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

    def test_gpu_types_403_raises_actionable_message(self):
        err = urllib.error.HTTPError(
            url="https://api.runpod.io/graphql",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"forbidden"}'),
        )
        with mock.patch("scripts.runpod_provision._graphql_json", side_effect=err):
            with self.assertRaises(SystemExit) as ctx:
                _gpu_types("SECRET")
        msg = str(ctx.exception)
        self.assertIn("GraphQL request was denied", msg)
        self.assertIn("provision --gpu-type-id", msg)

    def test_provision_can_fallback_with_explicit_gpu_type_id_when_graphql_denied(self):
        args = argparse.Namespace(
            api_key="SECRET",
            keyring_service="runpod",
            keyring_username="RUNPOD_API_KEY",
            rest_base="https://rest.runpod.io/v1",
            graphql_endpoint="https://api.runpod.io/graphql",
            verbose=False,
            name="test-pod",
            cloud_type="SECURE",
            gpu_count=1,
            gpu_type_id="gpu_explicit_123",
            min_memory_gb=24,
            max_hourly_price=3.0,
            template_id="",
            template_name="chess-bot-training",
            include_runpod_templates=True,
            include_public_templates=True,
            ports=[],
            volume_mount_path="/workspace",
            volume_in_gb=40,
            container_disk_in_gb=15,
            env=[],
            use_runpod_training_preset_env=False,
            support_public_ip_auto=True,
            wait_ready=False,
            wait_timeout_seconds=30,
            wait_poll_seconds=5,
        )
        with mock.patch("scripts.runpod_provision._resolve_api_key", return_value="SECRET"), mock.patch(
            "scripts.runpod_provision._gpu_types",
            side_effect=SystemExit("RunPod GraphQL request was denied (HTTP 403) during gpu-search"),
        ), mock.patch(
            "scripts.runpod_provision._list_templates",
            return_value=[{"id": "tpl1", "name": "chess-bot-training", "imageName": "ghcr.io/x/y:latest"}],
        ), mock.patch(
            "scripts.runpod_provision._create_pod",
            return_value={"id": "pod_123"},
        ) as create_mock, mock.patch(
            "scripts.runpod_provision._print_json"
        ):
            rc = cmd_provision(args)

        self.assertEqual(rc, 0)
        self.assertEqual(create_mock.call_args.kwargs["gpu_type_ids"], ["gpu_explicit_123"])

    def test_provision_preset_env_injection_is_opt_in_by_default(self):
        parser = build_parser()
        args = parser.parse_args(["provision"])
        self.assertFalse(args.use_runpod_training_preset_env)


if __name__ == "__main__":
    unittest.main()
