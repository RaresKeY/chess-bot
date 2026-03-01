import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.chessbot.runpod_sdk_component import (
    _choose_template,
    _extract_pod_state,
    _first_callable,
    _invoke_with_id,
    _invoke_with_payload,
    _load_runpod_sdk,
    _normalize_rows,
    _resolve_api_key,
    _rank_gpu_rows,
    build_parser,
    cmd_gpu_search,
    cmd_pod_status,
    cmd_pod_stop,
    cmd_pod_terminate,
    cmd_provision,
    cmd_template_list,
)


class _FakeRunpod:
    def __init__(self):
        self.api_key = ""
        self.created_payload = None

    def get_gpu_types(self):
        return {
            "gpus": [
                {
                    "id": "NVIDIA RTX A5000",
                    "displayName": "RTX A5000",
                    "memoryInGb": 24,
                    "communityPrice": 0.16,
                    "maxGpuCount": 8,
                }
            ]
        }

    def get_templates(self):
        return {
            "templates": [
                {"id": "tpl-1", "name": "chess-bot-training", "imageName": "ghcr.io/example/chess-bot:latest"}
            ]
        }

    def create_pod(self, payload=None, **kwargs):
        self.created_payload = payload or kwargs
        return {"id": "pod-123"}

    def get_pod(self, pod_id):
        return {"id": pod_id, "desiredStatus": "RUNNING"}

    def stop_pod(self, pod_id):
        return {"id": pod_id, "desiredStatus": "EXITED"}

    def terminate_pod(self, pod_id):
        return {"id": pod_id, "terminated": True}


class RunpodSdkComponentTests(unittest.TestCase):
    def test_first_callable_supports_nested_paths(self):
        class Obj:
            class A:
                @staticmethod
                def fn():
                    return 1

            a = A()

        fn = _first_callable(Obj(), ("missing", "a.fn"))
        self.assertIsNotNone(fn)
        self.assertEqual(fn(), 1)

    def test_rank_gpu_rows_filters_and_sorts(self):
        rows = [
            {"id": "g1", "displayName": "GPU1", "memoryInGb": 24, "communityPrice": 0.5, "maxGpuCount": 1},
            {"id": "g2", "displayName": "GPU2", "memoryInGb": 16, "communityPrice": 0.2, "maxGpuCount": 1},
            {"id": "g3", "displayName": "GPU3", "memoryInGb": 48, "communityPrice": 0.3, "maxGpuCount": 1},
        ]
        ranked = _rank_gpu_rows(rows, cloud_type="COMMUNITY", min_memory_gb=20, max_hourly_price=0.4)
        self.assertEqual([r["id"] for r in ranked], ["g3"])

    def test_choose_template_partial_name(self):
        templates = [{"id": "t1", "name": "Base"}, {"id": "t2", "name": "Chess Bot Training"}]
        chosen = _choose_template(templates, template_name="chess bot")
        self.assertEqual(chosen["id"], "t2")

    def test_resolve_api_key_reads_dotenv_fallback(self):
        args = argparse.Namespace(api_key="", keyring_service="runpod", keyring_username="RUNPOD_API_KEY")
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("src.chessbot.secrets.token_from_keyring", return_value=""):
                with tempfile.TemporaryDirectory() as td:
                    dotenv = Path(td) / ".env.runpod"
                    dotenv.write_text("RUNPOD_API_KEY=dotenv-token\n", encoding="utf-8")
                    with mock.patch("src.chessbot.runpod_sdk_component.default_dotenv_paths", return_value=[dotenv]):
                        token = _resolve_api_key(args)
        self.assertEqual(token, "dotenv-token")

    def test_resolve_api_key_precedence_explicit_over_env(self):
        args = argparse.Namespace(api_key="explicit", keyring_service="runpod", keyring_username="RUNPOD_API_KEY")
        with mock.patch.dict("os.environ", {"RUNPOD_API_KEY": "env-token"}, clear=True):
            token = _resolve_api_key(args)
        self.assertEqual(token, "explicit")

    def test_parser_defaults_keep_component_independent(self):
        parser = build_parser()
        args = parser.parse_args(["provision"])
        self.assertEqual(args.template_name, "chess-bot-training")
        self.assertTrue(args.wait_ready)
        self.assertFalse(args.interruptible)

    def test_parser_bool_overrides_for_template_list(self):
        parser = build_parser()
        args = parser.parse_args(["template-list", "--no-pods-only", "--include-serverless"])
        self.assertFalse(args.pods_only)
        self.assertTrue(args.include_serverless)

    def test_invoke_with_payload_falls_back_to_keyword_style(self):
        calls = []

        def fn(*args, **kwargs):
            calls.append((args, kwargs))
            if args or (set(kwargs.keys()) != {"payload"}):
                raise TypeError("wrong signature")
            return {"ok": True, "payload": kwargs["payload"]}

        out = _invoke_with_payload(fn, {"x": 1})
        self.assertEqual(out["payload"]["x"], 1)
        self.assertGreaterEqual(len(calls), 2)

    def test_invoke_with_id_falls_back_to_named_arg(self):
        calls = []

        def fn(*args, **kwargs):
            calls.append((args, kwargs))
            if args:
                raise TypeError("positional not supported")
            if "pod_id" in kwargs:
                return {"id": kwargs["pod_id"]}
            raise TypeError("wrong named arg")

        out = _invoke_with_id(fn, "pod-1")
        self.assertEqual(out["id"], "pod-1")
        self.assertGreaterEqual(len(calls), 2)

    def test_normalize_rows_supports_common_keys(self):
        rows = _normalize_rows({"items": [{"a": 1}, "bad"]}, ("gpus", "items"))
        self.assertEqual(rows, [{"a": 1}])
        rows_data = _normalize_rows({"data": [{"b": 2}]}, ("items",))
        self.assertEqual(rows_data, [{"b": 2}])

    def test_load_runpod_sdk_missing_package_is_actionable(self):
        with mock.patch("builtins.__import__", side_effect=ImportError("missing")):
            with self.assertRaises(SystemExit) as ctx:
                _load_runpod_sdk()
        self.assertIn("requires the `runpod` package", str(ctx.exception))

    def test_extract_pod_state_prefers_top_level_then_nested(self):
        self.assertEqual(_extract_pod_state({"desiredStatus": "RUNNING"}), "RUNNING")
        self.assertEqual(_extract_pod_state({"pod": {"state": "EXITED"}}), "EXITED")

    def test_provision_uses_sdk_component_without_raw_http(self):
        fake = _FakeRunpod()
        args = argparse.Namespace(
            api_key="SECRET",
            keyring_service="runpod",
            keyring_username="RUNPOD_API_KEY",
            name="sdk-test",
            cloud_type="COMMUNITY",
            gpu_count=1,
            gpu_type_id="",
            min_memory_gb=24,
            max_hourly_price=1.0,
            template_id="",
            template_name="chess-bot-training",
            include_runpod_templates=True,
            include_public_templates=True,
            ports=["22/tcp"],
            volume_mount_path="/workspace",
            volume_in_gb=40,
            container_disk_in_gb=15,
            env=["A=1"],
            interruptible=False,
            support_public_ip=True,
            wait_ready=True,
            wait_timeout_seconds=3,
            wait_poll_seconds=1,
        )

        with mock.patch("src.chessbot.runpod_sdk_component._resolve_api_key", return_value="SECRET"), mock.patch(
            "src.chessbot.runpod_sdk_component._load_runpod_sdk", return_value=fake
        ), mock.patch("src.chessbot.runpod_sdk_component._print_json") as print_mock:
            rc = cmd_provision(args)

        self.assertEqual(rc, 0)
        self.assertEqual(fake.api_key, "SECRET")
        self.assertEqual(fake.created_payload["gpuTypeIds"], ["NVIDIA RTX A5000"])
        self.assertEqual(fake.created_payload["env"]["A"], "1")
        out = print_mock.call_args.args[0]
        self.assertEqual(out["component"], "runpod_sdk")

    def test_provision_rejects_invalid_env_item(self):
        fake = _FakeRunpod()
        args = argparse.Namespace(
            api_key="SECRET",
            keyring_service="runpod",
            keyring_username="RUNPOD_API_KEY",
            name="sdk-test",
            cloud_type="COMMUNITY",
            gpu_count=1,
            gpu_type_id="NVIDIA RTX A5000",
            min_memory_gb=24,
            max_hourly_price=1.0,
            template_id="",
            template_name="chess-bot-training",
            include_runpod_templates=True,
            include_public_templates=True,
            ports=["22/tcp"],
            volume_mount_path="/workspace",
            volume_in_gb=40,
            container_disk_in_gb=15,
            env=["BAD_ENV"],
            interruptible=False,
            support_public_ip=True,
            wait_ready=False,
            wait_timeout_seconds=1,
            wait_poll_seconds=1,
        )
        with mock.patch("src.chessbot.runpod_sdk_component._resolve_api_key", return_value="SECRET"), mock.patch(
            "src.chessbot.runpod_sdk_component._load_runpod_sdk", return_value=fake
        ):
            with self.assertRaises(SystemExit) as ctx:
                cmd_provision(args)
        self.assertIn("Invalid --env item", str(ctx.exception))

    def test_cmd_gpu_search_outputs_component_tag_and_limit(self):
        fake = _FakeRunpod()
        args = argparse.Namespace(
            api_key="SECRET",
            keyring_service="runpod",
            keyring_username="RUNPOD_API_KEY",
            cloud_type="COMMUNITY",
            min_memory_gb=1,
            max_hourly_price=1.0,
            limit=1,
        )
        with mock.patch("src.chessbot.runpod_sdk_component._resolve_api_key", return_value="SECRET"), mock.patch(
            "src.chessbot.runpod_sdk_component._load_runpod_sdk", return_value=fake
        ), mock.patch("src.chessbot.runpod_sdk_component._print_json") as print_mock:
            rc = cmd_gpu_search(args)
        self.assertEqual(rc, 0)
        out = print_mock.call_args.args[0]
        self.assertEqual(out["component"], "runpod_sdk")
        self.assertEqual(out["count"], 1)

    def test_cmd_template_list_filters_serverless_and_name(self):
        fake = _FakeRunpod()
        with mock.patch.object(
            fake,
            "get_templates",
            return_value={
                "templates": [
                    {"id": "t1", "name": "Chess Bot Training", "templateType": "pod"},
                    {"id": "t2", "name": "Serverless Job", "templateType": "serverless"},
                ]
            },
        ):
            args = argparse.Namespace(
                api_key="SECRET",
                keyring_service="runpod",
                keyring_username="RUNPOD_API_KEY",
                include_runpod_templates=True,
                include_public_templates=True,
                include_serverless=True,
                pods_only=True,
                template_name="chess",
            )
            with mock.patch("src.chessbot.runpod_sdk_component._resolve_api_key", return_value="SECRET"), mock.patch(
                "src.chessbot.runpod_sdk_component._load_runpod_sdk", return_value=fake
            ), mock.patch("src.chessbot.runpod_sdk_component._print_json") as print_mock:
                rc = cmd_template_list(args)
        self.assertEqual(rc, 0)
        out = print_mock.call_args.args[0]
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["templates"][0]["id"], "t1")

    def test_cmd_pod_status_stop_terminate_emit_component_tag(self):
        fake = _FakeRunpod()
        args = argparse.Namespace(api_key="SECRET", keyring_service="runpod", keyring_username="RUNPOD_API_KEY", pod_id="pod-1")
        with mock.patch("src.chessbot.runpod_sdk_component._resolve_api_key", return_value="SECRET"), mock.patch(
            "src.chessbot.runpod_sdk_component._load_runpod_sdk", return_value=fake
        ), mock.patch("src.chessbot.runpod_sdk_component._print_json") as print_mock:
            self.assertEqual(cmd_pod_status(args), 0)
            self.assertEqual(print_mock.call_args.args[0]["component"], "runpod_sdk")
            self.assertEqual(cmd_pod_stop(args), 0)
            self.assertEqual(print_mock.call_args.args[0]["component"], "runpod_sdk")
            with mock.patch.object(fake, "terminate_pod", return_value={"ok": True}, create=True):
                self.assertEqual(cmd_pod_terminate(args), 0)
                self.assertEqual(print_mock.call_args.args[0]["component"], "runpod_sdk")


if __name__ == "__main__":
    unittest.main()
