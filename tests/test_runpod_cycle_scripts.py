import unittest
from pathlib import Path
import subprocess
import tempfile
import os
import json


class RunpodCycleScriptTests(unittest.TestCase):
    def _write_fake_curl(self, fakebin: str) -> None:
        Path(fakebin).mkdir(parents=True, exist_ok=True)
        curl_path = Path(fakebin) / "curl"
        curl_path.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
out_file=""
while (($#)); do
  case "$1" in
    -o)
      out_file="${2:-}"
      shift 2
      ;;
    -w)
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -n "${out_file}" ]]; then
  printf '%s' "${FAKE_CURL_BODY:-{\\"error\\":\\"pod not found to terminate\\",\\"status\\":404}}" > "${out_file}"
fi
printf '%s' "${FAKE_CURL_HTTP_CODE:-404}"
""",
            encoding="utf-8",
        )
        os.chmod(curl_path, 0o755)

    def _write_fake_ssh_for_watcher(self, fakebin: str) -> None:
        Path(fakebin).mkdir(parents=True, exist_ok=True)
        ssh_path = Path(fakebin) / "ssh"
        ssh_path.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
cat <<'OUT'
__RUNPOD_PROGRESS_BLOCK_BEGIN__
{"event":"script_start","epochs_requested":2}
__RUNPOD_PROGRESS_SPLIT__
__RUNPOD_PROGRESS_BLOCK_END__
__RUNPOD_PROGRESS_BLOCK_BEGIN__
{"event":"epoch_start","epoch":1,"epochs":2}
{"event":"epoch_end","epoch":1,"epochs":2,"metrics":{"train_loss":1.0,"val_loss":0.5,"top1":0.2}}
__RUNPOD_PROGRESS_SPLIT__
__RUNPOD_PROGRESS_BLOCK_END__
__RUNPOD_PROGRESS_BLOCK_BEGIN__
{"event":"script_complete","epochs_completed":1}
__RUNPOD_PROGRESS_SPLIT__
0
__RUNPOD_PROGRESS_BLOCK_END__
OUT
""",
            encoding="utf-8",
        )
        os.chmod(ssh_path, 0o755)

    def _write_fake_ssh_for_watcher_stdout_fallback(self, fakebin: str) -> None:
        Path(fakebin).mkdir(parents=True, exist_ok=True)
        ssh_path = Path(fakebin) / "ssh"
        ssh_path.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
cat <<'OUT'
__RUNPOD_PROGRESS_BLOCK_BEGIN__
__RUNPOD_PROGRESS_SPLIT__
__RUNPOD_PROGRESS_LOG_SPLIT__
{'train_setup': {'epochs': 100}}
[train] epoch 1/100 start
{'epoch': 1, 'train_loss': 8.1, 'val_loss': 7.5, 'top1': 0.11}
__RUNPOD_PROGRESS_BLOCK_END__
__RUNPOD_PROGRESS_BLOCK_BEGIN__
__RUNPOD_PROGRESS_SPLIT__
0
__RUNPOD_PROGRESS_LOG_SPLIT__
[train] epoch 2/100 start
{'epoch': 2, 'train_loss': 7.8, 'val_loss': 7.2, 'top1': 0.13}
__RUNPOD_PROGRESS_BLOCK_END__
OUT
""",
            encoding="utf-8",
        )
        os.chmod(ssh_path, 0o755)

    def _write_registry(self, path: Path, *, pod_id: str = "pod123", state: str = "RUNNING") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts_utc": "2026-02-26T10:00:00Z",
            "source_script": "test",
            "action": "start",
            "state": state,
            "pod_id": pod_id,
            "run_id": "run-1",
            "pod_name": "pod-name-1",
        }
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    def _latest_registry_event(self, path: Path) -> dict:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return lines[-1]

    def test_cycle_scripts_exist(self):
        names = [
            "scripts/runpod_cycle_common.sh",
            "scripts/runpod_cycle_start.sh",
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_local_validate.sh",
            "scripts/runpod_cycle_stop.sh",
            "scripts/runpod_cycle_full_smoke.sh",
            "scripts/runpod_full_train_easy.sh",
        ]
        for name in names:
            self.assertTrue(Path(name).is_file(), name)

    def test_full_smoke_calls_modular_steps(self):
        text = Path("scripts/runpod_cycle_full_smoke.sh").read_text(encoding="utf-8")
        for name in [
            "runpod_cycle_start.sh",
            "runpod_cycle_push_dataset.sh",
            "runpod_cycle_train.sh",
            "runpod_cycle_collect.sh",
            "runpod_cycle_local_validate.sh",
            "runpod_cycle_stop.sh",
        ]:
            self.assertIn(name, text)

    def test_stop_script_uses_graphql_pod_stop(self):
        text = Path("scripts/runpod_cycle_stop.sh").read_text(encoding="utf-8")
        self.assertIn("mutation StopPod", text)
        self.assertIn("podStop", text)
        self.assertIn("runpod_cycle_api_token", text)

    def test_common_api_token_prefers_env_without_keyring_call(self):
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "source scripts/runpod_cycle_common.sh && "
                "export RUNPOD_API_KEY='env-token-123' && "
                "out=\"$(runpod_cycle_api_token /definitely/not/a/python)\" && "
                "printf '%s' \"$out\"",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.stdout, "env-token-123")

    def test_common_api_token_falls_back_to_dotenv(self):
        with tempfile.TemporaryDirectory() as td:
            dotenv = Path(td) / ".env.runpod"
            dotenv.write_text("RUNPOD_API_KEY=dotenv-token-999\n", encoding="utf-8")
            env = os.environ.copy()
            env["RUNPOD_DOTENV_PATH"] = str(dotenv)
            env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "source scripts/runpod_cycle_common.sh && "
                    "unset RUNPOD_API_KEY && "
                    "out=\"$(runpod_cycle_api_token python3 /work)\" && "
                    "printf '%s' \"$out\"",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.stdout, "dotenv-token-999")

    def test_cycle_ssh_scripts_use_batch_mode_and_connect_timeout(self):
        for name in [
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_watch_progress.sh",
            "scripts/runpod_cycle_full_train_hf.sh",
        ]:
            text = Path(name).read_text(encoding="utf-8")
            self.assertIn("BatchMode=yes", text, name)
            self.assertIn("ConnectTimeout", text, name)
            self.assertIn("AddKeysToAgent=no", text, name)
            self.assertIn("IdentityAgent=none", text, name)

    def test_gateway_route_automation_removed(self):
        common = Path("scripts/runpod_cycle_common.sh").read_text(encoding="utf-8")
        self.assertNotIn("runpod_cycle_apply_gateway_overrides()", common)
        self.assertNotIn("runpod_cycle_apply_saved_gateway_route()", common)
        self.assertNotIn("RUNPOD_SSH_GATEWAY_ROUTE", common)
        self.assertNotIn("config/runpod_gateway_route.txt", common)
        for name in [
            "scripts/runpod_cycle_start.sh",
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_watch_progress.sh",
            "scripts/runpod_cycle_full_train_hf.sh",
            "scripts/runpod_full_train_easy.sh",
        ]:
            text = Path(name).read_text(encoding="utf-8")
            self.assertNotIn("runpod_cycle_apply_saved_gateway_route", text, name)
            self.assertNotIn("runpod_cycle_apply_gateway_overrides", text, name)

    def test_watch_progress_supports_manual_force_tty_and_stream_parser(self):
        text = Path("scripts/runpod_cycle_watch_progress.sh").read_text(encoding="utf-8")
        self.assertIn('SSH_FORCE_TTY="${RUNPOD_SSH_FORCE_TTY:-0}"', text)
        self.assertIn('SSH_TTY_ARGS=(-tt)', text)
        self.assertIn('if [[ "${SSH_FORCE_TTY}" == "1" ]]; then', text)
        self.assertIn("BLOCK_BEGIN='__RUNPOD_PROGRESS_BLOCK_BEGIN__'", text)
        self.assertIn('while IFS= read -r snapshot_line || [[ -n "${snapshot_line}" ]]; do', text)
        self.assertIn('done < <(', text)
        self.assertIn("tail -n 200", text)

    def test_full_train_and_watcher_have_interrupt_cleanup(self):
        full = Path("scripts/runpod_cycle_full_train_hf.sh").read_text(encoding="utf-8")
        self.assertIn("handle_interrupt()", full)
        self.assertIn("trap handle_interrupt INT TERM", full)
        self.assertIn("cleanup_child_processes()", full)
        self.assertIn("jobs -pr", full)

        watch = Path("scripts/runpod_cycle_watch_progress.sh").read_text(encoding="utf-8")
        self.assertIn("TTY_STATE_ORIG", watch)
        self.assertIn("restore_tty()", watch)
        self.assertIn("trap handle_interrupt INT TERM", watch)
        self.assertIn(r"\033[K", watch)

    def test_easy_full_train_wrapper_has_opinionated_defaults_and_calls_full_flow(self):
        text = Path("scripts/runpod_full_train_easy.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_HF_DATASET_REPO_ID", text)
        self.assertIn("RUNPOD_FULL_TRAIN_EPOCHS", text)
        self.assertIn("RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 5090", text)
        self.assertIn("runpod_cycle_prepare_ssh_client_files", text)
        self.assertIn("temp_ssh_key=$(runpod_cycle_ssh_key)", text)
        self.assertIn("batch_size_override=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>}", text)
        self.assertIn("num_workers_override=${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-<unset>}", text)
        self.assertIn("max_total_rows=${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS:-<unset>}", text)
        self.assertIn("scripts/runpod_cycle_full_train_hf.sh", text)

    def test_cycle_start_uses_managed_ssh_key_toggle_only(self):
        text = Path("scripts/runpod_cycle_start.sh").read_text(encoding="utf-8")
        self.assertIn('RUNPOD_INJECT_MANAGED_SSH_KEY_ENV', text)
        self.assertNotIn('RUNPOD_INJECT_LOCAL_SSH_KEY_ENV', text)

    def test_cycle_scripts_use_config_known_hosts_and_safer_default_host_key_checking(self):
        common = Path("scripts/runpod_cycle_common.sh").read_text(encoding="utf-8")
        self.assertIn('RUNPOD_SSH_HOST_KEY_CHECKING:-accept-new', common)
        self.assertIn('config/runpod_known_hosts', common)
        for name in [
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_watch_progress.sh",
            "scripts/runpod_cycle_full_train_hf.sh",
        ]:
            text = Path(name).read_text(encoding="utf-8")
            self.assertIn("runpod_cycle_prepare_ssh_client_files", text, name)
            self.assertIn("runpod_cycle_ssh_host_key_checking", text, name)
            self.assertIn("runpod_cycle_ssh_known_hosts_file", text, name)

    def test_managed_temp_key_is_forced_no_passphrase_on_reuse(self):
        text = Path("scripts/runpod_cycle_common.sh").read_text(encoding="utf-8")
        self.assertIn('managed_default_key="${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}"', text)
        self.assertIn('ssh-keygen -y -P "" -f "${key_path}"', text)
        self.assertIn("requires passphrase; regenerating no-passphrase key", text)

    def test_full_train_hf_context_probe_uses_quoted_heredoc(self):
        text = Path("scripts/runpod_cycle_full_train_hf.sh").read_text(encoding="utf-8")
        self.assertIn("<<'EOF' 2>&1 | tee \"${REMOTE_CONTEXT_LOG}\"", text)
        self.assertIn("CONTEXT_JSON='${REMOTE_CONTEXT_JSON}'", text)
        self.assertIn("'/opt/venvs/chessbot/bin/python' - <<'PY'", text)
        self.assertIn('rm -f "${REMOTE_PROGRESS_JSONL}" "${REMOTE_GPU_SAMPLES_CSV}" "${REMOTE_TRAIN_PID_FILE}" "${REMOTE_TRAIN_LOG}"', text)
        self.assertIn("override_batch_size=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>}", text)
        self.assertIn('cpu_threads="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"', text)
        self.assertIn('TRAIN_NUM_WORKERS="${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-${auto_num_workers}}"', text)
        self.assertIn('export TRAIN_EXTRA_ARGS="--epochs ${FLOW_EPOCHS} --early-stopping-patience 0"', text)
        self.assertNotIn("--no-progress", text)
        self.assertIn('if [[ "${gpu_name_for_batch}" == *"RTX 5090"* ]]; then', text)
        self.assertIn("batch_attempts=(4096 3072 2048)", text)
        self.assertIn("is_oom_failure_tail()", text)
        self.assertIn('FLOW_MAX_TOTAL_ROWS="${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS:-0}"', text)
        self.assertIn('export TRAIN_MAX_TOTAL_ROWS="${FLOW_MAX_TOTAL_ROWS}"', text)
        self.assertIn('RUNPOD_REMOTE_BEST_CHECKPOINT="${REMOTE_BEST_CHECKPOINT}"', text)
        self.assertIn("cpu_threads=${cpu_threads} auto_num_workers=${auto_num_workers}", text)

    def test_watch_progress_syncs_remote_checkpoints_and_writes_epoch_eta_report(self):
        text = Path("scripts/runpod_cycle_watch_progress.sh").read_text(encoding="utf-8")
        self.assertIn('REMOTE_BEST_CHECKPOINT="${RUNPOD_REMOTE_BEST_CHECKPOINT:-${REMOTE_RUN_DIR}/model_best_${RUN_ID}.pt}"', text)
        self.assertIn('REMOTE_EPOCH_CHECKPOINT_DIR="${RUNPOD_REMOTE_EPOCH_CHECKPOINT_DIR:-${REMOTE_RUN_DIR}/epoch_checkpoints}"', text)
        self.assertIn("EPOCH_ETA_REPORT_JSONL", text)
        self.assertIn("sync_epoch_artifacts()", text)
        self.assertIn("append_epoch_eta_report()", text)
        self.assertIn("runpod_cycle_require_cmd scp", text)

    def test_full_train_hf_remote_fetch_uses_unified_args_builder(self):
        text = Path("scripts/runpod_cycle_full_train_hf.sh").read_text(encoding="utf-8")
        self.assertIn("fetch_args=(", text)
        self.assertIn("fetch_args+=( --all-latest )", text)
        self.assertIn("fetch_args+=( --dataset-name", text)
        self.assertIn("hf_dataset_fetch.py", text)
        self.assertIn("\\${fetch_args[@]}", text)

    def test_terminate_all_reconciles_pod_not_found_404_as_terminated(self):
        with tempfile.TemporaryDirectory() as td:
            fakebin = Path(td) / "bin"
            self._write_fake_curl(str(fakebin))
            registry = Path(td) / "tracked.jsonl"
            self._write_registry(registry, pod_id="gonepod", state="RUNNING")

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}:{env['PATH']}"
            env["RUNPOD_TRACKED_PODS_FILE"] = str(registry)
            env["RUNPOD_CONFIRM_TERMINATE_ALL"] = "YES"
            env["RUNPOD_API_KEY"] = "test-token"
            env["FAKE_CURL_HTTP_CODE"] = "404"
            env["FAKE_CURL_BODY"] = '{"error":"pod not found to terminate","status":404}'

            proc = subprocess.run(
                ["bash", "scripts/runpod_cycle_terminate_all_tracked.sh", "--yes"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("reconciled_already_gone=1", proc.stdout)
            latest = self._latest_registry_event(registry)
            self.assertEqual(latest["state"], "TERMINATED")
            self.assertEqual(latest["action"], "terminate_reconcile")

    def test_terminate_all_keeps_non_matching_404_as_failure(self):
        with tempfile.TemporaryDirectory() as td:
            fakebin = Path(td) / "bin"
            self._write_fake_curl(str(fakebin))
            registry = Path(td) / "tracked.jsonl"
            self._write_registry(registry, pod_id="bad404", state="RUNNING")

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}:{env['PATH']}"
            env["RUNPOD_TRACKED_PODS_FILE"] = str(registry)
            env["RUNPOD_CONFIRM_TERMINATE_ALL"] = "YES"
            env["RUNPOD_API_KEY"] = "test-token"
            env["FAKE_CURL_HTTP_CODE"] = "404"
            env["FAKE_CURL_BODY"] = '{"error":"template not found","status":404}'

            proc = subprocess.run(
                ["bash", "scripts/runpod_cycle_terminate_all_tracked.sh", "--yes"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertNotEqual(proc.returncode, 0)
            latest = self._latest_registry_event(registry)
            self.assertNotEqual(latest["state"], "TERMINATED")

    def test_watch_progress_parses_single_ssh_stream_and_exits_with_remote_code(self):
        with tempfile.TemporaryDirectory() as td:
            fakebin = Path(td) / "bin"
            self._write_fake_ssh_for_watcher(str(fakebin))
            repo_root = Path(td) / "repo"
            (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
            # Script sources common.sh from scripts/, so expose repo scripts via symlink-like copy paths not needed;
            # invoke actual script from project and point outputs/provision json into temp.
            provision = Path(td) / "provision.json"
            provision.write_text('{"pod_id":"p1","pod_status":{"publicIp":"127.0.0.1","portMappings":{"22":22},"env":{"REPO_DIR":"/remote/repo"}}}\n', encoding="utf-8")
            logs_dir = Path(td) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            known_hosts = Path(td) / "known_hosts"

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}:{env['PATH']}"
            env["RUNPOD_POD_JSON"] = str(provision)
            env["RUNPOD_CYCLE_LOGS_DIR"] = str(logs_dir)
            env["RUNPOD_SSH_HOST"] = "127.0.0.1"
            env["RUNPOD_SSH_PORT"] = "22"
            env["RUNPOD_SSH_USER"] = "runner"
            env["RUNPOD_SSH_KEY"] = str(Path(td) / "dummy_key")
            env["RUNPOD_SSH_KNOWN_HOSTS_FILE"] = str(known_hosts)
            env["RUNPOD_SSH_HOST_KEY_CHECKING"] = "no"
            env["RUNPOD_PROGRESS_POLL_SECONDS"] = "1"
            env["RUNPOD_CYCLE_RUN_ID"] = "test-run"

            proc = subprocess.run(
                ["bash", "scripts/runpod_cycle_watch_progress.sh"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("remote_train_exit_code=0", proc.stdout)
            watch_log = logs_dir / "train_progress_watch.log"
            self.assertTrue(watch_log.is_file())
            watch_text = watch_log.read_text(encoding="utf-8")
            self.assertIn('"event":"script_start"', watch_text)
            self.assertIn('"event":"epoch_end"', watch_text)

    def test_watch_progress_falls_back_to_stdout_epoch_log_when_jsonl_missing(self):
        with tempfile.TemporaryDirectory() as td:
            fakebin = Path(td) / "bin"
            self._write_fake_ssh_for_watcher_stdout_fallback(str(fakebin))
            provision = Path(td) / "provision.json"
            provision.write_text(
                '{"pod_id":"p1","pod_status":{"publicIp":"127.0.0.1","portMappings":{"22":22},"env":{"REPO_DIR":"/remote/repo"}}}\n',
                encoding="utf-8",
            )
            logs_dir = Path(td) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            known_hosts = Path(td) / "known_hosts"

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}:{env['PATH']}"
            env["RUNPOD_POD_JSON"] = str(provision)
            env["RUNPOD_CYCLE_LOGS_DIR"] = str(logs_dir)
            env["RUNPOD_SSH_HOST"] = "127.0.0.1"
            env["RUNPOD_SSH_PORT"] = "22"
            env["RUNPOD_SSH_USER"] = "runner"
            env["RUNPOD_SSH_KEY"] = str(Path(td) / "dummy_key")
            env["RUNPOD_SSH_KNOWN_HOSTS_FILE"] = str(known_hosts)
            env["RUNPOD_SSH_HOST_KEY_CHECKING"] = "no"
            env["RUNPOD_PROGRESS_POLL_SECONDS"] = "1"
            env["RUNPOD_CYCLE_RUN_ID"] = "test-run"

            proc = subprocess.run(
                ["bash", "scripts/runpod_cycle_watch_progress.sh"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("using stdout fallback progress parser", proc.stderr)
            self.assertIn("src=stdout", proc.stdout)
            self.assertIn("epoch=2/100", proc.stdout)


if __name__ == "__main__":
    unittest.main()
