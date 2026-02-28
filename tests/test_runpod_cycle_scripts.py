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
            "scripts/runpod_cycle_report_style.py",
            "scripts/runpod_cycle_status.sh",
            "scripts/runpod_cycle_start.sh",
            "scripts/runpod_cycle_watchdog.sh",
            "scripts/cloud_connectivity_health_checks.sh",
            "scripts/runpod_connectivity_telemetry_checks.sh",
            "scripts/telemetry_control.sh",
            "scripts/telemetry_emit_event.sh",
            "scripts/telemetry_checkpoint.sh",
            "scripts/telemetry_healthchecks_ping.sh",
            "scripts/telemetry_status.sh",
            "scripts/telemetry_watchdog.sh",
            "scripts/runpod_cycle_push_dataset.sh",
            "scripts/runpod_cycle_train.sh",
            "scripts/runpod_cycle_collect.sh",
            "scripts/runpod_cycle_local_validate.sh",
            "scripts/runpod_cycle_stop.sh",
            "scripts/runpod_cycle_benchmark_matrix.sh",
            "scripts/runpod_cycle_benchmark_10k_sixpack.sh",
            "scripts/runpod_file_transfer.sh",
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
        self.assertIn('telemetry_checkpoint "full_smoke_flow" "running"', text)
        self.assertIn('telemetry_checkpoint "full_smoke_flow" "done"', text)
        self.assertIn('telemetry_event "full_smoke_flow_complete" "ok"', text)

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
            "scripts/runpod_cycle_status.sh",
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

    def test_cycle_collect_auto_log_bundle_outputs(self):
        text = Path("scripts/runpod_cycle_collect.sh").read_text(encoding="utf-8")
        self.assertIn('LOCAL_AUTO_LOGS_DIR="${LOCAL_COLLECT_DIR}/logs_auto"', text)
        self.assertIn("remote_state_snapshot.txt", text)
        self.assertIn("train_log_indexing_summary.json", text)
        self.assertIn("collection_manifest.json", text)
        self.assertIn("indexing_detected", text)

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
        self.assertIn("RUNPOD_GPU_COUNT:-2", text)
        self.assertIn("RUNPOD_FULL_TRAIN_NPROC_PER_NODE", text)
        self.assertIn("runpod_cycle_prepare_ssh_client_files", text)
        self.assertIn("temp_ssh_key=$(runpod_cycle_ssh_key)", text)
        self.assertIn("gpu_count=${RUNPOD_GPU_COUNT}", text)
        self.assertIn("train_nproc_per_node=${RUNPOD_FULL_TRAIN_NPROC_PER_NODE}", text)
        self.assertIn("batch_size_override=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>}", text)
        self.assertIn("num_workers_override=${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-<unset>}", text)
        self.assertIn("max_total_rows=${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS:-<unset>}", text)
        self.assertIn("final_report=artifacts/runpod_cycles/<run_id>/reports/easy_progress_report.md", text)
        self.assertIn("scripts/runpod_cycle_full_train_hf.sh", text)

    def test_benchmark_matrix_script_exists_and_covers_precision_trials(self):
        text = Path("scripts/runpod_cycle_benchmark_matrix.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_BENCH_TRIALS", text)
        self.assertIn("fp32,tf32,fp16,bf16,sparsity", text)
        self.assertIn("fp32_sparse", text)
        self.assertIn("fp16_sparse", text)
        self.assertIn("bf16_sparse", text)
        self.assertIn("RUNPOD_GPU_TYPE_ID:-NVIDIA A40", text)
        self.assertIn("RUNPOD_GPU_COUNT:-2", text)
        self.assertIn("TRAIN_NPROC_PER_NODE", text)
        self.assertIn("--amp-dtype bf16", text)
        self.assertIn("--no-amp --tf32 off", text)
        self.assertIn("runpod_cycle_collect.sh", text)
        self.assertIn("RUNPOD_BENCH_TERMINATE_POD", text)
        self.assertIn("runpod_cycle_terminate.sh", text)
        self.assertIn("RUNPOD_BENCH_HF_DATASET_NAME", text)
        self.assertIn("hf_dataset_fetch.py", text)
        self.assertIn("FLOW_EXPECTED_GIT_SHA", text)
        self.assertIn("git pull --ff-only origin main", text)
        self.assertIn("TRAIN_MAX_TOTAL_ROWS", text)
        self.assertIn("RUNPOD_BENCH_DISTRIBUTED_BACKEND", text)
        self.assertIn("--distributed-backend", text)
        self.assertIn("RUNPOD_BENCH_TRANSFER_TOOL", text)
        self.assertIn("RUNPOD_BENCH_TRANSFER_STRICT", text)
        self.assertIn("rclone copy", text)
        self.assertIn(":sftp,host=${SSH_HOST}", text)
        self.assertIn("rclone_missing_fallback_rsync", text)
        self.assertIn("benchmark_image_used", text)
        self.assertIn("dependency_check.json", text)
        self.assertIn("benchmark_dependencies", text)
        self.assertIn("requirements.txt", text)

    def test_benchmark_10k_sixpack_wrapper_defaults(self):
        text = Path("scripts/runpod_cycle_benchmark_10k_sixpack.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_BENCH_EPOCHS", text)
        self.assertIn("RUNPOD_BENCH_MAX_TOTAL_ROWS", text)
        self.assertIn("fp32,fp16,bf16,fp32_sparse,fp16_sparse,bf16_sparse", text)
        self.assertIn("RUNPOD_BENCH_TERMINATE_POD", text)
        self.assertIn("RUNPOD_BENCH_HF_DATASET_NAME", text)
        self.assertIn("scripts/runpod_cycle_benchmark_matrix.sh", text)
        self.assertIn("scripts/telemetry_status.sh", text)

    def test_file_transfer_script_has_retries_and_rsync_hardening(self):
        text = Path("scripts/runpod_file_transfer.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_TRANSFER_RETRIES", text)
        self.assertIn("pull|push|sync", text)
        self.assertIn("--append-verify", text)
        self.assertIn("BatchMode=yes", text)
        self.assertIn("AddKeysToAgent=no", text)
        self.assertIn("IdentityAgent=none", text)

    def test_watchdog_script_supports_stall_actions(self):
        alias_text = Path("scripts/runpod_cycle_watchdog.sh").read_text(encoding="utf-8")
        self.assertIn("telemetry_watchdog.sh", alias_text)
        self.assertIn("exec bash", alias_text)
        text = Path("scripts/telemetry_watchdog.sh").read_text(encoding="utf-8")
        self.assertIn("--on-stall", text)
        self.assertIn("collect-stop", text)
        self.assertIn("collect-terminate", text)
        self.assertIn("scripts/runpod_cycle_status.sh", text)
        self.assertIn("scripts/runpod_cycle_collect.sh", text)
        self.assertIn("scripts/runpod_cycle_stop.sh", text)
        self.assertIn("scripts/runpod_cycle_terminate.sh", text)

    def test_telemetry_control_routes_subcommands(self):
        text = Path("scripts/telemetry_control.sh").read_text(encoding="utf-8")
        self.assertIn("telemetry_status.sh", text)
        self.assertIn("telemetry_emit_event.sh", text)
        self.assertIn("telemetry_checkpoint.sh", text)
        self.assertIn("telemetry_watchdog.sh", text)
        self.assertIn("telemetry_healthchecks_ping.sh", text)

    def test_cycle_start_uses_managed_ssh_key_toggle_only(self):
        text = Path("scripts/runpod_cycle_start.sh").read_text(encoding="utf-8")
        self.assertIn('RUNPOD_INJECT_MANAGED_SSH_KEY_ENV', text)
        self.assertIn('START_OTEL_COLLECTOR=0', text)
        self.assertIn('RUNPOD_REQUIRE_SSH_READY', text)
        self.assertIn('RUNPOD_SSH_READY_TIMEOUT_SECONDS', text)
        self.assertIn('RUNPOD_TERMINATE_ON_SSH_NOT_READY', text)
        self.assertIn('ssh readiness timed out', text)
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

    def test_managed_temp_key_path_only_no_legacy_key_overrides(self):
        text = Path("scripts/runpod_cycle_common.sh").read_text(encoding="utf-8")
        self.assertIn('printf \'%s\\n\' "${RUNPOD_TEMP_SSH_KEY_BASE:-/tmp/chessbot_runpod_temp_id_ed25519}"', text)
        self.assertNotIn("RUNPOD_SSH_KEY", text)
        self.assertNotIn("RUNPOD_SSH_PUBKEY_PATH", text)

    def test_full_train_hf_context_probe_uses_quoted_heredoc(self):
        text = Path("scripts/runpod_cycle_full_train_hf.sh").read_text(encoding="utf-8")
        self.assertIn('FLOW_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"', text)
        self.assertIn('FLOW_TRAIN_NPROC_PER_NODE="${RUNPOD_FULL_TRAIN_NPROC_PER_NODE:-${FLOW_GPU_COUNT}}"', text)
        self.assertIn("<<'EOF' 2>&1 | tee \"${REMOTE_CONTEXT_LOG}\"", text)
        self.assertIn("CONTEXT_JSON='${REMOTE_CONTEXT_JSON}'", text)
        self.assertIn("'/opt/venvs/chessbot/bin/python' - <<'PY'", text)
        self.assertIn('rm -f "${REMOTE_PROGRESS_JSONL}" "${REMOTE_GPU_SAMPLES_CSV}" "${REMOTE_TRAIN_PID_FILE}" "${REMOTE_TRAIN_LOG}"', text)
        self.assertIn("override_batch_size=${RUNPOD_FULL_TRAIN_BATCH_SIZE_OVERRIDE:-<unset>}", text)
        self.assertIn('cpu_threads="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"', text)
        self.assertIn('suggested_num_workers="${suggested[1]:-6}"', text)
        self.assertIn('cpu_reserve_threads="${RUNPOD_FULL_TRAIN_CPU_RESERVE_THREADS:-0}"', text)
        self.assertIn("cpu_worker_budget_total=$((cpu_threads - TRAIN_NPROC_PER_NODE - cpu_reserve_threads))", text)
        self.assertIn("cpu_based_num_workers_per_rank=$((cpu_worker_budget_total / TRAIN_NPROC_PER_NODE))", text)
        self.assertIn('ddp_suggested_num_workers_per_rank="${suggested_num_workers}"', text)
        self.assertIn('hard_cap_num_workers="${RUNPOD_FULL_TRAIN_NUM_WORKERS_HARD_CAP:-32}"', text)
        self.assertIn("hard_cap_num_workers_per_rank=$((hard_cap_num_workers / TRAIN_NPROC_PER_NODE))", text)
        self.assertIn('TRAIN_NUM_WORKERS="${RUNPOD_FULL_TRAIN_NUM_WORKERS_OVERRIDE:-${auto_num_workers_per_rank}}"', text)
        self.assertIn('TRAIN_NPROC_PER_NODE="${FLOW_TRAIN_NPROC_PER_NODE:-1}"', text)
        self.assertIn('export TRAIN_NPROC_PER_NODE', text)
        self.assertIn('export TRAIN_EXTRA_ARGS="--epochs ${FLOW_EPOCHS} --early-stopping-patience 0"', text)
        self.assertNotIn("--no-progress", text)
        self.assertIn('if [[ "${gpu_name_for_batch}" == *"RTX 5090"* ]]; then', text)

    def test_cycle_train_and_easy_flow_emit_telemetry(self):
        train_text = Path("scripts/runpod_cycle_train.sh").read_text(encoding="utf-8")
        self.assertIn("cycle_train_start", train_text)
        self.assertIn('telemetry_checkpoint "cycle_train" "running"', train_text)
        self.assertIn('telemetry_checkpoint "cycle_train" "done"', train_text)
        easy_text = Path("scripts/runpod_full_train_easy.sh").read_text(encoding="utf-8")
        self.assertIn("full_train_easy_start", easy_text)
        self.assertIn('telemetry_checkpoint.sh" \\', easy_text)
        text = Path("scripts/runpod_cycle_full_train_hf.sh").read_text(encoding="utf-8")
        self.assertIn("batch_attempts=(4096 3072 2048)", text)
        self.assertIn("is_oom_failure_tail()", text)
        self.assertIn('FLOW_MAX_TOTAL_ROWS="${RUNPOD_FULL_TRAIN_MAX_TOTAL_ROWS:-0}"', text)
        self.assertIn('export TRAIN_MAX_TOTAL_ROWS="${FLOW_MAX_TOTAL_ROWS}"', text)
        self.assertIn('RUNPOD_REMOTE_BEST_CHECKPOINT="${REMOTE_BEST_CHECKPOINT}"', text)
        self.assertIn("cpu_threads=${cpu_threads} cpu_reserve_threads=${cpu_reserve_threads} train_nproc_per_node=${TRAIN_NPROC_PER_NODE} cpu_worker_budget_total=${cpu_worker_budget_total} cpu_based_num_workers_per_rank=${cpu_based_num_workers_per_rank} ddp_suggested_num_workers_per_rank=${ddp_suggested_num_workers_per_rank} hard_cap_total_num_workers=${hard_cap_num_workers} hard_cap_num_workers_per_rank=${hard_cap_num_workers_per_rank} auto_num_workers_per_rank=${auto_num_workers_per_rank} vram_suggested_num_workers_per_rank=${suggested_num_workers}", text)

    def test_train_preset_supports_torchrun_nproc(self):
        text = Path("deploy/runpod_cloud_training/train_baseline_preset.sh").read_text(encoding="utf-8")
        self.assertIn('TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-1}"', text)
        self.assertIn('cmd=( "${VENV_DIR}/bin/torchrun" --standalone --nnodes=1 --nproc-per-node "${TRAIN_NPROC_PER_NODE}" "${train_args[@]}" )', text)

    def test_cycle_train_prefers_repo_train_preset_over_image_copy(self):
        text = Path("scripts/runpod_cycle_train.sh").read_text(encoding="utf-8")
        self.assertIn("TRAIN_PRESET_REPO", text)
        self.assertIn("TRAIN_PRESET_IMAGE", text)
        self.assertIn('if [[ -f "${TRAIN_PRESET_REPO}" ]]; then', text)

    def test_watch_progress_syncs_remote_checkpoints_and_writes_epoch_eta_report(self):
        text = Path("scripts/runpod_cycle_watch_progress.sh").read_text(encoding="utf-8")
        self.assertIn('REMOTE_BEST_CHECKPOINT="${RUNPOD_REMOTE_BEST_CHECKPOINT:-${REMOTE_RUN_DIR}/model_best_${RUN_ID}.pt}"', text)
        self.assertIn('REMOTE_EPOCH_CHECKPOINT_DIR="${RUNPOD_REMOTE_EPOCH_CHECKPOINT_DIR:-${REMOTE_RUN_DIR}/epoch_checkpoints}"', text)
        self.assertIn('LOCAL_SYNC_BASE_DIR="${RUNPOD_LOCAL_SYNC_DIR:-${LOCAL_CYCLE_DIR}}"', text)
        self.assertIn('LOCAL_SYNC_RUN_DIR="${LOCAL_SYNC_BASE_DIR}/${RUN_ID}"', text)
        self.assertIn("EPOCH_ETA_REPORT_JSONL", text)
        self.assertIn("sync_epoch_artifacts()", text)
        self.assertIn("append_epoch_eta_report()", text)
        self.assertIn("runpod_cycle_require_cmd scp", text)

    def test_cycle_status_reports_remote_stage_and_supports_watch_mode(self):
        text = Path("scripts/runpod_cycle_status.sh").read_text(encoding="utf-8")
        self.assertIn("collect_once()", text)
        self.assertIn("--watch", text)
        self.assertIn("--auto-collect", text)
        self.assertIn('"remote_state"', text)
        self.assertIn("training_running", text)
        self.assertIn("manual_training_finished", text)
        self.assertIn("manual_training_or_artifacts_present", text)
        self.assertIn('"manual_runs"', text)
        self.assertIn("auto-collect triggered for manual run", text)

    def test_full_train_hf_remote_fetch_uses_unified_args_builder(self):
        text = Path("scripts/runpod_cycle_full_train_hf.sh").read_text(encoding="utf-8")
        self.assertIn("fetch_args=(", text)
        self.assertIn("fetch_args+=( --all-latest )", text)
        self.assertIn("fetch_args+=( --dataset-name", text)
        self.assertIn("hf_dataset_fetch.py", text)
        self.assertIn("\\${fetch_args[@]}", text)
        self.assertIn("runpod_cycle_report_style.py", text)
        self.assertIn("easy_progress_report.md", text)

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
            env["RUNPOD_TEMP_SSH_KEY_BASE"] = str(Path(td) / "dummy_key")
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
            env["RUNPOD_TEMP_SSH_KEY_BASE"] = str(Path(td) / "dummy_key")
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
