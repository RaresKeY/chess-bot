import json
from pathlib import Path

from src.chessbot.runpod_cycle_verify import verify_full_hf_cycle_run


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def test_verify_full_hf_cycle_run_ok(tmp_path: Path) -> None:
    run_id = "fulltrain-20260226T000000Z"
    cycle_dir = tmp_path / "artifacts" / "runpod_cycles" / run_id
    run_artifacts = cycle_dir / "collected" / "run_artifacts"
    run_artifacts.mkdir(parents=True)

    _write_json(cycle_dir / "provision.json", {"pod_id": "abc"})
    _write_json(cycle_dir / "stop_response.json", {"data": {"podStop": {"id": "abc", "desiredStatus": "EXITED"}}})
    _write_json(cycle_dir / "terminate_response.json", {"pod_id": "abc", "http_code": "204", "response_body": ""})

    _write_json(
        run_artifacts / "hf_dataset_fetch_manifest.json",
        {
            "aggregate": {"dataset_count": 1},
            "aggregate_by_format": {"game_jsonl_runtime_splice_v1": {"dataset_count": 1, "train_paths": ["x"], "val_paths": ["y"]}},
        },
    )
    _write_json(run_artifacts / f"context_probe_{run_id}.json", {"dataset": {}, "gpu": {}})
    _write_json(run_artifacts / f"metrics_{run_id}.json", {"epochs": 1, "history": [{"epoch": 1, "val_loss": 1.23, "top1": 0.5}]})
    (run_artifacts / f"model_{run_id}.pt").write_bytes(b"pt")
    (run_artifacts / "train_exit_code.txt").write_text("0\n", encoding="utf-8")
    (run_artifacts / f"train_stdout_{run_id}.log").write_text("ok\n", encoding="utf-8")
    (run_artifacts / f"train_progress_{run_id}.jsonl").write_text("{}\n", encoding="utf-8")
    (run_artifacts / f"gpu_usage_samples_{run_id}.csv").write_text("a,b,c,d,e\n", encoding="utf-8")

    res = verify_full_hf_cycle_run(tmp_path, run_id, require_terminated=True)
    assert res["ok"] is True
    assert res["checks"]["train_exit_code_zero"] is True
    assert res["checks"]["terminate_http_ok"] is True
    assert res["hf_manifest"]["has_game_format_bucket"] is True


def test_verify_full_hf_cycle_run_fails_when_missing_model(tmp_path: Path) -> None:
    run_id = "fulltrain-20260226T000001Z"
    cycle_dir = tmp_path / "artifacts" / "runpod_cycles" / run_id
    run_artifacts = cycle_dir / "collected" / "run_artifacts"
    run_artifacts.mkdir(parents=True)
    _write_json(cycle_dir / "provision.json", {"pod_id": "abc"})
    _write_json(cycle_dir / "stop_response.json", {"data": {"podStop": {"desiredStatus": "EXITED"}}})
    (run_artifacts / "train_exit_code.txt").write_text("0\n", encoding="utf-8")

    res = verify_full_hf_cycle_run(tmp_path, run_id)
    assert res["ok"] is False
    assert res["checks"]["model_path"] is False
