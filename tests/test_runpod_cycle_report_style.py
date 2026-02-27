import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunpodCycleReportStyleTests(unittest.TestCase):
    def test_report_script_writes_easy_style_summary(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            run_id = "runpod-cycle-20260227T120000Z-test"
            cycle = repo / "artifacts" / "runpod_cycles" / run_id
            collected = cycle / "collected" / "run_artifacts"
            eta_dir = cycle / run_id / "reports"
            collected.mkdir(parents=True, exist_ok=True)
            eta_dir.mkdir(parents=True, exist_ok=True)

            (collected / f"train_progress_{run_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "train_setup",
                                "epochs": 20,
                                "world_size": 4,
                                "distributed": "on",
                                "cache_load_reason_by_split": {"train": "hit", "val": "hit"},
                            }
                        ),
                        json.dumps(
                            {
                                "event": "epoch_end",
                                "epoch": 14,
                                "epochs": 20,
                                "metrics": {
                                    "train_loss": 6.1133,
                                    "val_loss": 6.184,
                                    "top1": 0.1114,
                                    "top5": 0.2540,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (collected / f"gpu_usage_samples_{run_id}.csv").write_text(
                "\n".join(
                    [
                        "2026-02-27T12:00:00Z,NVIDIA GeForce RTX 4090,97,12000,24564",
                        "2026-02-27T12:00:00Z,NVIDIA GeForce RTX 4090,95,12150,24564",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (collected / f"metrics_{run_id}.json").write_text(
                json.dumps({"epochs": 20, "history": [{"epoch": 14, "train_loss": 6.1133, "val_loss": 6.184, "top1": 0.1114, "top5": 0.254}]})
                + "\n",
                encoding="utf-8",
            )
            (collected / f"model_{run_id}.pt").write_bytes(b"model-bytes")
            (eta_dir / f"epoch_eta_report_{run_id}.jsonl").write_text(
                json.dumps(
                    {
                        "event": "epoch_end",
                        "epoch": 14,
                        "epochs": 20,
                        "eta_seconds_remaining": 199,
                        "eta_utc": "2026-02-27T12:03:19Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report_md = cycle / "reports" / "easy_progress_report.md"
            report_json = cycle / "reports" / "easy_progress_report.json"

            proc = subprocess.run(
                [
                    "python3",
                    "scripts/runpod_cycle_report_style.py",
                    "--run-id",
                    run_id,
                    "--repo-root",
                    str(repo),
                    "--write-md",
                    str(report_md),
                    "--write-json",
                    str(report_json),
                    "--quiet",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(report_md.is_file())
            self.assertTrue(report_json.is_file())

            text = report_md.read_text(encoding="utf-8")
            self.assertIn("# RunPod Easy Progress Report", text)
            self.assertIn("- epoch: `14/20`", text)
            self.assertIn("- eta_remaining: `3m 19s`", text)
            self.assertIn("- latest_util_avg_pct: `96.00`", text)
            self.assertIn("- world_size: `4`", text)
            self.assertIn("- cache_train: `hit`", text)
            self.assertIn("- model_local: `yes`", text)

