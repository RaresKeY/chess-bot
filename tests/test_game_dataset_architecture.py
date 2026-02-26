import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.chessbot.training import train_next_move_model_from_jsonl_paths


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class GameDatasetArchitectureTests(unittest.TestCase):
    def _sample_validated_games(self):
        # Legal UCI move sequences; enough plies for runtime splicing min_context=4.
        return [
            {
                "game_id": "g1",
                "winner_side": "W",
                "result": "1-0",
                "plies": 10,
                "moves_uci": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"],
                "headers": {"Event": "Test"},
            },
            {
                "game_id": "g2",
                "winner_side": "B",
                "result": "0-1",
                "plies": 10,
                "moves_uci": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8"],
            },
            {
                "game_id": "g3",
                "winner_side": "W",
                "result": "1-0",
                "plies": 10,
                "moves_uci": ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5", "f1g2", "d5b6"],
            },
            {
                "game_id": "g4",
                "winner_side": "B",
                "result": "0-1",
                "plies": 10,
                "moves_uci": ["g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "c1g5", "h7h6"],
            },
        ]

    def test_build_game_dataset_cli_outputs_compact_moves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid_in = base / "valid_games.jsonl"
            out_dir = base / "dataset"
            _write_jsonl(valid_in, self._sample_validated_games())
            cmd = [
                str((Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")),
                "scripts/build_game_dataset.py",
                "--input",
                str(valid_in),
                "--output-dir",
                str(out_dir),
                "--runtime-min-context",
                "4",
                "--runtime-min-target",
                "1",
                "--progress-every",
                "0",
            ]
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["dataset_format"], "game_jsonl_runtime_splice_v1")
            self.assertGreater(sum(stats["split_games"].values()), 0)
            any_row = None
            for split in ("train", "val", "test"):
                p = out_dir / f"{split}.jsonl"
                if p.exists() and p.stat().st_size > 0:
                    any_row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
                    break
            self.assertIsNotNone(any_row)
            self.assertIn("moves", any_row)
            self.assertNotIn("moves_uci", any_row)
            self.assertEqual(any_row.get("schema"), "game_dataset_runtime_splice_v1")

    def test_train_from_game_dataset_single_step_runtime_splice(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train_path = base / "train.jsonl"
            val_path = base / "val.jsonl"
            rows = []
            for row in self._sample_validated_games():
                rows.append(
                    {
                        "game_id": row["game_id"],
                        "winner_side": row["winner_side"],
                        "result": row["result"],
                        "plies": row["plies"],
                        "moves": row["moves_uci"],
                    }
                )
            _write_jsonl(train_path, rows[:3])
            _write_jsonl(val_path, rows[3:])

            artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                epochs=1,
                batch_size=4,
                lr=1e-3,
                seed=7,
                embed_dim=8,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                winner_weight=1.0,
                use_winner=True,
                device_str="cpu",
                num_workers=0,
                pin_memory=False,
                amp=False,
                restore_best=False,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                runtime_min_context=4,
                runtime_min_target=1,
                runtime_max_samples_per_game=2,
            )
            self.assertEqual(dataset_info["dataset_schema"], "game")
            self.assertEqual(dataset_info["data_loading"], "indexed_game_jsonl_runtime_splice")
            self.assertEqual(dataset_info["train_games"], 3)
            self.assertEqual(dataset_info["val_games"], 1)
            self.assertGreater(dataset_info["train_rows"], 0)
            self.assertEqual(artifact["runtime"]["training_objective"], "single_step_next_move")
            self.assertEqual(len(history), 1)

    def test_train_from_game_dataset_multistep_runtime_splice(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train_path = base / "train.jsonl"
            val_path = base / "val.jsonl"
            rows = []
            for row in self._sample_validated_games():
                rows.append(
                    {
                        "game_id": row["game_id"],
                        "winner_side": row["winner_side"],
                        "result": row["result"],
                        "plies": row["plies"],
                        "moves": row["moves_uci"],
                    }
                )
            _write_jsonl(train_path, rows[:3])
            _write_jsonl(val_path, rows[3:])

            artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                epochs=1,
                batch_size=4,
                lr=1e-3,
                seed=7,
                embed_dim=8,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                winner_weight=1.0,
                use_winner=True,
                device_str="cpu",
                num_workers=0,
                pin_memory=False,
                amp=False,
                restore_best=False,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                rollout_horizon=4,
                closeness_horizon=4,
                rollout_loss_decay=0.7,
                runtime_min_context=4,
                runtime_min_target=1,
                runtime_max_samples_per_game=2,
            )
            self.assertEqual(dataset_info["dataset_schema"], "game")
            self.assertEqual(dataset_info["data_loading"], "indexed_game_jsonl_runtime_splice")
            self.assertEqual(dataset_info["runtime_splice"]["max_samples_per_game"], 2)
            self.assertEqual(artifact["runtime"]["training_objective"], "multistep_teacher_forced_recursive")
            self.assertEqual(artifact["runtime"]["rollout_horizon"], 4)
            self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
