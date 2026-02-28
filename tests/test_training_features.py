from array import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from src.chessbot.model import (
    NextMoveLSTM,
    SIDE_TO_MOVE_BLACK,
    SIDE_TO_MOVE_WHITE,
    side_to_move_id_from_context_len,
)
from src.chessbot.training import (
    RuntimeSpliceConfig,
    _build_rollout_step_weights,
    _index_game_jsonl_paths_cached_or_runtime,
    _prefix_match_len,
    _try_load_runtime_splice_vocab_rows_meta_for_paths,
    _weighted_rollout_closeness,
    collate_train,
    collate_train_rollout,
    train_next_move_model,
    train_next_move_model_from_jsonl_paths,
)


class TrainingFeatureTests(unittest.TestCase):
    @staticmethod
    def _write_runtime_cache_split(
        dataset_dir: Path,
        split: str,
        src_path: Path,
        offsets: list[int],
        splice_indices: list[int],
        phase_ids: list[int],
    ) -> None:
        split_dir = dataset_dir / "runtime_splice_cache" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "paths.json").write_text(json.dumps([str(src_path.resolve())]), encoding="utf-8")

        def _write_arr(name: str, typecode: str, vals: list[int]) -> None:
            arr = array(typecode, vals)
            (split_dir / name).write_bytes(arr.tobytes())

        _write_arr("path_ids.u32.bin", "I", [0 for _ in offsets])
        _write_arr("offsets.u64.bin", "Q", offsets)
        _write_arr("splice_indices.u32.bin", "I", splice_indices)
        _write_arr("sample_phase_ids.u8.bin", "B", phase_ids)

    @staticmethod
    def _write_runtime_cache_manifest(dataset_dir: Path, *, min_context: int, min_target: int, max_samples_per_game: int, seed: int) -> None:
        manifest = {
            "schema_version": 1,
            "kind": "runtime_splice_cache",
            "config": {
                "min_context": int(min_context),
                "min_target": int(min_target),
                "max_samples_per_game": int(max_samples_per_game),
                "seed": int(seed),
            },
            "splits": {
                "train": {"game_rows_total": 1, "sample_rows_total": 1},
                "val": {"game_rows_total": 1, "sample_rows_total": 1},
            },
        }
        (dataset_dir / "runtime_splice_cache").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "runtime_splice_cache" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def _write_runtime_vocab_meta(
        dataset_dir: Path,
        *,
        min_context: int,
        min_target: int,
        max_samples_per_game: int,
        seed: int,
        train_game_rows: int,
        train_sample_rows: int,
        val_game_rows: int,
        val_sample_rows: int,
        vocab_tokens: list[str],
    ) -> None:
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for tok in vocab_tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        payload = {
            "kind": "runtime_splice_vocab_rows_meta_v1",
            "config": {
                "min_context": int(min_context),
                "min_target": int(min_target),
                "max_samples_per_game": int(max_samples_per_game),
                "seed": int(seed),
            },
            "splits": {
                "train": {"path": "train.jsonl", "game_rows": int(train_game_rows), "sample_rows": int(train_sample_rows)},
                "val": {"path": "val.jsonl", "game_rows": int(val_game_rows), "sample_rows": int(val_sample_rows)},
            },
            "vocab": vocab,
        }
        (dataset_dir / "runtime_splice_cache").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "runtime_splice_cache" / "vocab_rows_meta.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_side_to_move_id_from_context_len_parity_and_fallback(self):
        self.assertEqual(side_to_move_id_from_context_len(0), SIDE_TO_MOVE_WHITE)
        self.assertEqual(side_to_move_id_from_context_len(1), SIDE_TO_MOVE_BLACK)
        self.assertEqual(side_to_move_id_from_context_len(2), SIDE_TO_MOVE_WHITE)
        self.assertEqual(side_to_move_id_from_context_len("3"), SIDE_TO_MOVE_BLACK)
        self.assertEqual(side_to_move_id_from_context_len("bad"), SIDE_TO_MOVE_WHITE)

    def test_collate_train_returns_phase_and_side_to_move_tensors(self):
        batch = [
            ([1, 2], 7, 0, 1, SIDE_TO_MOVE_WHITE),
            ([3], 8, 1, 3, SIDE_TO_MOVE_BLACK),
        ]
        tokens, lengths, labels, winners, phases, side_to_moves = collate_train(batch)
        self.assertEqual(tokens.shape, (2, 2))
        self.assertTrue(torch.equal(lengths, torch.tensor([2, 1])))
        self.assertTrue(torch.equal(labels, torch.tensor([7, 8])))
        self.assertTrue(torch.equal(winners, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(phases, torch.tensor([1, 3])))
        self.assertTrue(torch.equal(side_to_moves, torch.tensor([0, 1])))

    def test_collate_train_rollout_returns_rollout_targets_and_mask(self):
        batch = [
            ([1, 2], 7, 0, 1, SIDE_TO_MOVE_WHITE, [7, 8, 0], [1, 1, 0]),
            ([3], 8, 1, 3, SIDE_TO_MOVE_BLACK, [8, 9, 10], [1, 1, 1]),
        ]
        out = collate_train_rollout(batch)
        self.assertEqual(len(out), 8)
        tokens, lengths, labels, winners, phases, side_to_moves, rollout_targets, rollout_mask = out
        self.assertEqual(tokens.shape, (2, 2))
        self.assertEqual(rollout_targets.shape, (2, 3))
        self.assertEqual(rollout_mask.shape, (2, 3))
        self.assertTrue(torch.equal(lengths, torch.tensor([2, 1])))
        self.assertTrue(torch.equal(labels, torch.tensor([7, 8])))
        self.assertTrue(torch.equal(rollout_targets[0], torch.tensor([7, 8, 0])))
        self.assertTrue(torch.equal(rollout_mask[0], torch.tensor([True, True, False])))

    def test_rollout_weight_helpers(self):
        ws = _build_rollout_step_weights(4, 0.7)
        self.assertEqual(len(ws), 4)
        self.assertAlmostEqual(ws[0], 1.0)
        self.assertAlmostEqual(ws[1], 0.7)
        self.assertEqual(_prefix_match_len([True, True, False, True], 4), 2)
        score = _weighted_rollout_closeness([True, False, True, False], [1.0, 0.7, 0.5, 0.35], 4)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_model_forward_supports_phase_and_side_features(self):
        torch.manual_seed(0)
        model = NextMoveLSTM(
            vocab_size=32,
            embed_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            use_winner=True,
            use_phase=True,
            phase_embed_dim=4,
            use_side_to_move=True,
            side_to_move_embed_dim=2,
        )
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        lengths = torch.tensor([3, 2], dtype=torch.long)
        winners = torch.tensor([0, 1], dtype=torch.long)
        phases = torch.tensor([1, 3], dtype=torch.long)
        sides = torch.tensor([SIDE_TO_MOVE_BLACK, SIDE_TO_MOVE_WHITE], dtype=torch.long)

        logits_explicit = model(tokens, lengths, winners, phases, sides)
        logits_implicit_side = model(tokens, lengths, winners, phases, None)

        self.assertEqual(logits_explicit.shape, (2, 32))
        self.assertEqual(logits_implicit_side.shape, (2, 32))

    def test_train_next_move_model_records_scheduler_and_early_stopping(self):
        train_rows = [
            {"context": ["e2e4", "e7e5"], "target": ["g1f3"], "next_move": "g1f3", "winner_side": "W", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
            {"context": ["c2c4", "e7e6"], "target": ["g2g3"], "next_move": "g2g3", "winner_side": "D", "phase": "middlegame"},
            {"context": ["g1f3", "d7d5"], "target": ["c2c4"], "next_move": "c2c4", "winner_side": "W", "phase": "middlegame"},
        ]
        val_rows = [
            {"context": ["e2e4"], "target": ["c7c5"], "next_move": "c7c5", "winner_side": "B", "phase": "opening"},
            {"context": ["e2e4", "c7c5"], "target": ["g1f3"], "next_move": "g1f3", "winner_side": "W", "phase": "opening"},
        ]

        constant_val = {"val_loss": 1.0, "top1": 0.1, "top5": 0.2}
        with mock.patch("src.chessbot.training.evaluate_loader", return_value=constant_val):
            artifact, history = train_next_move_model(
                train_rows=train_rows,
                val_rows=val_rows,
                epochs=6,
                batch_size=2,
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
                amp=False,
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="plateau",
                lr_scheduler_metric="val_loss",
                lr_plateau_factor=0.5,
                lr_plateau_patience=0,
                lr_plateau_threshold=0.0,
                early_stopping_patience=2,
                early_stopping_metric="val_loss",
                early_stopping_min_delta=0.0,
                verbose=False,
                show_progress=False,
            )

        self.assertLess(len(history), 6)
        self.assertEqual(len(history), 3)
        runtime = artifact["runtime"]
        self.assertTrue(runtime["early_stopping"]["used"])
        self.assertEqual(runtime["early_stopping"]["stopped_epoch"], 3)
        self.assertEqual(runtime["early_stopping"]["metric"], "val_loss")
        self.assertEqual(runtime["lr_scheduler"]["kind"], "plateau")
        self.assertTrue(runtime["lr_scheduler"]["enabled"])
        self.assertLess(runtime["lr_scheduler"]["final_lr"], 1e-3)
        self.assertTrue(artifact["config"]["use_phase"])
        self.assertTrue(artifact["config"]["use_side_to_move"])

    def test_train_next_move_model_from_jsonl_paths_emits_progress_events(self):
        train_rows = [
            {"context": ["e2e4"], "target": ["e7e5"], "next_move": "e7e5", "winner_side": "B", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
            {"context": ["g1f3"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "middlegame"},
            {"context": ["c2c4"], "target": ["e7e5"], "next_move": "e7e5", "winner_side": "B", "phase": "middlegame"},
        ]
        val_rows = [
            {"context": ["e2e4", "e7e5"], "target": ["g1f3"], "next_move": "g1f3", "winner_side": "W", "phase": "opening"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            train_path = f"{tmp}/train.jsonl"
            val_path = f"{tmp}/val.jsonl"
            with open(train_path, "w", encoding="utf-8") as f:
                for row in train_rows:
                    f.write(json.dumps(row) + "\n")
            with open(val_path, "w", encoding="utf-8") as f:
                for row in val_rows:
                    f.write(json.dumps(row) + "\n")

            events = []

            def on_progress(evt):
                events.append(evt)

            constant_val = {"val_loss": 1.0, "top1": 0.25, "top5": 0.5}
            with mock.patch("src.chessbot.training.evaluate_loader", return_value=constant_val):
                artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                    train_paths=[train_path],
                    val_paths=[val_path],
                    epochs=2,
                    batch_size=2,
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
                    restore_best=True,
                    use_phase_feature=True,
                    use_side_to_move_feature=True,
                    lr_scheduler="none",
                    early_stopping_patience=0,
                    verbose=False,
                    show_progress=False,
                    progress_callback=on_progress,
                )

        self.assertEqual(dataset_info["train_rows"], 4)
        self.assertEqual(dataset_info["val_rows"], 1)
        self.assertEqual(len(history), 2)
        self.assertTrue(artifact["runtime"]["best_checkpoint"]["enabled"])

        event_names = [e.get("event") for e in events]
        self.assertIn("train_setup", event_names)
        self.assertEqual(event_names.count("epoch_start"), 2)
        self.assertEqual(event_names.count("epoch_end"), 2)
        self.assertEqual(event_names[-1], "train_complete")
        train_setup = [e for e in events if e.get("event") == "train_setup"][0]
        self.assertIsNone(train_setup.get("cache_load_reason_by_split"))

        epoch_end_events = [e for e in events if e.get("event") == "epoch_end"]
        self.assertEqual(epoch_end_events[0]["epoch"], 1)
        self.assertEqual(epoch_end_events[1]["epoch"], 2)
        self.assertIn("metrics", epoch_end_events[0])
        self.assertIn("val_loss", epoch_end_events[0]["metrics"])
        self.assertIn("lr", epoch_end_events[0]["metrics"])

    def test_train_next_move_model_from_jsonl_paths_multi_input_tracks_per_file_rows(self):
        train_a_rows = [
            {"context": ["e2e4"], "target": ["e7e5"], "next_move": "e7e5", "winner_side": "B", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
        ]
        train_b_rows = [
            {"context": ["c2c4"], "target": ["e7e6"], "next_move": "e7e6", "winner_side": "B", "phase": "middlegame"},
        ]
        val_a_rows = [
            {"context": ["g1f3"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
        ]
        val_b_rows = [
            {"context": ["e2e4", "e7e5"], "target": ["g1f3"], "next_move": "g1f3", "winner_side": "W", "phase": "opening"},
            {"context": ["d2d4", "d7d5"], "target": ["c2c4"], "next_move": "c2c4", "winner_side": "W", "phase": "middlegame"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            train_a = Path(tmp) / "train_a.jsonl"
            train_b = Path(tmp) / "train_b.jsonl"
            val_a = Path(tmp) / "val_a.jsonl"
            val_b = Path(tmp) / "val_b.jsonl"
            for path, rows in (
                (train_a, train_a_rows),
                (train_b, train_b_rows),
                (val_a, val_a_rows),
                (val_b, val_b_rows),
            ):
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            events = []

            def on_progress(evt):
                events.append(evt)

            artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_a), str(train_b)],
                val_paths=[str(val_a), str(val_b)],
                epochs=1,
                batch_size=2,
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
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                progress_callback=on_progress,
            )

        self.assertIn("runtime", artifact)
        self.assertEqual(len(history), 1)
        self.assertEqual(dataset_info["train_rows"], 3)
        self.assertEqual(dataset_info["val_rows"], 3)
        self.assertEqual(dataset_info["train_rows_by_file"][str(train_a)], 2)
        self.assertEqual(dataset_info["train_rows_by_file"][str(train_b)], 1)
        self.assertEqual(dataset_info["val_rows_by_file"][str(val_a)], 1)
        self.assertEqual(dataset_info["val_rows_by_file"][str(val_b)], 2)
        self.assertEqual(dataset_info["train_index_rows"], 3)
        self.assertEqual(dataset_info["val_index_rows"], 3)

        train_setup = [e for e in events if e.get("event") == "train_setup"][0]
        self.assertEqual(train_setup["train_rows"], 3)
        self.assertEqual(train_setup["val_rows"], 3)
        epoch_end = [e for e in events if e.get("event") == "epoch_end"][0]
        self.assertIn("val_loss", epoch_end["metrics"])
        self.assertIn("top1", epoch_end["metrics"])
        self.assertIn("top5", epoch_end["metrics"])
        self.assertIn("lr", epoch_end["metrics"])

    def test_runtime_cache_index_loader_uses_cache_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset"
            ds.mkdir(parents=True, exist_ok=True)
            train_path = ds / "train.jsonl"
            train_path.write_text('{"game_id":"g1","moves":["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1"]}\n', encoding="utf-8")
            self._write_runtime_cache_manifest(ds, min_context=8, min_target=1, max_samples_per_game=0, seed=7)
            self._write_runtime_cache_split(ds, "train", train_path, offsets=[0], splice_indices=[7], phase_ids=[1])

            idx, used_cache, reason = _index_game_jsonl_paths_cached_or_runtime(
                [str(train_path)],
                RuntimeSpliceConfig(min_context=8, min_target=1, max_samples_per_game=0, seed=7),
                expected_split="train",
            )
            self.assertTrue(used_cache)
            self.assertEqual(reason, "loaded_runtime_splice_cache")
            self.assertEqual(len(idx[1]), 1)
            self.assertEqual(int(idx[3][0]), 7)
            self.assertEqual(int(idx[4][0]), 1)

    def test_runtime_cache_index_loader_falls_back_on_config_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset"
            ds.mkdir(parents=True, exist_ok=True)
            train_path = ds / "train.jsonl"
            train_path.write_text(
                '{"game_id":"g1","moves":["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1"]}\n',
                encoding="utf-8",
            )
            # cache config mismatch: min_context differs from runtime cfg below.
            self._write_runtime_cache_manifest(ds, min_context=4, min_target=1, max_samples_per_game=0, seed=7)
            self._write_runtime_cache_split(ds, "train", train_path, offsets=[0], splice_indices=[3], phase_ids=[1])

            idx, used_cache, reason = _index_game_jsonl_paths_cached_or_runtime(
                [str(train_path)],
                RuntimeSpliceConfig(min_context=8, min_target=1, max_samples_per_game=0, seed=7),
                expected_split="train",
            )
            self.assertFalse(used_cache)
            self.assertIn("cache_config_mismatch", reason)

    def test_runtime_vocab_rows_meta_loader_reads_counts_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            train_path = ds / "train.jsonl"
            val_path = ds / "val.jsonl"
            train_path.write_text('{"moves":["e2e4","e7e5"]}\n', encoding="utf-8")
            val_path.write_text('{"moves":["d2d4","d7d5"]}\n', encoding="utf-8")
            self._write_runtime_vocab_meta(
                ds,
                min_context=8,
                min_target=1,
                max_samples_per_game=0,
                seed=7,
                train_game_rows=11,
                train_sample_rows=101,
                val_game_rows=3,
                val_sample_rows=17,
                vocab_tokens=["e2e4", "e7e5", "g1f3"],
            )
            cfg = RuntimeSpliceConfig(min_context=8, min_target=1, max_samples_per_game=0, seed=7)
            out, reason = _try_load_runtime_splice_vocab_rows_meta_for_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                runtime_cfg=cfg,
            )
            self.assertEqual(reason, "loaded_runtime_splice_vocab_rows_meta")
            self.assertIsNotNone(out)
            vocab = out[0]
            self.assertIn("e2e4", vocab)
            self.assertEqual(out[2], 101)  # train_rows_total
            self.assertEqual(out[6], 17)   # val_rows_total

    def test_game_training_uses_runtime_cache_when_cache_present(self):
        rows = [
            {"game_id": "g1", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"], "winner_side": "W"},
            {"game_id": "g2", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"], "winner_side": "B"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset"
            ds.mkdir(parents=True, exist_ok=True)
            train_path = ds / "train.jsonl"
            val_path = ds / "val.jsonl"
            with train_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            with val_path.open("w", encoding="utf-8") as f:
                for row in rows[:1]:
                    f.write(json.dumps(row) + "\n")
            (ds / "stats.json").write_text(json.dumps({"dataset_format": "game_jsonl_runtime_splice_v1"}) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_runtime_splice_cache.py",
                    "--dataset-dir",
                    str(ds),
                    "--splits",
                    "train,val",
                    "--jobs",
                    "1",
                    "--no-progress-bar",
                    "--no-verbose",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )

            artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                epochs=1,
                batch_size=2,
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
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                require_runtime_splice_cache=True,
            )
            self.assertEqual(dataset_info["dataset_schema"], "game")
            self.assertEqual(dataset_info["data_loading"], "indexed_game_jsonl_runtime_splice_cache")
            self.assertEqual(dataset_info["cache_load_reason_by_split"], {"train": "hit", "val": "hit"})
            self.assertEqual(len(history), 1)
            self.assertIn("runtime", artifact)

    def test_game_training_require_runtime_cache_fails_when_cache_missing(self):
        rows = [
            {"game_id": "g1", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"], "winner_side": "W"},
            {"game_id": "g2", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"], "winner_side": "B"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset"
            ds.mkdir(parents=True, exist_ok=True)
            train_path = ds / "train.jsonl"
            val_path = ds / "val.jsonl"
            with train_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            with val_path.open("w", encoding="utf-8") as f:
                for row in rows[:1]:
                    f.write(json.dumps(row) + "\n")
            (ds / "stats.json").write_text(json.dumps({"dataset_format": "game_jsonl_runtime_splice_v1"}) + "\n", encoding="utf-8")

            self._write_runtime_cache_manifest(ds, min_context=8, min_target=1, max_samples_per_game=0, seed=7)
            self._write_runtime_cache_split(ds, "train", train_path, offsets=[0], splice_indices=[7], phase_ids=[1])

            with self.assertRaises(RuntimeError) as exc_info:
                train_next_move_model_from_jsonl_paths(
                    train_paths=[str(train_path)],
                    val_paths=[str(val_path)],
                    epochs=1,
                    batch_size=2,
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
                    restore_best=True,
                    use_phase_feature=True,
                    use_side_to_move_feature=True,
                    lr_scheduler="none",
                    early_stopping_patience=0,
                    verbose=False,
                    show_progress=False,
                    require_runtime_splice_cache=True,
                )
            self.assertIn("Runtime splice cache required but unavailable", str(exc_info.exception))

    def test_game_training_runtime_cache_accepts_dataset_alias_paths(self):
        rows = [
            {"game_id": "g1", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"], "winner_side": "W"},
            {"game_id": "g2", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"], "winner_side": "B"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = (
                Path(tmp)
                / "hf_datasets"
                / "elite_2025-01_game"
                / "20260227T044455Z"
                / "dataset"
            )
            dataset_dir.mkdir(parents=True, exist_ok=True)
            train_path = dataset_dir / "train.jsonl"
            val_path = dataset_dir / "val.jsonl"
            with train_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            with val_path.open("w", encoding="utf-8") as f:
                for row in rows[:1]:
                    f.write(json.dumps(row) + "\n")
            (dataset_dir / "stats.json").write_text(
                json.dumps({"dataset_format": "game_jsonl_runtime_splice_v1"}) + "\n",
                encoding="utf-8",
            )

            self._write_runtime_cache_manifest(
                dataset_dir,
                min_context=8,
                min_target=1,
                max_samples_per_game=0,
                seed=7,
            )
            self._write_runtime_cache_split(
                dataset_dir,
                "train",
                train_path,
                offsets=[0],
                splice_indices=[7],
                phase_ids=[1],
            )
            self._write_runtime_cache_split(
                dataset_dir,
                "val",
                val_path,
                offsets=[0],
                splice_indices=[7],
                phase_ids=[1],
            )

            # Simulate cache paths created on another machine/root while keeping
            # the dataset token + split filename stable.
            stale_train = "/home/mintmainog/workspace/vs_code_workspace/chess_bot/data/dataset/elite_2025-01_game/train.jsonl"
            stale_val = "/home/mintmainog/workspace/vs_code_workspace/chess_bot/data/dataset/elite_2025-01_game/val.jsonl"
            (dataset_dir / "runtime_splice_cache" / "train" / "paths.json").write_text(
                json.dumps([stale_train]),
                encoding="utf-8",
            )
            (dataset_dir / "runtime_splice_cache" / "val" / "paths.json").write_text(
                json.dumps([stale_val]),
                encoding="utf-8",
            )

            _artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                epochs=1,
                batch_size=2,
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
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                require_runtime_splice_cache=True,
            )
            self.assertEqual(dataset_info["data_loading"], "indexed_game_jsonl_runtime_splice_cache")
            self.assertEqual(dataset_info["cache_load_reason_by_split"], {"train": "hit", "val": "hit"})
            self.assertEqual(len(history), 1)

    def test_game_training_reports_cache_load_reason_per_split(self):
        rows = [
            {"game_id": "g1", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"], "winner_side": "W"},
            {"game_id": "g2", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"], "winner_side": "B"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset"
            ds.mkdir(parents=True, exist_ok=True)
            train_path = ds / "train.jsonl"
            val_path = ds / "val.jsonl"
            with train_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            with val_path.open("w", encoding="utf-8") as f:
                for row in rows[:1]:
                    f.write(json.dumps(row) + "\n")
            (ds / "stats.json").write_text(json.dumps({"dataset_format": "game_jsonl_runtime_splice_v1"}) + "\n", encoding="utf-8")

            self._write_runtime_cache_manifest(ds, min_context=8, min_target=1, max_samples_per_game=0, seed=7)
            self._write_runtime_cache_split(ds, "train", train_path, offsets=[0], splice_indices=[7], phase_ids=[1])

            events = []

            def on_progress(evt):
                events.append(evt)

            _artifact, _history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                epochs=1,
                batch_size=2,
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
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                progress_callback=on_progress,
            )

        self.assertEqual(dataset_info["cache_load_reason_by_split"]["train"], "hit")
        self.assertIn("cache_file_missing:", dataset_info["cache_load_reason_by_split"]["val"])
        train_setup = [e for e in events if e.get("event") == "train_setup"][0]
        self.assertEqual(train_setup["cache_load_reason_by_split"]["train"], "hit")
        self.assertIn("cache_file_missing:", train_setup["cache_load_reason_by_split"]["val"])

    def test_train_next_move_model_from_jsonl_paths_multistep_emits_rollout_metrics(self):
        train_rows = [
            {"context": ["e2e4"], "target": ["e7e5", "g1f3", "b8c6", "f1b5"], "next_move": "e7e5", "winner_side": "B", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5", "c2c4", "e7e6", "b1c3"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
            {"context": ["g1f3"], "target": ["d7d5", "d2d4", "g8f6", "c2c4"], "next_move": "d7d5", "winner_side": "B", "phase": "middlegame"},
            {"context": ["c2c4"], "target": ["e7e5", "b1c3", "g8f6", "g2g3"], "next_move": "e7e5", "winner_side": "B", "phase": "middlegame"},
        ]
        val_rows = [
            {"context": ["e2e4", "e7e5"], "target": ["g1f3", "b8c6", "f1b5", "a7a6"], "next_move": "g1f3", "winner_side": "W", "phase": "opening"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            train_path = f"{tmp}/train.jsonl"
            val_path = f"{tmp}/val.jsonl"
            with open(train_path, "w", encoding="utf-8") as f:
                for row in train_rows:
                    f.write(json.dumps(row) + "\n")
            with open(val_path, "w", encoding="utf-8") as f:
                for row in val_rows:
                    f.write(json.dumps(row) + "\n")

            events = []

            def on_progress(evt):
                events.append(evt)

            artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[train_path],
                val_paths=[val_path],
                epochs=1,
                batch_size=2,
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
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                progress_callback=on_progress,
                rollout_horizon=4,
                closeness_horizon=4,
                rollout_loss_decay=0.7,
            )

        self.assertEqual(len(history), 1)
        self.assertEqual(dataset_info["training_objective"], "multistep_teacher_forced_recursive")
        self.assertEqual(dataset_info["rollout_horizon"], 4)
        self.assertIn("rollout_step1_acc", history[0])
        self.assertIn("rollout_step4_acc", history[0])
        self.assertIn("rollout_prefix_match_len_avg", history[0])
        self.assertIn("rollout_weighted_continuation_score", history[0])
        self.assertEqual(artifact["runtime"]["training_objective"], "multistep_teacher_forced_recursive")
        self.assertEqual(artifact["runtime"]["rollout_horizon"], 4)

        event_names = [e.get("event") for e in events]
        self.assertIn("train_setup", event_names)
        epoch_end = [e for e in events if e.get("event") == "epoch_end"][0]
        self.assertIn("rollout_step4_acc", epoch_end["metrics"])
        self.assertIn("rollout_weighted_continuation_score", epoch_end["metrics"])

    def test_train_next_move_model_from_jsonl_paths_multistep_multi_input_tracks_rows_and_metrics(self):
        train_a_rows = [
            {"context": ["e2e4"], "target": ["e7e5", "g1f3", "b8c6", "f1b5"], "next_move": "e7e5", "winner_side": "B", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5", "c2c4", "e7e6", "b1c3"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
        ]
        train_b_rows = [
            {"context": ["g1f3"], "target": ["d7d5", "d2d4", "g8f6", "c2c4"], "next_move": "d7d5", "winner_side": "B", "phase": "middlegame"},
        ]
        val_a_rows = [
            {"context": ["c2c4"], "target": ["e7e5", "b1c3", "g8f6", "g2g3"], "next_move": "e7e5", "winner_side": "B", "phase": "middlegame"},
        ]
        val_b_rows = [
            {"context": ["e2e4", "e7e5"], "target": ["g1f3", "b8c6", "f1b5", "a7a6"], "next_move": "g1f3", "winner_side": "W", "phase": "opening"},
            {"context": ["d2d4", "d7d5"], "target": ["c2c4", "e7e6", "b1c3", "g8f6"], "next_move": "c2c4", "winner_side": "W", "phase": "opening"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            train_a = Path(tmp) / "train_a.jsonl"
            train_b = Path(tmp) / "train_b.jsonl"
            val_a = Path(tmp) / "val_a.jsonl"
            val_b = Path(tmp) / "val_b.jsonl"
            for path, rows in (
                (train_a, train_a_rows),
                (train_b, train_b_rows),
                (val_a, val_a_rows),
                (val_b, val_b_rows),
            ):
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            events = []

            def on_progress(evt):
                events.append(evt)

            artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                train_paths=[str(train_a), str(train_b)],
                val_paths=[str(val_a), str(val_b)],
                epochs=1,
                batch_size=2,
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
                restore_best=True,
                use_phase_feature=True,
                use_side_to_move_feature=True,
                lr_scheduler="none",
                early_stopping_patience=0,
                verbose=False,
                show_progress=False,
                progress_callback=on_progress,
                rollout_horizon=4,
                closeness_horizon=4,
                rollout_loss_decay=0.7,
            )

        self.assertEqual(len(history), 1)
        self.assertEqual(dataset_info["train_rows"], 3)
        self.assertEqual(dataset_info["val_rows"], 3)
        self.assertEqual(dataset_info["train_rows_by_file"][str(train_a)], 2)
        self.assertEqual(dataset_info["train_rows_by_file"][str(train_b)], 1)
        self.assertEqual(dataset_info["val_rows_by_file"][str(val_a)], 1)
        self.assertEqual(dataset_info["val_rows_by_file"][str(val_b)], 2)
        self.assertEqual(dataset_info["training_objective"], "multistep_teacher_forced_recursive")
        self.assertIn("rollout_step4_acc", history[0])
        self.assertEqual(artifact["runtime"]["training_objective"], "multistep_teacher_forced_recursive")

        train_setup = [e for e in events if e.get("event") == "train_setup"][0]
        self.assertEqual(train_setup["train_rows"], 3)
        self.assertEqual(train_setup["val_rows"], 3)
        epoch_end = [e for e in events if e.get("event") == "epoch_end"][0]
        self.assertIn("val_loss", epoch_end["metrics"])
        self.assertIn("rollout_step2_acc", epoch_end["metrics"])
        self.assertIn("rollout_step4_acc", epoch_end["metrics"])
        self.assertIn("rollout_weighted_continuation_score", epoch_end["metrics"])

    def test_distributed_non_primary_suppresses_progress_events(self):
        class _DummyDDP(torch.nn.Module):
            def __init__(self, module, **_kwargs):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class _DummyDistributedSampler(torch.utils.data.Sampler):
            def __init__(self, data_source, **_kwargs):
                self._n = len(data_source)

            def __iter__(self):
                return iter(range(self._n))

            def __len__(self):
                return self._n

            def set_epoch(self, _epoch):
                return None

        rows = [
            {"context": ["e2e4"], "target": ["e7e5"], "next_move": "e7e5", "winner_side": "B", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.jsonl"
            val_path = Path(tmp) / "val.jsonl"
            for path in (train_path, val_path):
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            events = []

            def on_progress(evt):
                events.append(evt)

            with mock.patch("src.chessbot.training.DDP", _DummyDDP), mock.patch(
                "src.chessbot.training.DistributedSampler", _DummyDistributedSampler
            ):
                artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                    train_paths=[str(train_path)],
                    val_paths=[str(val_path)],
                    epochs=1,
                    batch_size=2,
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
                    restore_best=True,
                    use_phase_feature=True,
                    use_side_to_move_feature=True,
                    lr_scheduler="none",
                    early_stopping_patience=0,
                    verbose=False,
                    show_progress=False,
                    progress_callback=on_progress,
                    distributed_enabled=True,
                    distributed_rank=1,
                    distributed_world_size=2,
                )
        self.assertEqual(events, [])
        self.assertEqual(len(history), 1)
        self.assertTrue(dataset_info["distributed"]["enabled"])
        self.assertEqual(dataset_info["distributed"]["rank"], 1)
        self.assertEqual(artifact["runtime"]["distributed"]["world_size"], 2)

    def test_distributed_primary_emits_progress_and_metadata(self):
        class _DummyDDP(torch.nn.Module):
            def __init__(self, module, **_kwargs):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        class _DummyDistributedSampler(torch.utils.data.Sampler):
            def __init__(self, data_source, **_kwargs):
                self._n = len(data_source)

            def __iter__(self):
                return iter(range(self._n))

            def __len__(self):
                return self._n

            def set_epoch(self, _epoch):
                return None

        rows = [
            {"context": ["e2e4"], "target": ["e7e5"], "next_move": "e7e5", "winner_side": "B", "phase": "opening"},
            {"context": ["d2d4"], "target": ["d7d5"], "next_move": "d7d5", "winner_side": "B", "phase": "opening"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.jsonl"
            val_path = Path(tmp) / "val.jsonl"
            for path in (train_path, val_path):
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
            events = []

            def on_progress(evt):
                events.append(evt)

            with mock.patch("src.chessbot.training.DDP", _DummyDDP), mock.patch(
                "src.chessbot.training.DistributedSampler", _DummyDistributedSampler
            ):
                artifact, history, dataset_info = train_next_move_model_from_jsonl_paths(
                    train_paths=[str(train_path)],
                    val_paths=[str(val_path)],
                    epochs=1,
                    batch_size=2,
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
                    restore_best=True,
                    use_phase_feature=True,
                    use_side_to_move_feature=True,
                    lr_scheduler="none",
                    early_stopping_patience=0,
                    verbose=False,
                    show_progress=False,
                    progress_callback=on_progress,
                    distributed_enabled=True,
                    distributed_rank=0,
                    distributed_world_size=2,
                )
        self.assertEqual(len(history), 1)
        event_names = [e.get("event") for e in events]
        self.assertIn("train_setup", event_names)
        self.assertIn("train_complete", event_names)
        setup = [e for e in events if e.get("event") == "train_setup"][0]
        self.assertEqual(setup["distributed"]["rank"], 0)
        self.assertEqual(setup["distributed"]["world_size"], 2)
        self.assertEqual(dataset_info["distributed"]["rank"], 0)
        self.assertTrue(artifact["runtime"]["distributed"]["enabled"])


if __name__ == "__main__":
    unittest.main()
