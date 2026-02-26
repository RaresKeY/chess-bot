import json
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
from src.chessbot.training import collate_train, train_next_move_model, train_next_move_model_from_jsonl_paths


class TrainingFeatureTests(unittest.TestCase):
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

        epoch_end_events = [e for e in events if e.get("event") == "epoch_end"]
        self.assertEqual(epoch_end_events[0]["epoch"], 1)
        self.assertEqual(epoch_end_events[1]["epoch"], 2)
        self.assertIn("metrics", epoch_end_events[0])
        self.assertIn("val_loss", epoch_end_events[0]["metrics"])


if __name__ == "__main__":
    unittest.main()
