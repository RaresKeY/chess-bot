import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment-dependent skip path
    torch = None

if torch is not None:
    from src.chessbot.board_features import BOARD_FEATURE_PLANES
    from src.chessbot.model import NextMoveSeqBoardLSTM, NextMoveSeqLSTM
    from src.chessbot.training_dual_sequence import (
        MODEL_FAMILY_DUAL_BOOTSTRAP_CURRICULUM,
        MODEL_FAMILY_DUAL_SEQUENCE_BOARD,
        _build_step_weights,
        _compute_mate_bias_weights,
        _masked_sequence_loss,
        compute_sequence_match_metrics,
        train_bootstrap_dual_head_curriculum_from_jsonl_paths,
        train_dual_sequence_model_from_jsonl_paths,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class DualSequenceTrainingTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_next_move_seq_lstm_forward_shape(self) -> None:
        model = NextMoveSeqLSTM(
            vocab_size=32,
            horizon=8,
            embed_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            use_side_to_move=True,
            side_to_move_embed_dim=2,
        )
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        lengths = torch.tensor([3, 2], dtype=torch.long)
        sides = torch.tensor([0, 1], dtype=torch.long)
        out = model(tokens, lengths, sides)
        self.assertEqual(out.shape, (2, 8, 32))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_next_move_seq_board_lstm_forward_shape(self) -> None:
        model = NextMoveSeqBoardLSTM(
            vocab_size=32,
            horizon=8,
            embed_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            use_side_to_move=True,
            side_to_move_embed_dim=2,
            board_feature_planes=BOARD_FEATURE_PLANES,
        )
        tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        lengths = torch.tensor([3, 2], dtype=torch.long)
        sides = torch.tensor([0, 1], dtype=torch.long)
        boards = torch.zeros((2, BOARD_FEATURE_PLANES, 8, 8), dtype=torch.float32)
        out = model(tokens, lengths, sides, board_state=boards)
        self.assertEqual(out.shape, (2, 8, 32))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_masked_sequence_loss_respects_mask(self) -> None:
        logits = torch.tensor(
            [
                [[6.0, 0.0], [0.0, 6.0], [0.0, 6.0]],
                [[0.0, 6.0], [6.0, 0.0], [6.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        step_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        loss = _masked_sequence_loss(
            logits=logits,
            targets=targets,
            mask=mask,
            step_weights=step_weights,
            mate_weights=None,
        )
        self.assertLess(float(loss.item()), 0.02)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_masked_sequence_loss_uses_target_probability_distance_and_step_decay(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 0.4054651081],  # p(target=0)=0.4 -> distance=0.6
                    [0.0, -2.1972245773],  # p(target=0)=0.9 -> distance=0.1
                ]
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1]], dtype=torch.bool)
        step_weights = torch.tensor([1.0, 0.5], dtype=torch.float32)
        loss = _masked_sequence_loss(
            logits=logits,
            targets=targets,
            mask=mask,
            step_weights=step_weights,
            mate_weights=None,
        )
        expected = (0.6 * 1.0 + 0.1 * 0.5) / (1.0 + 0.5)
        self.assertAlmostEqual(float(loss.item()), expected, places=4)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_step_weights_default_decay_frontloads_earlier_plies(self) -> None:
        weights = _build_step_weights(4, 1.0).tolist()
        self.assertEqual(len(weights), 4)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertGreater(weights[2], weights[3])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_compute_sequence_match_metrics_counts_exact_matches(self) -> None:
        logits = torch.tensor(
            [[[3.0, 1.0], [0.5, 2.5], [2.2, 0.2]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0, 1, 1]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        out = compute_sequence_match_metrics(logits=logits, targets=targets, mask=mask)
        self.assertEqual(out["ply_match_count"], 2.0)
        self.assertAlmostEqual(out["ply_match_rate"], 2.0 / 3.0, places=6)
        self.assertEqual(out["full_seq_exact_hit_rate"], 0.0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_endgame_mate_bias_weights_boost_until_mate(self) -> None:
        context = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]
        target = ["h5f7", "a7a6"]
        mask = [1, 1, 0, 0]
        out = _compute_mate_bias_weights(
            context=context,
            target=target,
            horizon=4,
            mask=mask,
            phase_name_hint="endgame",
            enabled=True,
            mate_in_x=2,
            mate_weight=2.0,
        )
        self.assertEqual(out[0], 2.0)
        self.assertEqual(out[1], 1.0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_dual_sequence_filters_by_winner_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.jsonl"
            val_path = tmp_path / "val.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5", "g1f3"], "winner_side": "W", "phase": "opening"},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"], "winner_side": "B", "phase": "opening"},
                {"context": ["c2c4"], "target": ["e7e5"], "winner_side": "D", "phase": "opening"},
            ]
            _write_jsonl(train_path, rows)
            _write_jsonl(val_path, rows)

            out_model = tmp_path / "model_white.pt"
            out_metrics = tmp_path / "metrics_white.json"
            result = train_dual_sequence_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                model_side="white",
                out_model_path=str(out_model),
                out_metrics_path=str(out_metrics),
                seed=1,
                horizon=4,
                epochs=1,
                batch_size=1,
                lr=1e-3,
                embed_dim=8,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                use_side_to_move_feature=True,
                side_to_move_embed_dim=2,
                step_loss_decay=1.0,
                num_workers=0,
                device_str="cpu",
                mate_bias_enabled=False,
                verbose=False,
            )
            self.assertTrue(out_model.is_file())
            self.assertTrue(out_metrics.is_file())
            metrics = result["metrics"]
            self.assertEqual(metrics["model_side"], "white")
            self.assertEqual(metrics["train_rows_total"], 1)
            self.assertEqual(metrics["val_rows_total"], 1)
            self.assertEqual(metrics["train_dropped_rows_total"], 2)
            self.assertEqual(len(metrics["history"]), 1)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_dual_sequence_board_family_sets_artifact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.jsonl"
            val_path = tmp_path / "val.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5", "g1f3"], "winner_side": "W", "phase": "opening"},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"], "winner_side": "B", "phase": "opening"},
            ]
            _write_jsonl(train_path, rows)
            _write_jsonl(val_path, rows)

            out_model = tmp_path / "model_white_board.pt"
            out_metrics = tmp_path / "metrics_white_board.json"
            result = train_dual_sequence_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                model_side="white",
                out_model_path=str(out_model),
                out_metrics_path=str(out_metrics),
                seed=1,
                horizon=4,
                epochs=1,
                batch_size=1,
                lr=1e-3,
                embed_dim=8,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                use_side_to_move_feature=True,
                side_to_move_embed_dim=2,
                step_loss_decay=0.9,
                num_workers=0,
                device_str="cpu",
                mate_bias_enabled=False,
                model_family=MODEL_FAMILY_DUAL_SEQUENCE_BOARD,
                verbose=False,
            )
            self.assertTrue(out_model.is_file())
            self.assertTrue(out_metrics.is_file())
            artifact = result["artifact"]
            metrics = result["metrics"]
            self.assertEqual(artifact["model_family"], MODEL_FAMILY_DUAL_SEQUENCE_BOARD)
            self.assertEqual(metrics["model_family"], MODEL_FAMILY_DUAL_SEQUENCE_BOARD)
            self.assertTrue(bool(metrics["use_board_state_feature"]))
            self.assertEqual(int(artifact["config"]["board_feature_planes"]), 18)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_dual_sequence_all_side_keeps_w_b_d_and_drops_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train_all.jsonl"
            val_path = tmp_path / "val_all.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5"], "winner_side": "W"},
                {"context": ["d2d4"], "target": ["d7d5"], "winner_side": "B"},
                {"context": ["c2c4"], "target": ["e7e6"], "winner_side": "D"},
                {"context": ["g1f3"], "target": ["d7d5"], "winner_side": "?"},
            ]
            _write_jsonl(train_path, rows)
            _write_jsonl(val_path, rows)

            out_model = tmp_path / "model_all.pt"
            result = train_dual_sequence_model_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                model_side="all",
                out_model_path=str(out_model),
                out_metrics_path=None,
                seed=1,
                horizon=4,
                epochs=1,
                batch_size=1,
                lr=1e-3,
                embed_dim=8,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                use_side_to_move_feature=True,
                side_to_move_embed_dim=2,
                step_loss_decay=0.9,
                num_workers=0,
                device_str="cpu",
                mate_bias_enabled=False,
                verbose=False,
            )
            metrics = result["metrics"]
            self.assertEqual(metrics["model_side"], "all")
            self.assertEqual(metrics["train_rows_total"], 3)
            self.assertEqual(metrics["val_rows_total"], 3)
            self.assertEqual(metrics["train_dropped_rows_total"], 1)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_bootstrap_dual_head_curriculum_sets_new_model_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train_bootstrap.jsonl"
            val_path = tmp_path / "val_bootstrap.jsonl"
            rows = [
                {"context": ["e2e4"], "target": ["e7e5", "g1f3"], "winner_side": "W", "phase": "opening"},
                {"context": ["d2d4"], "target": ["d7d5", "c2c4"], "winner_side": "B", "phase": "opening"},
                {"context": ["c2c4"], "target": ["e7e6", "d2d4"], "winner_side": "D", "phase": "middlegame"},
            ]
            _write_jsonl(train_path, rows)
            _write_jsonl(val_path, rows)

            out_shared = tmp_path / "model_shared.pt"
            out_white = tmp_path / "model_white.pt"
            out_black = tmp_path / "model_black.pt"
            result = train_bootstrap_dual_head_curriculum_from_jsonl_paths(
                train_paths=[str(train_path)],
                val_paths=[str(val_path)],
                out_model_shared_path=str(out_shared),
                out_model_white_path=str(out_white),
                out_model_black_path=str(out_black),
                out_metrics_path=None,
                seed=1,
                horizon=4,
                bootstrap_epochs=1,
                finetune_epochs=1,
                batch_size=1,
                lr=1e-3,
                embed_dim=8,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                use_side_to_move_feature=True,
                side_to_move_embed_dim=2,
                step_loss_decay=0.9,
                num_workers=0,
                device_str="cpu",
                mate_bias_enabled=False,
                model_family="dual_side_sequence_lstm",
                curriculum_start_horizon=1,
                curriculum_end_horizon=4,
                curriculum_ramp_epochs=1,
                verbose=False,
            )
            self.assertTrue(out_shared.is_file())
            self.assertTrue(out_white.is_file())
            self.assertTrue(out_black.is_file())
            white_artifact = torch.load(out_white, map_location="cpu")
            black_artifact = torch.load(out_black, map_location="cpu")
            self.assertEqual(white_artifact["model_family"], MODEL_FAMILY_DUAL_BOOTSTRAP_CURRICULUM)
            self.assertEqual(black_artifact["model_family"], MODEL_FAMILY_DUAL_BOOTSTRAP_CURRICULUM)
            merged = result["metrics"]
            self.assertEqual(merged["training_mode"], "dual_bootstrap_curriculum")
            self.assertEqual(merged["output_model_family"], MODEL_FAMILY_DUAL_BOOTSTRAP_CURRICULUM)
            self.assertIn("white", merged["sides"])
            self.assertIn("black", merged["sides"])


if __name__ == "__main__":
    unittest.main()
