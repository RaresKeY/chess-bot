import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment-dependent skip path
    torch = None

if torch is not None:
    from src.chessbot.model import NextMoveSeqLSTM
    from src.chessbot.training_dual_sequence import (
        _compute_mate_bias_weights,
        _masked_sequence_loss,
        compute_sequence_match_metrics,
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


if __name__ == "__main__":
    unittest.main()
