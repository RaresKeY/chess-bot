import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment-dependent skip path
    torch = None

if torch is not None:
    from src.chessbot.model import NextMoveLSTM, NextMoveSeqLSTM


def _build_dual_artifact(model_side: str) -> dict:
    vocab = {"<PAD>": 0, "<UNK>": 1, "e2e4": 2, "e7e5": 3}
    model = NextMoveSeqLSTM(
        vocab_size=len(vocab),
        horizon=4,
        embed_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        use_side_to_move=False,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.classifier.bias[2] = 2.0
        model.classifier.bias[3] = 1.0
    return {
        "artifact_format_version": 1,
        "model_family": "dual_side_sequence_lstm",
        "training_objective": "one_shot_sequence_match",
        "model_side": model_side,
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "horizon": 4,
            "embed_dim": 4,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "use_side_to_move": False,
            "side_to_move_embed_dim": 4,
        },
    }


def _build_next_move_artifact() -> dict:
    vocab = {"<PAD>": 0, "<UNK>": 1, "e2e4": 2, "e7e5": 3}
    model = NextMoveLSTM(
        vocab_size=len(vocab),
        embed_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        use_winner=False,
        use_phase=False,
        use_side_to_move=False,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.classifier.bias[2] = 2.0
        model.classifier.bias[3] = 1.0
    return {
        "artifact_format_version": 2,
        "model_family": "next_move_lstm",
        "training_objective": "single_step_next_move",
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": 4,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "use_winner": False,
            "use_phase": False,
            "phase_embed_dim": 8,
            "use_side_to_move": False,
            "side_to_move_embed_dim": 4,
        },
        "runtime": {
            "training_objective": "single_step_next_move",
        },
    }


class PlayModelVsModelCliTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_help_exposes_dual_pair_flags(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "scripts/play_model_vs_model.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--model-a-white", proc.stdout)
        self.assertIn("--model-a-black", proc.stdout)
        self.assertIn("--model-b-white", proc.stdout)
        self.assertIn("--model-b-black", proc.stdout)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_dual_pair_vs_single_runs_and_writes_summary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            a_white = tmp / "a_white.pt"
            a_black = tmp / "a_black.pt"
            b_single = tmp / "b_single.pt"
            summary_out = tmp / "summary.json"
            pgn_out = tmp / "games.pgn"

            torch.save(_build_dual_artifact("white"), a_white)
            torch.save(_build_dual_artifact("black"), a_black)
            torch.save(_build_next_move_artifact(), b_single)

            subprocess.run(
                [
                    sys.executable,
                    "scripts/play_model_vs_model.py",
                    "--model-a-white",
                    str(a_white),
                    "--model-a-black",
                    str(a_black),
                    "--model-b",
                    str(b_single),
                    "--alias-a",
                    "dual_pair",
                    "--alias-b",
                    "legacy_single",
                    "--games",
                    "1",
                    "--max-plies",
                    "12",
                    "--no-progress",
                    "--no-verbose",
                    "--summary-out",
                    str(summary_out),
                    "--pgn-out",
                    str(pgn_out),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(summary_out.is_file())
            self.assertTrue(pgn_out.is_file())
            payload = json.loads(summary_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["settings"]["model_a_mode"], "dual_pair")
            self.assertEqual(payload["settings"]["model_b_mode"], "single")
            self.assertEqual(int(payload["games"]), 1)


if __name__ == "__main__":
    unittest.main()
