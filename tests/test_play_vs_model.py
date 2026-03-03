import unittest

import torch

from src.chessbot.model import NextMoveLSTM, NextMoveSeqLSTM
from src.chessbot.play_vs_model import (
    LoadedMoveModel,
    PlayConfig,
    apply_model_move,
    apply_user_and_model_move,
    model_move_response,
)


def _build_tiny_next_artifact() -> dict:
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
        # White turn prefers e2e4, black turn legal filter picks e7e5.
        model.classifier.bias[2] = 2.0
        model.classifier.bias[3] = 1.0
    return {
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
    }


def _build_dual_sequence_artifact(model_side: str) -> dict:
    vocab = {"<PAD>": 0, "<UNK>": 1, "e2e4": 2, "e7e5": 3}
    model = NextMoveSeqLSTM(
        vocab_size=len(vocab),
        horizon=2,
        embed_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        use_side_to_move=False,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        if model_side == "white":
            model.classifier.bias[2] = 2.0
            model.classifier.bias[3] = 1.0
        else:
            model.classifier.bias[3] = 2.0
            model.classifier.bias[2] = 1.0
    return {
        "artifact_format_version": 1,
        "model_family": "dual_side_sequence_lstm",
        "training_objective": "one_shot_sequence_match",
        "model_side": model_side,
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "horizon": 2,
            "embed_dim": 4,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "use_side_to_move": False,
            "side_to_move_embed_dim": 4,
        },
    }


class PlayVsModelTests(unittest.TestCase):
    def test_black_user_requires_model_opening_move(self) -> None:
        runtime = LoadedMoveModel(artifact=_build_tiny_next_artifact(), device_str="cpu")
        cfg = PlayConfig(winner_side="W", topk=2, user_color="black")
        with self.assertRaisesRegex(ValueError, "not black's turn"):
            apply_user_and_model_move(runtime, [], "e7e5", cfg)

        state = apply_model_move(runtime, [], cfg)
        self.assertEqual(state["context"][:1], ["e2e4"])
        self.assertEqual((state.get("last_model_move") or {}).get("uci"), "e2e4")
        self.assertEqual(state["turn"], "black")

    def test_white_user_move_then_model_reply_single_artifact(self) -> None:
        runtime = LoadedMoveModel(artifact=_build_tiny_next_artifact(), device_str="cpu")
        cfg = PlayConfig(winner_side="W", topk=2, user_color="white")
        state = apply_user_and_model_move(runtime, [], "e2e4", cfg)
        self.assertEqual(state["context"][:2], ["e2e4", "e7e5"])
        self.assertEqual((state.get("last_user_move") or {}).get("uci"), "e2e4")
        self.assertEqual((state.get("last_model_move") or {}).get("uci"), "e7e5")

    def test_dual_pair_routes_black_side_after_white_user_move(self) -> None:
        runtime = LoadedMoveModel(
            white_artifact=_build_dual_sequence_artifact("white"),
            black_artifact=_build_dual_sequence_artifact("black"),
            device_str="cpu",
        )
        cfg = PlayConfig(winner_side="W", topk=2, user_color="white")
        state = apply_user_and_model_move(runtime, [], "e2e4", cfg)
        self.assertEqual((state.get("last_model_move") or {}).get("uci"), "e7e5")
        self.assertEqual((state.get("last_model_move") or {}).get("selected_model_side"), "black")

    def test_model_move_response_returns_snapshots(self) -> None:
        runtime = LoadedMoveModel(artifact=_build_tiny_next_artifact(), device_str="cpu")
        cfg = PlayConfig(winner_side="W", topk=2, user_color="black")
        state = model_move_response(runtime, [], cfg)
        self.assertIn("snapshots", state)
        self.assertEqual(len(state["snapshots"]), 2)
        self.assertEqual((state.get("last_model_move") or {}).get("uci"), "e2e4")


if __name__ == "__main__":
    unittest.main()
