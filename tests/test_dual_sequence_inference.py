import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment-dependent skip path
    torch = None

if torch is not None:
    from src.chessbot.inference import (
        infer_first_move_auto_from_artifact_on_device,
        infer_first_move_dual_artifacts_on_device,
    )
    from src.chessbot.model import NextMoveSeqLSTM


def _build_dual_sequence_artifact(model_side: str) -> dict:
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


class DualSequenceInferenceTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_dual_artifact_routing_selects_model_by_side_to_move(self) -> None:
        white_artifact = _build_dual_sequence_artifact("white")
        black_artifact = _build_dual_sequence_artifact("black")

        out_white_turn = infer_first_move_dual_artifacts_on_device(
            white_artifact=white_artifact,
            black_artifact=black_artifact,
            context=[],
            topk=2,
            device_str="cpu",
        )
        self.assertEqual(out_white_turn["selected_model_side"], "white")
        self.assertEqual(out_white_turn["move_uci"], "e2e4")

        out_black_turn = infer_first_move_dual_artifacts_on_device(
            white_artifact=white_artifact,
            black_artifact=black_artifact,
            context=["e2e4"],
            topk=2,
            device_str="cpu",
        )
        self.assertEqual(out_black_turn["selected_model_side"], "black")
        self.assertEqual(out_black_turn["move_uci"], "e7e5")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_infer_auto_supports_dual_sequence_artifact_family(self) -> None:
        artifact = _build_dual_sequence_artifact("white")
        out = infer_first_move_auto_from_artifact_on_device(
            artifact=artifact,
            context=[],
            winner_side="W",
            topk=2,
            device_str="cpu",
            policy_mode="auto",
        )
        self.assertEqual(out["policy_mode_used"], "sequence")
        self.assertEqual(out["move_uci"], "e2e4")


if __name__ == "__main__":
    unittest.main()
