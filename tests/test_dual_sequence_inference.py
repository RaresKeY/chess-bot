import unittest
from unittest import mock

try:
    import torch
except Exception:  # pragma: no cover - environment-dependent skip path
    torch = None

if torch is not None:
    from src.chessbot.inference import (
        infer_sequence_from_artifact_on_device,
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

    @unittest.skipIf(torch is None, "torch not installed")
    def test_sequence_path_policy_uses_horizon_scores(self) -> None:
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "e2e4": 2,
            "d2d4": 3,
            "a7a6": 4,
            "e2e3": 5,
            "h2h3": 6,
        }
        artifact = {
            "artifact_format_version": 1,
            "model_family": "dual_side_sequence_lstm",
            "training_objective": "one_shot_sequence_match",
            "model_side": "white",
            "state_dict": {},
            "vocab": vocab,
            "config": {
                "horizon": 3,
                "embed_dim": 4,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "use_side_to_move": False,
                "side_to_move_embed_dim": 4,
            },
        }

        class _FakeSeqModel:
            def __init__(self, *args, **kwargs):
                self._logits = torch.tensor(
                    [
                        [
                            # step 1 prefers e2e4 over d2d4
                            [0.0, -10.0, 10.0, 9.0, -10.0, -10.0, -10.0],
                            # step 2 (black move) prefers a7a6
                            [0.0, -10.0, -10.0, -10.0, 8.0, -10.0, -10.0],
                            # step 3 strongly prefers e2e3 over h2h3
                            [0.0, -10.0, -10.0, -10.0, -10.0, 9.0, 1.0],
                        ]
                    ],
                    dtype=torch.float32,
                )

            def to(self, *_args, **_kwargs):
                return self

            def load_state_dict(self, *_args, **_kwargs):
                return None

            def eval(self):
                return None

            def __call__(self, *_args, **_kwargs):
                return self._logits

        with mock.patch("src.chessbot.inference.NextMoveSeqLSTM", _FakeSeqModel):
            out_step1 = infer_sequence_from_artifact_on_device(
                artifact=artifact,
                context=[],
                topk=2,
                device_str="cpu",
                sequence_decode_policy="step1_legal",
            )
            out_sequence = infer_sequence_from_artifact_on_device(
                artifact=artifact,
                context=[],
                topk=2,
                device_str="cpu",
                sequence_decode_policy="sequence_path",
            )

        self.assertEqual(out_step1["best_legal"], "e2e4")
        # sequence_path should prefer d2d4 because step3 can then keep high-score e2e3 legal
        self.assertEqual(out_sequence["best_legal"], "d2d4")
        self.assertEqual(out_sequence["legal_sequence"][:2], ["d2d4", "a7a6"])


if __name__ == "__main__":
    unittest.main()
