import unittest

import torch

from src.chessbot.inference import (
    artifact_training_objective,
    infer_first_move_auto_from_artifact_on_device,
    infer_from_artifact_on_device,
    infer_rollout_from_artifact_on_device,
)
from src.chessbot.model import NextMoveLSTM


class InferenceRolloutTests(unittest.TestCase):
    def _build_tiny_artifact(self):
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
            # Prefer e2e4, then e7e5. Legal filtering should pick the legal one by turn.
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

    def test_infer_rollout_returns_continuation_and_first_move(self):
        artifact = self._build_tiny_artifact()
        out = infer_rollout_from_artifact_on_device(
            artifact=artifact,
            context=[],
            winner_side="W",
            topk=2,
            rollout_plies=4,
            device_str="cpu",
            fallback_legal=False,
        )
        self.assertIn("rollout", out)
        self.assertIn("first_move", out)
        self.assertIn("steps_generated", out)
        self.assertIn("step_debug", out)
        self.assertEqual(out["first_move"], "e2e4")
        self.assertGreaterEqual(out["steps_generated"], 1)
        self.assertEqual(out["rollout"][0], "e2e4")

    def test_infer_auto_dispatch_uses_legacy_next_for_old_artifact(self):
        artifact = self._build_tiny_artifact()
        artifact.pop("training_objective", None)
        artifact.pop("runtime", None)
        self.assertEqual(artifact_training_objective(artifact), "single_step_next_move")
        out = infer_first_move_auto_from_artifact_on_device(
            artifact=artifact,
            context=[],
            winner_side="W",
            topk=2,
            device_str="cpu",
            policy_mode="auto",
        )
        self.assertEqual(out["policy_mode_used"], "next")
        self.assertEqual(out["move_uci"], "e2e4")

    def test_infer_auto_dispatch_prefers_rollout_for_multistep_artifact(self):
        artifact = self._build_tiny_artifact()
        artifact["runtime"] = {
            "training_objective": "multistep_teacher_forced_recursive",
            "rollout_horizon": 4,
        }
        out = infer_first_move_auto_from_artifact_on_device(
            artifact=artifact,
            context=[],
            winner_side="W",
            topk=2,
            device_str="cpu",
            policy_mode="auto",
            rollout_fallback_legal=False,
        )
        self.assertEqual(out["policy_mode_used"], "rollout")
        self.assertEqual(out["move_uci"], "e2e4")
        self.assertIn("rollout", out)

    def test_next_inference_clamps_topk_to_vocab_size(self):
        artifact = self._build_tiny_artifact()
        out = infer_from_artifact_on_device(
            artifact=artifact,
            context=[],
            winner_side="W",
            topk=99,
            device_str="cpu",
        )
        self.assertIn("topk", out)
        self.assertGreaterEqual(len(out["topk"]), 1)
        # tiny artifact has vocab size 4; topk must be clamped
        self.assertLessEqual(len(out["topk"]), 4)

    def test_next_inference_legal_masking_finds_legal_move_without_retry(self):
        artifact = self._build_tiny_artifact()
        # On black-to-move, top-1 raw token e2e4 is illegal; legal masking should still pick e7e5 at k=1.
        out = infer_from_artifact_on_device(
            artifact=artifact,
            context=["e2e4"],
            winner_side="W",
            topk=1,
            device_str="cpu",
            fallback_topk_multipliers=[1, 2],
        )
        self.assertEqual(out["best_legal"], "e7e5")
        self.assertEqual(out["decode_topk_used"], 1)
        self.assertEqual(out["decode_topk_attempts"], [1, 2])


if __name__ == "__main__":
    unittest.main()
