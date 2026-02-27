import unittest
from pathlib import Path


class TrainBaselineCheckpointFlagsTests(unittest.TestCase):
    def test_train_baseline_exposes_subset_and_checkpoint_flags(self):
        text = Path("scripts/train_baseline.py").read_text(encoding="utf-8")
        self.assertIn("--max-train-rows", text)
        self.assertIn("--max-val-rows", text)
        self.assertIn("--max-total-rows", text)
        self.assertIn("--best-checkpoint-out", text)
        self.assertIn("--epoch-checkpoint-dir", text)

    def test_training_supports_subset_sampling_and_checkpoint_paths(self):
        text = Path("src/chessbot/training.py").read_text(encoding="utf-8")
        self.assertIn("def _sample_subset_indices", text)
        self.assertIn("max_total_rows: int = 0", text)
        self.assertIn("best_checkpoint_out: str = \"\"", text)
        self.assertIn("epoch_checkpoint_dir: str = \"\"", text)
        self.assertIn("best_checkpoint_saved_to_disk", text)
        self.assertIn("epoch_checkpoint_saved_to_disk", text)


if __name__ == "__main__":
    unittest.main()
