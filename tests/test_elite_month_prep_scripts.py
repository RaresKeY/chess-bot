import subprocess
import tempfile
import unittest
from pathlib import Path


class EliteMonthPrepScriptTests(unittest.TestCase):
    def test_acquire_script_defaults_cap8_and_dynamic_dataset_suffix(self):
        text = Path("scripts/acquire_and_prepare_elite_month.py").read_text(encoding="utf-8")
        self.assertIn('--max-samples-per-game", type=int, default=8', text)
        self.assertIn("elite_<month>_cap<max-samples-per-game>", text)
        self.assertIn('f"elite_{month}_cap{args.max_samples_per_game}"', text)

    def test_batch_script_parses_url_list_and_emits_cap8_acquire_commands_in_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            list_file = Path(td) / "months.txt"
            list_file.write_text(
                "\n".join(
                    [
                        "# comment",
                        "https://database.nikonoel.fr/lichess_elite_2025-12.zip",
                        "2026-01",
                        "https://database.nikonoel.fr/lichess_elite_2025-12.zip",  # duplicate
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    ".venv/bin/python",
                    "scripts/batch_prepare_elite_months.py",
                    "--list-file",
                    str(list_file),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        stdout = proc.stdout
        self.assertIn("'months': ['2025-12', '2026-01']", stdout)
        self.assertIn("scripts/acquire_and_prepare_elite_month.py", stdout)
        self.assertIn("'--max-samples-per-game', '8'", stdout)
        self.assertIn("'dry_run': True", stdout)

    def test_default_link_list_exists_and_uses_elite_zip_urls(self):
        path = Path("config/elite_month_validator_links.txt")
        self.assertTrue(path.is_file())
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        self.assertGreaterEqual(len(lines), 3)
        for line in lines:
            self.assertIn("database.nikonoel.fr/lichess_elite_", line)
            self.assertTrue(line.endswith(".zip"), line)


if __name__ == "__main__":
    unittest.main()
