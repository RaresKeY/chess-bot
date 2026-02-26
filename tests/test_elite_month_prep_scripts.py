import subprocess
import tempfile
import unittest
import zipfile
import importlib.util
from pathlib import Path
from unittest import mock


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

    def test_default_link_list_omits_known_unpublished_months(self):
        text = Path("config/elite_month_validator_links.txt").read_text(encoding="utf-8")
        self.assertNotIn("2025-12", text)
        self.assertNotIn("2026-01", text)

    def test_acquire_script_skips_completed_month_outputs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            validated = root / "validated"
            dataset = root / "dataset"
            validated.mkdir()
            dataset.mkdir()
            (validated / "summary.json").write_text("{}", encoding="utf-8")
            (validated / "valid_games.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            (dataset / "stats.json").write_text("{}", encoding="utf-8")
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                (dataset / name).write_text('{"x":1}\n', encoding="utf-8")

            proc = subprocess.run(
                [
                    ".venv/bin/python",
                    "scripts/acquire_and_prepare_elite_month.py",
                    "--month",
                    "2025-10",
                    "--validated-dir",
                    str(validated),
                    "--dataset-dir",
                    str(dataset),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("skip_existing_complete", proc.stdout)

    def test_acquire_script_fails_on_partial_existing_outputs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            validated = root / "validated"
            dataset = root / "dataset"
            validated.mkdir()
            (validated / "summary.json").write_text("{}", encoding="utf-8")
            # Missing dataset files => partial state, should fail unless --overwrite.
            proc = subprocess.run(
                [
                    ".venv/bin/python",
                    "scripts/acquire_and_prepare_elite_month.py",
                    "--month",
                    "2025-10",
                    "--validated-dir",
                    str(validated),
                    "--dataset-dir",
                    str(dataset),
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("does not look complete", proc.stderr + proc.stdout)


class EliteMonthDownloadHardeningTests(unittest.TestCase):
    @staticmethod
    def _load_module():
        path = Path("scripts/download_lichess_elite_month.py")
        spec = importlib.util.spec_from_file_location("download_lichess_elite_month", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_validate_zip_file_rejects_html_file(self):
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fake.zip"
            p.write_text("<html><body>Not found</body></html>", encoding="utf-8")
            with self.assertRaises(RuntimeError) as cm:
                mod.validate_zip_file(p, context="cached ZIP")
            self.assertIn("likely HTML/error page", str(cm.exception))

    def test_download_file_rejects_html_response_and_leaves_no_zip(self):
        mod = self._load_module()

        class _FakeResp:
            def __init__(self, payload: bytes):
                self._payload = payload
                self._offset = 0
                self.headers = {"Content-Length": str(len(payload)), "Content-Type": "text/html"}

            def read(self, n: int) -> bytes:
                if self._offset >= len(self._payload):
                    return b""
                chunk = self._payload[self._offset : self._offset + n]
                self._offset += len(chunk)
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "bad.zip"
            with mock.patch.object(mod, "urlopen", return_value=_FakeResp(b"<html>oops</html>")):
                with self.assertRaises(RuntimeError) as cm:
                    mod.download_file("https://example.invalid/fake.zip", out_path)
            self.assertIn("HTML, not ZIP", str(cm.exception))
            self.assertFalse(out_path.exists())
            self.assertFalse((Path(str(out_path) + ".part")).exists())

    def test_validate_zip_file_accepts_real_zip(self):
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.zip"
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr("a.txt", "hello")
            mod.validate_zip_file(p, context="test")


if __name__ == "__main__":
    unittest.main()
