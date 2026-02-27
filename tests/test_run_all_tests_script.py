import unittest
from pathlib import Path


class RunAllTestsScriptTests(unittest.TestCase):
    def test_uses_project_venv_python_with_python3_fallback_and_pytest(self):
        text = Path("scripts/run_all_tests.sh").read_text(encoding="utf-8")
        self.assertIn('PY_BIN="${REPO_ROOT}/.venv/bin/python"', text)
        self.assertIn('PY_BIN="python3"', text)
        self.assertIn('"${PY_BIN}" -m pytest -q "$@"', text)


if __name__ == "__main__":
    unittest.main()
