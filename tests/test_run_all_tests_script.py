import unittest
from pathlib import Path


class RunAllTestsScriptTests(unittest.TestCase):
    def test_test_sh_uses_project_venv_python_with_python3_fallback_and_pytest(self):
        text = Path("scripts/test.sh").read_text(encoding="utf-8")
        self.assertIn('PY_BIN="${REPO_ROOT}/.venv/bin/python"', text)
        self.assertIn('PY_BIN="python3"', text)
        self.assertIn('"${PY_BIN}" -m pytest -q "$@"', text)

    def test_run_all_tests_sh_forwards_to_test_sh(self):
        text = Path("scripts/run_all_tests.sh").read_text(encoding="utf-8")
        self.assertIn('forwarding to scripts/test.sh', text)
        self.assertIn('exec bash "${REPO_ROOT}/scripts/test.sh" "$@"', text)


if __name__ == "__main__":
    unittest.main()
