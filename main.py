import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAY_SERVER_SCRIPT = ROOT / "scripts" / "play_vs_model_server.py"


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _find_latest_model(artifacts_dir: Path) -> Path:
    candidates = [p for p in artifacts_dir.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise SystemExit(
            f"No model artifacts found under {artifacts_dir}. "
            "Train a model first or pass --model PATH."
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def main() -> None:
    argv = sys.argv[1:]

    # Preserve native server help output without requiring a local model artifact.
    if "-h" in argv or "--help" in argv:
        launch_argv = ["scripts/play_vs_model_server.py", *argv]
    else:
        launch_argv = ["scripts/play_vs_model_server.py"]
        if not _has_option(argv, "--dir"):
            launch_argv.extend(["--dir", str(ROOT)])
        if not _has_option(argv, "--model"):
            latest_model = _find_latest_model(ROOT / "artifacts")
            print(f"Launching play-vs-model with latest model: {latest_model}")
            launch_argv.extend(["--model", str(latest_model)])
        launch_argv.extend(argv)

    original_argv = sys.argv[:]
    try:
        sys.argv = launch_argv
        runpy.run_path(str(PLAY_SERVER_SCRIPT), run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
