# Chess Bot Environment

## Container Assumption
Expected execution context is the containerized workspace rooted at `/work`.

## Container Verification
Run before assuming host and container environments are aligned:

```sh
test -f /run/.containerenv && echo containerenv=yes || echo containerenv=no
cat /proc/1/cgroup
cat /proc/1/comm
pwd
ls -a /work
ls /home/mintmainog/workspace/vs_code_workspace/chess_bot 2>/dev/null || echo "host path not present here"
```

Expected in this workspace:
- `/run/.containerenv` present
- `pwd` is `/work`
- host-local project path not present

If checks fail, warn the user immediately before proceeding.

## Python / Venv
- Canonical setup in container: `python3 -m venv --clear /work/.venv`
- If a newly created/reused venv lacks `pip`, bootstrap it with `python -m ensurepip --upgrade` before installs.
- Prefer module invocation for tooling reliability: `/work/.venv/bin/python -m pip ...`
- Externally-synced shim wrappers (for example `uv`-generated `pip`) may break due to absolute shebangs.

## Observed Smoke Runtime (2026-02-22)
- `/work/.venv` rebuilt in-container
- End-to-end CLI pipeline executed successfully on sample PGN fixture

## Gemini Sandbox Execution (2026-02-26)
When running in the Gemini sandbox (where `/.containerenv` is at root), the existing `.venv` may be broken due to Python version mismatches (e.g., symlinked to `python3.12` when system is `3.11`).

**Execution Strategy:**
Use system `python3` but inject the virtual environment's site-packages into `PYTHONPATH`.

**Example Commands:**
```bash
# General script execution
PYTHONPATH=.:.venv/lib/python3.12/site-packages python3 scripts/render_game_viewer.py --help

# Running tests
PYTHONPATH=.:.venv/lib/python3.12/site-packages python3 .venv/bin/pytest tests/test_runpod_api_helpers.py
```

**Known Limitations:**
- **Native Extensions:** Libraries with compiled C extensions (like `torch`) built for a different Python version (3.12) will fail with `ImportError` when loaded by the system Python (3.11).
- **Pure Python:** Pure Python libraries (like `chess`, `PyYAML`) work correctly with this strategy.

## GPU Availability Snapshot (2026-02-22)
- Container exposes NVIDIA device nodes (`/dev/nvidia*`) and `nvidia-smi` binary exists.
- GPU is not usable from this runtime:
  - `nvidia-smi` fails with `Failed to initialize NVML: Unknown Error`
  - `torch.cuda.is_available()` returns `False` (PyTorch build is `2.10.0+cu128`)
  - `ctypes` can load both `libcuda.so.1` and `libnvidia-ml.so.1` inside the container
  - PyTorch CUDA init warns with CUDA error `304` (`OS call failed or operation not supported on this OS`) and `Can't initialize NVML`
- Practical consequence: training runs CPU-only until container NVIDIA runtime/NVML access is fixed.
- Detailed diagnostic bundle for external debugging/AI triage:
  - `artifacts/reports/gpu_diagnostics_report_2026-02-22.md`
