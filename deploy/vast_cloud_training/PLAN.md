# Plan

## Goals
- Provide a Vast.ai deployment path parallel to RunPod.
- Keep cloud-provider logic isolated by directory and script namespace.
- Preserve existing RunPod behavior.

## Current Components
- Provisioning/API helper: `scripts/vast_provision.py`
- Cycle helpers: `scripts/vast_cycle_*.sh`
- Diagnostics/regression entrypoints: `scripts/vast_cli_doctor.sh`, `scripts/vast_regression_checks.sh`
- Training preset entrypoint: `deploy/vast_cloud_training/train_baseline_preset.sh`

## Next Iterations
- Add full dataset push/train/collect/watch wrappers for Vast.
- Add provider-specific local smoke checks.
- Add richer instance status summarization for large fleets.
