# vast_cloud_training

Vast.ai-specific deployment module for this repository.

## Scope
- Kept separate from `deploy/runpod_cloud_training/`.
- Uses Vast.ai API conventions and auth (`Bearer` token).
- Intended for SSH-based training orchestration with host-side cycle scripts.

## Related Scripts
- `scripts/vast_provision.py`
- `scripts/vast_cycle_common.sh`
- `scripts/vast_cycle_start.sh`
- `scripts/vast_cycle_stop.sh`
- `scripts/vast_cycle_terminate.sh`
- `scripts/vast_cycle_status.sh`
- `scripts/vast_cli_doctor.sh`
- `scripts/vast_local_smoke_test.sh`
- `scripts/vast_noauth_deploy_checks.sh`
- `scripts/vast_regression_checks.sh`

## Example Flow
```bash
bash scripts/vast_cycle_start.sh
bash scripts/vast_cycle_status.sh
bash scripts/vast_cycle_stop.sh
bash scripts/vast_cycle_terminate.sh
```

## No-Auth Local Deployment Smoke
Run the local Vast deployment smoke lane without `VAST_API_KEY`:

```bash
bash scripts/vast_noauth_deploy_checks.sh
```

This runs:
- Vast unit/regression tests (`test_vast*.py`)
- Vast provider local connectivity checks (`--no-live`)
- local smoke training through `deploy/vast_cloud_training/train_baseline_preset.sh`

## Environment
See `deploy/vast_cloud_training/env.example`.
