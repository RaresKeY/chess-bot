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
- `scripts/vast_regression_checks.sh`

## Example Flow
```bash
bash scripts/vast_cycle_start.sh
bash scripts/vast_cycle_status.sh
bash scripts/vast_cycle_stop.sh
bash scripts/vast_cycle_terminate.sh
```

## Environment
See `deploy/vast_cloud_training/env.example`.
