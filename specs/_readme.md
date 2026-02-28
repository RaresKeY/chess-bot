# work - Specs Index

**Tech Stack**: Python

---

**IMPORTANT** Before making changes or researching any part of the codebase, use the table below to find and read the relevant spec first. This ensures you understand existing patterns and constraints.

## Documentation

Populate this table following template style.

| Spec | Code | Purpose |
|------|------|---------|
| [Commit Workflow](./commit_workflow.md) | `AGENTS.md`, git workflow tasks | Canonical commit process, safeguards, and skill routing. |
| [Chess Bot MVP Plan](./chess_bot_mvp_plan.md) | `implement.md`, `README.md`, `scripts/*.py`, `src/chessbot/*` | Umbrella chess-bot spec linking component specs, validation gates, and rollout notes. |
| [Chess Bot Overview](./chess_bot_overview.md) | `README.md`, `scripts/*.py`, `src/chessbot/*` | High-level scope, architecture pattern, decisions, and artifact contract. |
| [Chess Bot README Contract](./chess_bot_readme_contract.md) | `README.md` | Rules for component-oriented README sections and spec/README sync. |
| [Chess Bot Environment](./chess_bot_environment.md) | `AGENTS.md`, `.venv`, runtime checks | Container verification, venv guidance, and runtime assumptions. |
| [Chess Bot Telemetry](./chess_bot_telemetry.md) | `scripts/telemetry*.sh`, `scripts/runpod_cycle_*`, `deploy/runpod_cloud_training/otel-collector-config.yaml`, `deploy/runpod_cloud_training/entrypoint.sh` | Host telemetry components, checkpoint/watchdog contract, and container OpenTelemetry/healthchecks behavior. |
| [Chess Bot Cloud Connectivity Checks](./chess_bot_cloud_connectivity_checks.md) | `scripts/cloud_connectivity_health_checks.sh`, `scripts/cloud_checks/providers/*`, `scripts/runpod_connectivity_telemetry_checks.sh` | Modular timeout-guarded cloud connectivity/health/telemetry checks shared across RunPod/Vast provider modules. |
| [Chess Bot Secrets Contract](./chess_bot_secrets_contract.md) | `src/chessbot/secrets.py`, token-using scripts in `scripts/*`, `src/chessbot/lichess_bot.py` | Canonical secret/token resolution order, dotenv fallback rules, and security/testing contract. |
| [Chess Bot Regression Testing](./chess_bot_regression_testing.md) | `tests/*`, bug-fix changes across `scripts/*` and `src/chessbot/*` | Rules and checklist for adding focused regression tests after fixes. |
| [Chess Bot Validation](./chess_bot_validation.md) | `scripts/validate_games.py`, `src/chessbot/validation.py` | PGN replay validation behavior, outputs, and record schema. |
| [Chess Bot Splicing](./chess_bot_splicing.md) | `scripts/build_game_dataset.py`, `scripts/build_splice_dataset.py`, `src/chessbot/splicing.py`, `src/chessbot/phase.py` | Compact game-level dataset prep plus legacy splice sample generation, phase labeling, and game-level split guarantees. |
| [Chess Bot Training](./chess_bot_training.md) | `scripts/train_baseline.py`, `src/chessbot/training.py`, `src/chessbot/model.py`, `src/chessbot/phase.py` | Baseline model training behavior, winner/phase-aware loss weighting, and artifact format. |
| [Chess Bot Training Runbook](./chess_bot_training_runbook.md) | `scripts/train_baseline.py`, compact datasets under `data/dataset/*_game` | Operator runbook with copy/paste full-training and smoke-training commands plus required artifacts/inputs. |
| [Chess Bot Evaluation](./chess_bot_evaluation.md) | `scripts/eval_model.py`, `src/chessbot/evaluation.py`, `src/chessbot/phase.py` | Offline metrics, per-phase breakdowns, and legality-rate evaluation behavior. |
| [Chess Bot Model Olympics](./chess_bot_model_olympics.md) | `scripts/play_model_vs_model.py`, `artifacts/reports/*.json`, model artifacts in `artifacts/*` | Canonical participant list and repeatable head-to-head methodology for tracking model-vs-model comparisons. |
| [Chess Bot Inference](./chess_bot_inference.md) | `scripts/infer_move.py`, `src/chessbot/inference.py` | Top-k decoding and best-legal move selection behavior. |
| [Chess Bot HF Datasets](./chess_bot_hf_datasets.md) | `scripts/hf_dataset_publish.py`, `scripts/hf_dataset_fetch.py`, HF repo `LogicLark-QuantumQuill/chess-bot-datasets` | Canonical HF dataset layout, keyring/token auth contract, publish/fetch flow, and stale-version cleanup policy. |
| [Chess Bot Cloud Training Deployment](./chess_bot_cloud_training_deployment.md) | `deploy/runpod_cloud_training/*`, `scripts/runpod_provision.py`, `scripts/build_runpod_image.sh`, `scripts/runpod_local_smoke_test.sh` | RunPod-oriented Dockerized cloud module with SSH/Jupyter/inference API/HF sync/autostop watchdog plus API provisioning, image-build, and local smoke-test helpers. |
| [Chess Bot RunPod CLI Controls](./chess_bot_runpod_cli_controls.md) | `scripts/runpod_provision.py`, `scripts/runpod_cli_doctor.sh`, `scripts/runpod_quick_launch.sh`, `scripts/runpod_cycle_*.sh`, `config/runpod_gpu_types_catalog_*.json`, `artifacts/reports/runpod_cycle_observations_*.md`, host `ssh`/`rsync`/`scp`/`curl` workflows | Host-side CLI workflows for RunPod template discovery, GPU checks, provisioning, diagnostics, modular lifecycle runs (dataset push/train/collect/stop), and dated GPU/report snapshots. |
| [Chess Bot RunPod Preferred Training Flow](./chess_bot_runpod_preferred_training_flow.md) | `scripts/runpod_full_train_easy.sh`, `scripts/runpod_cycle_full_train_hf.sh`, `scripts/runpod_cycle_watch_progress.sh` | Canonical start-to-finish preferred RunPod full-training recipe (HF aggregate fetch, managed SSH key, cache-required runtime splice, and current exact run parameters). |
| [Chess Bot Vast.ai CLI Controls](./chess_bot_vast_cli_controls.md) | `scripts/vast_provision.py`, `scripts/vast_cli_doctor.sh`, `scripts/vast_regression_checks.sh`, `scripts/vast_cycle_*.sh`, `deploy/vast_cloud_training/*`, `config/vast_tracked_instances.jsonl` | Host-side Vast.ai workflows for offer search, provisioning, instance lifecycle controls, diagnostics, and isolated cloud-provider deployment module. |
| [Chess Bot Game Viewer](./chess_bot_game_viewer.md) | `scripts/render_game_viewer.py`, `src/chessbot/viewer.py`, `assets/pieces/cburnett/*` | HTML board viewer with left/right navigation and local piece assets. |
| [Chess Bot Play-vs-Model](./chess_bot_play_vs_model.md) | `main.py`, `scripts/play_vs_model_server.py`, `src/chessbot/play_vs_model.py` | Interactive browser UI + API server for playing against the model. |
| [Chess Bot Lichess Bot Play](./chess_bot_lichess_bot_play.md) | `scripts/lichess_bot.py`, `scripts/serve_lichess_preview.py`, `scripts/run_lichess_bot_with_preview.py`, `src/chessbot/lichess_bot.py`, `tests/test_lichess_bot.py` | Lichess Bot API integration with challenge/game streams, live/offline preview modes, combined wrapper flow, throttling, and tests. |
| [Chess Bot Viewer Server Utility](./chess_bot_viewer_server.md) | `scripts/serve_viewer.py` | Local HTTP server utility for viewing generated HTML and piece assets in-browser. |
