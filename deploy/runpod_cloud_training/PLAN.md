# RunPod Cloud Training Deployment Plan

## Goal
Provide a modular RunPod deployment package for this repo that supports:
- startup repo clone/pull (public GitHub repo)
- prebuilt Python venv with startup requirements drift check/install
- SSH access for CLI workflows
- JupyterLab for exploration and file download
- HTTP inference API endpoint
- Hugging Face model sync (manual + automatic watcher)
- idle autostop watchdog for RunPod pods

## User-Selected Decisions
- Repo bootstrap: clone/pull at startup (optional, enabled by env default)
- Requirements sync: check and install if repo `requirements.txt` changed vs image runtime stamp
- Cloud sync target: Hugging Face (`huggingface_hub`) plus `rclone` installed for generic transfer
- Inference access: both SSH CLI and HTTP API endpoint
- Autostop: include in-container idle watchdog script
- Repo auth: public repo only (no clone token flow)

## Deliverables (module folder)
- `Dockerfile`
- `entrypoint.sh`
- `requirements-extra.txt`
- `env.example`
- `README.md`
- `inference_api.py`
- `hf_sync.py`
- `hf_auto_sync_watch.py`
- `idle_watchdog.py`

## Runtime Design
1. Container boots and runs `entrypoint.sh` under `tini`
2. Entry point configures SSH keys and starts `sshd`
3. Entry point clones or updates repo into `/workspace/chess-bot`
4. Entry point syncs Python requirements into prebuilt venv if repo requirements drifted
5. Entry point starts JupyterLab and inference API (optional toggles)
6. Entry point starts HF auto-sync watcher and idle watchdog (optional toggles)
7. All services write logs to stdout/stderr for RunPod log visibility

## Open/Optional Follow-Ups
- Add `--progress-only` training mode (optional local repo feature, not deployment-specific)
- Add TLS termination / reverse proxy if serving beyond trusted tunnel/private networking
- Add a small auth layer in front of inference API if publicly exposed
- Add S3/GCS sync helpers if needed later (rclone already installed)
