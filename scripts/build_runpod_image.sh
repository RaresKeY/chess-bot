#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/deploy/runpod_cloud_training/Dockerfile"

IMAGE_REPO="${IMAGE_REPO:-}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_LATEST_TAG="${IMAGE_LATEST_TAG:-latest}"
PUSH_IMAGE="${PUSH_IMAGE:-0}"
PLATFORM="${PLATFORM:-linux/amd64}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${REPO_ROOT}}"

usage() {
  cat <<'EOF'
Build/push the RunPod cloud-training image for this repo.

Env vars:
  IMAGE_REPO        Required. Example: ghcr.io/<user>/chess-bot-runpod
  IMAGE_TAG         Optional. Default: git short SHA (or timestamp fallback)
  IMAGE_LATEST_TAG  Optional. Default: latest
  PUSH_IMAGE        Optional. 0=local build (default), 1=buildx --push
  PLATFORM          Optional. Default: linux/amd64
  BUILD_CONTEXT     Optional. Default: repo root

Examples:
  IMAGE_REPO=ghcr.io/me/chess-bot-runpod bash scripts/build_runpod_image.sh
  IMAGE_REPO=ghcr.io/me/chess-bot-runpod PUSH_IMAGE=1 bash scripts/build_runpod_image.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ -z "${IMAGE_REPO}" ]]; then
  echo "[runpod-image] IMAGE_REPO is required" >&2
  usage >&2
  exit 1
fi

IMAGE_REPO_ORIG="${IMAGE_REPO}"
IMAGE_REPO="${IMAGE_REPO,,}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "[runpod-image] Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

if [[ -z "${IMAGE_TAG}" ]]; then
  if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    IMAGE_TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
  else
    IMAGE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  fi
fi

TAG1="${IMAGE_REPO}:${IMAGE_TAG}"
TAG2="${IMAGE_REPO}:${IMAGE_LATEST_TAG}"

if [[ "${IMAGE_REPO}" != "${IMAGE_REPO_ORIG}" ]]; then
  echo "[runpod-image] normalized IMAGE_REPO to lowercase: ${IMAGE_REPO_ORIG} -> ${IMAGE_REPO}"
fi
echo "[runpod-image] repo=${IMAGE_REPO}"
echo "[runpod-image] tags=${TAG1}, ${TAG2}"
echo "[runpod-image] dockerfile=${DOCKERFILE}"
echo "[runpod-image] context=${BUILD_CONTEXT}"
echo "[runpod-image] platform=${PLATFORM}"
echo "[runpod-image] push=${PUSH_IMAGE}"

if [[ "${PUSH_IMAGE}" == "1" ]]; then
  cmd=(
    docker buildx build
    --platform "${PLATFORM}"
    -f "${DOCKERFILE}"
    -t "${TAG1}"
    -t "${TAG2}"
    --push
    "${BUILD_CONTEXT}"
  )
else
  cmd=(
    docker build
    -f "${DOCKERFILE}"
    -t "${TAG1}"
    -t "${TAG2}"
    "${BUILD_CONTEXT}"
  )
fi

printf '[runpod-image] exec:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
