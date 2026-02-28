FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_DIR=/opt/venvs/chessbot \
    REPO_DIR=/workspace/chess-bot

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    git-lfs \
    openssh-server \
    rsync \
    rclone \
    curl \
    wget \
    jq \
    pigz \
    zstd \
    lz4 \
    pv \
    aria2 \
    unzip \
    ca-certificates \
    tini \
    procps \
    iproute2 \
    tmux \
    htop \
    openssl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash runner \
    && passwd -d runner \
    && mkdir -p /var/run/sshd /workspace /opt/runpod_cloud_training \
    && chown -R runner:runner /workspace /home/runner

RUN python3 -m venv "${VENV_DIR}" \
    && "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

COPY deploy/runpod_cloud_training/requirements-extra.txt /tmp/requirements-extra.txt
RUN "${VENV_DIR}/bin/pip" install -r /tmp/requirements-extra.txt

COPY deploy/runpod_cloud_training /opt/runpod_cloud_training

RUN chmod +x /opt/runpod_cloud_training/entrypoint.sh

EXPOSE 22 8888 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/runpod_cloud_training/entrypoint.sh"]
