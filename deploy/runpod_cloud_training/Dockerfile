FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG OTELCOL_VERSION=0.121.0

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
    iputils-ping \
    dnsutils \
    net-tools \
    sysstat \
    tmux \
    htop \
    openssl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) otel_arch="amd64" ;; \
      arm64) otel_arch="arm64" ;; \
      *) echo "unsupported architecture for otelcol-contrib: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/otelcol-contrib.tgz "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/otelcol-contrib_${OTELCOL_VERSION}_linux_${otel_arch}.tar.gz"; \
    tar -xzf /tmp/otelcol-contrib.tgz -C /tmp; \
    install -m 0755 /tmp/otelcol-contrib /usr/local/bin/otelcol-contrib; \
    rm -rf /tmp/otelcol-contrib /tmp/otelcol-contrib.tgz

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
RUN chmod +x /opt/runpod_cloud_training/healthchecks_ping.sh

EXPOSE 22 8888 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/runpod_cloud_training/entrypoint.sh"]
