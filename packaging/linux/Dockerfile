FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHARDAI_HOME_DIR=/home/shardai/.shardai \
    SHARDAI_ENGINE_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash shardai

WORKDIR /opt/shardai
COPY core-engine /opt/shardai/core-engine

RUN python3 -m venv /opt/shardai/venv \
    && /opt/shardai/venv/bin/pip install --upgrade pip \
    && /opt/shardai/venv/bin/pip install --no-cache-dir -r /opt/shardai/core-engine/requirements.txt \
    && /opt/shardai/venv/bin/pip install --no-cache-dir --no-build-isolation --no-deps /opt/shardai/core-engine \
    && ln -sf /opt/shardai/venv/bin/shardai /usr/local/bin/shardai \
    && chown -R shardai:shardai /opt/shardai /home/shardai

USER shardai
WORKDIR /home/shardai

EXPOSE 8000 49152

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["shardai"]
CMD ["--help"]
