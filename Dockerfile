# syntax=docker/dockerfile:1
# Headless robits: non-root, read-only root filesystem, all state on /data.
#   docker build -t robits .
#   docker run --read-only -v robits-data:/data -e OPENAI_BASE_URL=... robits
# Optional extras: --build-arg EXTRAS="sqlite-vec" (vector memory search).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
FROM ${PYTHON_IMAGE}

ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-core.txt ./
RUN pip install -r requirements-core.txt ${EXTRAS}
COPY main.py tools.yaml personas.yaml ./
COPY robits/ robits/
RUN python -m compileall -q /app \
    && useradd --uid 10001 --home-dir /data --no-create-home --shell /usr/sbin/nologin robits \
    && mkdir /data && chown 10001:10001 /data

# Relative paths (logs/, var/, .wait_state.json) land on /data too, via WORKDIR.
ENV HOME=/data \
    PYTHONPATH=/app \
    ROBITS_HEADLESS=1 \
    ROBITS_PROVIDER_API=chat \
    ROBITS_PERSONAS_FILE=/app/personas.yaml \
    ROBITS_MEMORY_DB=/data/memory.db \
    ROBITS_AGENT_WORKSPACE_ROOT=/data/agents \
    ROBITS_TOOL_PROPOSALS_FILE=/data/tool_proposals.json \
    ROBITS_HEARTBEAT_FILE=/data/heartbeat
USER 10001:10001
WORKDIR /data
VOLUME ["/data"]
ENTRYPOINT ["python", "/app/main.py", "--headless"]
