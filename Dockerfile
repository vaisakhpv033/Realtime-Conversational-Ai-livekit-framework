# syntax=docker/dockerfile:1
# LiveKit Voice Agent Framework Production Dockerfile
# Hardened multi-stage build with UV and Python 3.13 on Debian Bookworm Slim

ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ==============================================================================
# 1. Builder Stage: Compile dependencies, build project, and pre-cache models
# ==============================================================================
FROM base AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications and project readme to leverage Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies deterministically (without installing the project itself yet)
RUN uv sync --locked --no-dev --no-install-project

# Copy remaining source files
COPY . .

# Install the project and pre-compile bytecode
RUN uv sync --locked --no-dev

# Pre-cache model weights (e.g., Silero VAD and Google Live plugin assets)
RUN uv run -m livekit.agents download-files

# ==============================================================================
# 2. Production Runner Stage: Minimal attack surface, unprivileged execution
# ==============================================================================
FROM base AS runner

# Install essential runtime utilities: curl (health checks), ca-certificates (TLS), tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tzdata \
  && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user appuser
ARG UID=10001
RUN useradd -u ${UID} -d /home/appuser -m -s /sbin/nologin appuser

# Set runtime environment
ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/.cache/uv \
    XDG_CACHE_HOME=/tmp/.cache

# Ensure cache directories exist with appropriate unprivileged user permissions
RUN mkdir -p /tmp/.cache/uv /home/appuser/.cache && chown -R appuser:appuser /tmp/.cache /home/appuser

WORKDIR /app

# Single-layer copy of application and virtualenv with correct ownership
COPY --from=build --chown=appuser:appuser /app /app

# Internal HTTP health check endpoint exposed by AgentServer
EXPOSE 8081

# Standardized health probe against local health check server
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8081/ || exit 1

# Graceful termination signal handling for call draining
STOPSIGNAL SIGTERM

# Switch to unprivileged runtime user
USER appuser

# Start the LiveKit agent worker via CLI entrypoint
CMD ["agent-framework", "start"]
