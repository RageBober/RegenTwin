# syntax=docker/dockerfile:1.7
# RegenTwin backend-only image (FastAPI + DuckDB).
# Frontend (Tauri) собирается отдельно через GitHub Actions release.yml.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.5.11

# =============================================================================
# Stage 1: builder — компиляция numpy/scipy/PyMC и сборка .venv
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv: pinned binary copy from official image (no pip needed)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Layer-cache friendly: lock files copied first.
COPY pyproject.toml uv.lock ./

# Install runtime deps only (no dev/docs groups).
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself (without re-resolving deps).
COPY src/ ./src/
COPY alembic.ini ./
RUN uv sync --frozen --no-dev


# =============================================================================
# Stage 2: runtime — slim image, только runtime зависимости
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    REGENTWIN_DATABASE_URL="duckdb:///app/data/regentwin.duckdb"

RUN apt-get update && apt-get install --no-install-recommends -y \
        libopenblas0 \
        liblapack3 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 regentwin

WORKDIR /app

COPY --from=builder --chown=regentwin:regentwin /app/.venv /app/.venv
COPY --chown=regentwin:regentwin src/ ./src/
COPY --chown=regentwin:regentwin alembic.ini ./

# Runtime artefacts: data volume, logs, bench output
RUN mkdir -p /app/data /app/logs /app/output \
    && chown -R regentwin:regentwin /app

USER regentwin

EXPOSE 8000

# Healthcheck (Docker engine, не FastAPI — отдельно от /api/v1/health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/v1/health/live || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
