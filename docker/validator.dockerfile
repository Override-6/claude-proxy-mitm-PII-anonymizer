# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PHASE 1: Install dependencies (only when pyproject.toml/poetry.lock changes)
# This layer is independent of source code, so it's cached even if code changes
COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-cache

# PHASE 2: Copy source code (doesn't invalidate dependency cache thanks to .dockerignore)
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --no-cache

VOLUME ["/app/data", "/app/models", "/root/.cache/huggingface"]

CMD ["poetry", "run", "python", "validator/main.py", "--daemon"]
