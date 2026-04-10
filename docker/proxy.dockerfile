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
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PHASE 1: Install dependencies (only when pyproject.toml/poetry.lock changes)
# This layer is independent of source code, so it's cached even if code changes
COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true && \
    poetry lock && \
    poetry install --no-root --no-cache

# PHASE 2: Copy source code (doesn't invalidate dependency cache thanks to .dockerignore)
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --no-cache

# paddlepaddle has platform-specific wheels that poetry can't resolve — install via pip
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    poetry run pip install paddlepaddle

# Pre-download models (cached on host via docker-compose volume mounts)
RUN --mount=type=cache,target=/root/.cache/huggingface,sharing=locked \
    poetry run python -m spacy download en_core_web_lg || true


EXPOSE 8080

# mitmproxy cert + state persisted via volume
VOLUME ["/root/.mitmproxy", "/app/cache", "/app/ignore"]

CMD ["poetry", "run", "mitmdump", \
     "-s", "proxy/main.py", \
     "--listen-port", "8080", \
     "--listen-host", "0.0.0.0", \
     "--set", "console_eventlog_verbosity=info", \
     "--set", "flow_detail=0", \
     "--set", "script_reload=false"]
