FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PHASE 1: Install dependencies
COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-cache

# PHASE 2: Download Gemma model into the image at build time.
# HF_TOKEN is a build arg — not persisted as an ENV, so it's absent at runtime.
# Model is stored at /app/hf_cache (separate from ~/.cache/huggingface so the
# runtime volume mount on that path doesn't shadow it).
ARG HF_TOKEN
ENV HF_HOME=/app/hf_cache
# Download all models needed at runtime so the container works fully offline.
# Gemma is gated (requires token); WikiNEural is public.
RUN HF_TOKEN=${HF_TOKEN} .venv/bin/python -c \
    "from huggingface_hub import snapshot_download; import os; \
     snapshot_download('google/gemma-3-1b-it', token=os.environ['HF_TOKEN']); \
     snapshot_download('Babelscape/wikineural-multilingual-ner')"

# PHASE 3: Copy source code (after model download to preserve layer cache)
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --no-cache

VOLUME ["/app/data", "/app/models"]

CMD ["poetry", "run", "python", "validator/main.py", "--daemon"]
