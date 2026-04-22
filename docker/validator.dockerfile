FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    cmake \
    git \
    clang \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PHASE 1: Install Python dependencies
COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-cache

# PHASE 2: Build bitnet.cpp (llama.cpp fork with 1.58-bit kernels)
# Produces llama-server and llama-cli. We copy llama-server to /usr/local/bin.
#
# Switch BITNET_MODEL_REPO / BITNET_MODEL_FILE at build time to use
# Falcon3-7B-1.58bit instead of BitNet-2B — the default is the smaller/faster
# BitNet-2B-4T (best speed/quality tradeoff on 8-core CPU boxes).
ARG BITNET_MODEL_REPO=microsoft/BitNet-b1.58-2B-4T-gguf
ARG BITNET_MODEL_FILE=ggml-model-i2_s.gguf

RUN git clone --depth 1 https://github.com/microsoft/BitNet.git /tmp/bitnet && \
    cd /tmp/bitnet && \
    git submodule update --init --depth 1 && \
    mkdir -p /tmp/bitnet-model/BitNet-b1.58-2B-4T && \
    python setup_env.py -md /tmp/bitnet-model/BitNet-b1.58-2B-4T -q i2_s 2>&1 | grep -v '\^\['  || true && \
    sed -i 's/int8_t \* y_col = y + col \* by;/const int8_t * y_col = y + col * by;/' \
        src/ggml-bitnet-mad.cpp && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBITNET_X86_TL2=ON && \
    cmake --build build --target llama-server llama-cli -j"$(nproc)" && \
    cp build/bin/llama-server build/bin/llama-cli /usr/local/bin/ && \
    find build -name '*.so' -exec cp {} /usr/local/lib/ \; && \
    ldconfig && \
    rm -rf /tmp/bitnet /tmp/bitnet-model

# PHASE 3: Download models
# - WikiNEural (HF cache, public) for NER inference
# - BitNet GGUF weights for LLM validation
ENV HF_HOME=/app/hf_cache
ENV BITNET_MODEL=/app/hf_cache/bitnet/${BITNET_MODEL_FILE}

RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Babelscape/wikineural-multilingual-ner')" && \
    .venv/bin/python -c "from huggingface_hub import hf_hub_download; \
    import shutil, os; \
    p = hf_hub_download('${BITNET_MODEL_REPO}', '${BITNET_MODEL_FILE}'); \
    os.makedirs('/app/hf_cache/bitnet', exist_ok=True); \
    shutil.copy(p, '/app/hf_cache/bitnet/${BITNET_MODEL_FILE}')"

# PHASE 4: Copy source code (after heavy layers for cache stability)
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --no-cache

VOLUME ["/app/data", "/app/models"]

CMD ["poetry", "run", "python", "validator/main.py", "--daemon"]
