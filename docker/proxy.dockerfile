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

COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install poetry

RUN --mount=type=cache,target=/root/.cache/poetry \
    poetry install

# Pre-download EasyOCR English + French models — EasyOCR uses ~/.EasyOCR by default
RUN --mount=type=cache,target=/root/.EasyOCR \
    poetry run python -c "import easyocr; easyocr.Reader(['en', 'fr'], gpu=False)"

COPY . .

EXPOSE 8080

# mitmproxy cert + state persisted via volume
VOLUME ["/root/.mitmproxy", "/app/cache", "/app/ignore"]

CMD ["poetry", "run", "mitmdump", \
     "-s", "src/proxy.py", \
     "--listen-port", "8080", \
     "--listen-host", "0.0.0.0", \
     "--set", "console_eventlog_verbosity=info", \
     "--set", "flow_detail=0", \
     "--set", "script_reload=false"]
