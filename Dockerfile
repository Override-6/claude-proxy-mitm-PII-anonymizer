FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Pip download cache persists across builds — torch/gliner/easyocr won't re-download
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download GLiNER model — HuggingFace cache persists across builds
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi-v2.1')"

# Pre-download EasyOCR English + French models — EasyOCR uses ~/.EasyOCR by default
RUN --mount=type=cache,target=/root/.EasyOCR \
    python -c "import easyocr; easyocr.Reader(['en', 'fr'], gpu=False)"

COPY . .

EXPOSE 8080

# mitmproxy cert + state persisted via volume
VOLUME ["/root/.mitmproxy", "/app/cache", "/app/ignore"]

CMD ["mitmdump", \
     "-s", "src/proxy.py", \
     "--listen-port", "8080", \
     "--listen-host", "0.0.0.0", \
     "--set", "console_eventlog_verbosity=info", \
     "--set", "flow_detail=0", \
     "--set", "script_reload=false"]
