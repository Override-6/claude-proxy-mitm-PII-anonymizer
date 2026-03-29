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
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download GLiNER model so the first request isn't slow
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi-v2.1')"

# Pre-download EasyOCR English model
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

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
