FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Clone at build time; rebuild the image to pick up upstream changes.
RUN git clone --depth 1 https://github.com/taylorwilsdon/google_workspace_mcp.git .

RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "main.py", "--transport", "streamable-http"]
