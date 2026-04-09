FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    iptables \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Install tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

COPY docker/ts-test-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
