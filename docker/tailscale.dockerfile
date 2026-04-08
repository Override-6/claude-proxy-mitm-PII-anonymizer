FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    iptables \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Install tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

COPY docker/tailscale-entrypoint.sh /entrypoint.sh
COPY docker/tailscale-post-auth.sh /usr/local/bin/tailscale-post-auth
RUN chmod +x /entrypoint.sh /usr/local/bin/tailscale-post-auth

ENTRYPOINT ["/entrypoint.sh"]
