#!/bin/bash
set -e

# Env vars:
#   MITM_PROXY_HOST   — hostname of the mitm-proxy container (default: mitm-proxy)
#   MITM_PROXY_PORT   — transparent proxy port (default: 8080)
#
# Auth is NOT done here — run scripts/headscale-setup.sh after the container starts.

mkdir -p /var/run/tailscale

echo "[tailscale] starting tailscaled..."
tailscaled \
    --state=/var/lib/tailscale/tailscaled.state \
    --socket=/var/run/tailscale/tailscaled.sock \
    &
TAILSCALED_PID=$!

echo "[tailscale] ready. Run scripts/headscale-setup.sh to authenticate."
echo "[tailscale] or manually: docker compose exec tailscale tailscale up --login-server=<URL> --authkey=<KEY> --advertise-exit-node"

wait $TAILSCALED_PID
