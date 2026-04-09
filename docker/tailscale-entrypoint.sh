#!/bin/bash
set -e

# Env vars:
#   MITM_PROXY_HOST   — hostname of the mitm-proxy container (default: mitm-proxy)
#   MITM_PROXY_PORT   — transparent proxy port (default: 8080)
#   TS_AUTHKEY        — if set, authenticate automatically on startup
#   HEADSCALE_URL     — headscale coordination server URL
#   TS_HOSTNAME       — tailscale node hostname

MITM_PROXY_PORT="${MITM_PROXY_PORT:-8080}"

mkdir -p /var/run/tailscale

echo "[tailscale] starting tailscaled..."
tailscaled \
    --state=/var/lib/tailscale/tailscaled.state \
    --socket=/var/run/tailscale/tailscaled.sock \
    &
TAILSCALED_PID=$!

# If auth key provided, authenticate automatically
if [[ -n "${TS_AUTHKEY:-}" && -n "${HEADSCALE_URL:-}" ]]; then
    echo "[tailscale] waiting for tailscaled to be ready..."
    sleep 2
    echo "[tailscale] authenticating automatically..."
    tailscale up \
        --login-server="${HEADSCALE_URL}" \
        --authkey="${TS_AUTHKEY}" \
        --advertise-exit-node \
        --accept-routes \
        --accept-dns=false \
        --hostname="${TS_HOSTNAME:-tailscale-exit}"
    echo "[tailscale] authenticated, applying DNAT rules..."
    tailscale-post-auth
else
    echo "[tailscale] ready. Run scripts/headscale-setup.sh to authenticate."
fi

wait $TAILSCALED_PID
