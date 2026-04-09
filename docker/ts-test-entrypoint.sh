#!/bin/bash
set -e

# Env vars:
#   HEADSCALE_URL     — headscale coordination server URL
#   TS_AUTHKEY        — authkey to join the tailnet (reusable key)
#   TS_HOSTNAME       — tailscale node hostname (default: ts-test-curl)
#   TS_EXIT_NODE      — Tailscale IP of the exit node to use (default: 100.64.0.1)
#   MITM_CA_CERT_PATH — path to mitmproxy CA cert to trust (default: /etc/mitmproxy-ca/mitmproxy-ca-cert.pem)

MITM_CA_CERT_PATH="${MITM_CA_CERT_PATH:-/etc/mitmproxy-ca/mitmproxy-ca-cert.pem}"
TS_EXIT_NODE="${TS_EXIT_NODE:-100.64.0.1}"

# Install mitmproxy CA cert so HTTPS interception is transparent
if [ -f "${MITM_CA_CERT_PATH}" ]; then
    echo "[ts-test] installing mitmproxy CA cert..."
    cp "${MITM_CA_CERT_PATH}" /usr/local/share/ca-certificates/mitmproxy-ca.crt
    update-ca-certificates --fresh 2>&1 | grep -v '^$' || true
    echo "[ts-test] CA cert installed"
else
    echo "[ts-test] WARNING: CA cert not found at ${MITM_CA_CERT_PATH}, HTTPS will fail cert verification"
fi

mkdir -p /var/run/tailscale

echo "[ts-test] starting tailscaled..."
tailscaled \
    --state=/var/lib/tailscale/tailscaled.state \
    --socket=/var/run/tailscale/tailscaled.sock \
    --port=41641 \
    &
TAILSCALED_PID=$!

if [[ -n "${TS_AUTHKEY:-}" && -n "${HEADSCALE_URL:-}" ]]; then
    echo "[ts-test] waiting for tailscaled to be ready..."
    sleep 2

    echo "[ts-test] authenticating..."
    tailscale up \
        --login-server="${HEADSCALE_URL}" \
        --authkey="${TS_AUTHKEY}" \
        --accept-routes \
        --accept-dns=false \
        --exit-node="${TS_EXIT_NODE}" \
        --exit-node-allow-lan-access \
        --hostname="${TS_HOSTNAME:-ts-test-curl}"

    echo "[ts-test] ready — all traffic routed via exit node ${TS_EXIT_NODE}"
else
    echo "[ts-test] no TS_AUTHKEY/HEADSCALE_URL set — tailscaled started, authenticate manually"
fi

exec "$@"
wait $TAILSCALED_PID
