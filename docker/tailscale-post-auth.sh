#!/bin/bash
# Runs inside the tailscale container after `tailscale up` succeeds.
# Sets up IP forwarding + DNAT to forward VPN traffic to the mitm-proxy container.
set -euo pipefail

MITM_PROXY_HOST="${MITM_PROXY_HOST:-mitm-proxy}"
MITM_PROXY_PORT="${MITM_PROXY_PORT:-8080}"

echo "[post-auth] waiting for tailscale0..."
for i in $(seq 1 30); do
    if ip link show tailscale0 &>/dev/null; then
        echo "[post-auth] tailscale0 is up"
        break
    fi
    sleep 1
done

if ! ip link show tailscale0 &>/dev/null; then
    echo "[post-auth] ERROR: tailscale0 never appeared" >&2
    exit 1
fi

echo "[post-auth] resolving ${MITM_PROXY_HOST}..."
MITM_IP=$(getent hosts "${MITM_PROXY_HOST}" | awk '{print $1}' | head -1)
if [[ -z "$MITM_IP" ]]; then
    echo "[post-auth] ERROR: could not resolve ${MITM_PROXY_HOST}" >&2
    exit 1
fi
echo "[post-auth] mitm-proxy IP: ${MITM_IP}"

echo "[post-auth] enabling IP forwarding..."
echo 1 > /proc/sys/net/ipv4/ip_forward

echo "[post-auth] applying DNAT rules..."
iptables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 80  -j DNAT --to-destination "${MITM_IP}:${MITM_PROXY_PORT}"
iptables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 443 -j DNAT --to-destination "${MITM_IP}:${MITM_PROXY_PORT}"

# Allow forwarding of DNAT'd packets to the mitm-proxy container
iptables -A FORWARD -i tailscale0 -d "${MITM_IP}" -p tcp --dport "${MITM_PROXY_PORT}" -j ACCEPT
iptables -A FORWARD -s "${MITM_IP}" -o tailscale0 -j ACCEPT

echo "[post-auth] done — tailscale0 :80/:443 → DNAT → ${MITM_IP}:${MITM_PROXY_PORT}"
