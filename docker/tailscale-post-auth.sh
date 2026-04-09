#!/bin/bash
# Runs inside the tailscale container after `tailscale up` succeeds.
# Sets up IP forwarding + DNAT to forward VPN traffic to the mitm-proxy container.
set -euo pipefail

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

echo "[post-auth] applying DNAT + MASQUERADE rules..."
# DNAT to loopback: mitmproxy runs in the same network namespace (network_mode: service:tailscale),
# so SO_ORIGINAL_DST works correctly and there is no cross-namespace conntrack problem.
iptables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 80  -j DNAT --to-destination "127.0.0.1:${MITM_PROXY_PORT}"
iptables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 443 -j DNAT --to-destination "127.0.0.1:${MITM_PROXY_PORT}"

# MASQUERADE for exit-node forwarding: packets from VPN clients leave via eth0
# with the container's real IP so the internet can reply.
# Skip MASQUERADE for traffic destined to 127.0.0.1 (already DNAT'd to mitmproxy).
iptables -t nat -A POSTROUTING ! -d 127.0.0.0/8 -o eth0 -j MASQUERADE

# Allow forward of exit-node traffic (kernel IP forwarding is already on via sysctl)
iptables -A FORWARD -i tailscale0 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o tailscale0 -m state --state RELATED,ESTABLISHED -j ACCEPT

echo "[post-auth] done — tailscale0 :80/:443 → DNAT → 127.0.0.1:${MITM_PROXY_PORT}, exit-node MASQUERADE on eth0"
