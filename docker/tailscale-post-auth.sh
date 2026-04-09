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

echo "[post-auth] applying iptables rules (IPv4 + IPv6)..."

# ── IPv4 ─────────────────────────────────────────────────────────────────────

# Block QUIC (HTTP/3 over UDP 443) so browsers fall back to TCP and hit mitmproxy.
# Must come before the ACCEPT rules below.
iptables -I FORWARD -i tailscale0 -p udp --dport 443 -j REJECT --reject-with icmp-port-unreachable

# DNAT TCP :80/:443 → mitmproxy.
# net.ipv4.conf.all.route_localnet=1 (set via docker-compose sysctls) allows
# DNAT to 127.0.0.1 from a non-loopback interface.
iptables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 80  -j DNAT --to-destination "127.0.0.1:${MITM_PROXY_PORT}"
iptables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 443 -j DNAT --to-destination "127.0.0.1:${MITM_PROXY_PORT}"

# MASQUERADE outbound exit-node traffic (skip loopback — already DNAT'd).
iptables -t nat -A POSTROUTING ! -d 127.0.0.0/8 -o eth0 -j MASQUERADE

# Allow exit-node forwarding.
iptables -A FORWARD -i tailscale0 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o tailscale0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# ── IPv6 ─────────────────────────────────────────────────────────────────────
# Best-effort: rules fail gracefully if Docker IPv6 is disabled or ip6table_nat
# is unavailable.

# Block QUIC (UDP 443) over IPv6 — same reason as IPv4.
ip6tables -I FORWARD -i tailscale0 -p udp --dport 443 -j REJECT --reject-with icmp6-port-unreachable 2>/dev/null || true

# MASQUERADE outbound IPv6 exit-node traffic.
ip6tables -t nat -A POSTROUTING -o eth0 -j MASQUERADE 2>/dev/null || true

# Allow exit-node IPv6 forwarding.
ip6tables -A FORWARD -i tailscale0 -o eth0 -j ACCEPT 2>/dev/null || true
ip6tables -A FORWARD -i eth0 -o tailscale0 -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true

# NOTE: IPv6 TCP :80/:443 REDIRECT to mitmproxy is intentionally omitted here.
# mitmproxy's [::]:8080 listener causes a startup race condition (DNS resolution
# failure on the shared network namespace before tailscale0 is fully up), which
# makes mitmproxy exit. Once that is resolved, add:
#   ip6tables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 80  -j REDIRECT --to-port "${MITM_PROXY_PORT}"
#   ip6tables -t nat -A PREROUTING -i tailscale0 -p tcp --dport 443 -j REDIRECT --to-port "${MITM_PROXY_PORT}"
# and add --mode transparent@[::]:8080 to the mitmproxy command.

echo "[post-auth] done — TCP :80/:443 → mitmproxy, QUIC (UDP 443) blocked, IPv6 rules applied where available"
