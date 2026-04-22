#!/bin/bash
# Env vars:
#   MITM_PROXY_PORT — transparent proxy port (default: 8080)
set -e

MITM_PROXY_PORT="${MITM_PROXY_PORT:-8080}"

echo "[wireguard] bringing up wg0..."
# Table=off prevents wg-quick from adding a default route through the tunnel,
# which would loop mitmproxy's outbound traffic back into the VPN.
wg-quick up /etc/wireguard/wg0.conf

echo "[wireguard] waiting for wg0..."
for i in $(seq 1 30); do
    if ip link show wg0 &>/dev/null; then
        echo "[wireguard] wg0 is up"
        break
    fi
    sleep 1
done

if ! ip link show wg0 &>/dev/null; then
    echo "[wireguard] ERROR: wg0 never appeared" >&2
    exit 1
fi

echo "[wireguard] applying iptables rules (IPv4)..."

# Block QUIC (HTTP/3 over UDP 443) so clients fall back to TCP and hit mitmproxy.
iptables -I FORWARD -i wg0 -p udp --dport 443 -j REJECT --reject-with icmp-port-unreachable

# DNAT TCP :80/:443 → mitmproxy (transparent mode).
# net.ipv4.conf.all.route_localnet=1 (set via compose sysctls) allows DNAT to 127.x.x.x
# from a non-loopback interface.
iptables -t nat -A PREROUTING -i wg0 -p tcp --dport 80  -j DNAT --to-destination "127.0.0.1:${MITM_PROXY_PORT}"
iptables -t nat -A PREROUTING -i wg0 -p tcp --dport 443 -j DNAT --to-destination "127.0.0.1:${MITM_PROXY_PORT}"

# MASQUERADE outbound exit-node traffic via the container's eth0 (skip loopback — already DNAT'd).
iptables -t nat -A POSTROUTING ! -d 127.0.0.0/8 -o eth0 -j MASQUERADE

# Allow exit-node forwarding.
iptables -A FORWARD -i wg0 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o wg0 -m state --state RELATED,ESTABLISHED -j ACCEPT

echo "[wireguard] done — TCP :80/:443 → mitmproxy, QUIC (UDP 443) blocked"

# Keep container alive — mitm-proxy shares this network namespace.
exec sleep infinity
