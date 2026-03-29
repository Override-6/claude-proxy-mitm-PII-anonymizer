#!/usr/bin/env bash
# local-proxy-teardown.sh — Remove the iptables redirect rules and stop redsocks.
# Run as root (sudo).
set -euo pipefail

echo "==> Removing iptables (IPv4) REDSOCKS chain ..."
iptables -t nat -D OUTPUT -p tcp --dport 443 -j REDSOCKS 2>/dev/null || true
iptables -t nat -F REDSOCKS 2>/dev/null || true
iptables -t nat -X REDSOCKS 2>/dev/null || true

echo "==> Restoring IPv6 :443 ..."
ip6tables -D OUTPUT -p tcp --dport 443 -j DROP 2>/dev/null || true

echo "==> Stopping redsocks ..."
systemctl stop redsocks 2>/dev/null || true
systemctl disable redsocks 2>/dev/null || true

if command -v netfilter-persistent &>/dev/null; then
  netfilter-persistent save
fi

echo "==> Done — direct :443 traffic restored."
