#!/usr/bin/env bash
# local-proxy-setup.sh — Route all outbound :443 traffic through the mitmproxy
# running on maxou.dev:8080.
#
# Uses redsocks to convert transparently-intercepted TCP connections into
# HTTP CONNECT requests that mitmproxy understands.
# Handles both IPv4 and IPv6 outbound traffic.
#
# Run as root (sudo).
set -euo pipefail

PROXY_HOST="127.0.0.1"   # reached via SSH tunnel (see below)
PROXY_PORT=8080
REDSOCKS_PORT=12345
REDSOCKS_CONF="/etc/redsocks.conf"
REDSOCKS_SYSD="/etc/systemd/system/redsocks.service"

# ---- install redsocks if missing ----------------------------------------
if ! command -v redsocks &>/dev/null; then
  echo "==> Installing redsocks ..."
  apt-get update -q && apt-get install -y redsocks
fi

# ---- resolve VPS IP (used to exclude it from iptables redirect) ---------
VPS_HOSTNAME="maxou.dev"
VPS_IP4=$(getent ahostsv4 "$VPS_HOSTNAME" | awk '{ print $1 }' | head -1)

if [[ -z "$VPS_IP4" ]]; then
  echo "ERROR: cannot resolve IPv4 for $VPS_HOSTNAME"
  exit 1
fi
echo "==> VPS IPv4: $VPS_IP4 (excluded from redirect)"
echo "==> Proxy reachable via SSH tunnel at 127.0.0.1:$PROXY_PORT"

# ---- write redsocks config -----------------------------------------------
cat > "$REDSOCKS_CONF" <<EOF
base {
    log_debug = off;
    log_info = on;
    log = "syslog:daemon";
    daemon = on;
    redirector = iptables;
}

redsocks {
    local_ip  = 127.0.0.1;
    local_port = $REDSOCKS_PORT;
    ip   = $PROXY_HOST;
    port = $PROXY_PORT;
    type = http-connect;
}
EOF

# ---- systemd unit for redsocks (if missing) ------------------------------
if [[ ! -f "$REDSOCKS_SYSD" ]]; then
  cat > "$REDSOCKS_SYSD" <<'EOF'
[Unit]
Description=Redsocks transparent proxy redirector
After=network.target

[Service]
Type=forking
ExecStart=/usr/sbin/redsocks -c /etc/redsocks.conf
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
fi

systemctl enable --now redsocks
systemctl restart redsocks
echo "==> redsocks started"

# ---- iptables NAT rules (IPv4) -------------------------------------------
iptables -t nat -F REDSOCKS 2>/dev/null || true
iptables -t nat -D OUTPUT -p tcp --dport 443 -j REDSOCKS 2>/dev/null || true
iptables -t nat -X REDSOCKS 2>/dev/null || true

iptables -t nat -N REDSOCKS

# Exclude loopback and private ranges
iptables -t nat -A REDSOCKS -d 0.0.0.0/8      -j RETURN
iptables -t nat -A REDSOCKS -d 10.0.0.0/8     -j RETURN
iptables -t nat -A REDSOCKS -d 127.0.0.0/8    -j RETURN
iptables -t nat -A REDSOCKS -d 169.254.0.0/16 -j RETURN
iptables -t nat -A REDSOCKS -d 172.16.0.0/12  -j RETURN
iptables -t nat -A REDSOCKS -d 192.168.0.0/16 -j RETURN

# Exclude the VPS itself to avoid a redirect loop
iptables -t nat -A REDSOCKS -d "$VPS_IP4/32"  -j RETURN

# Redirect remaining :443 to redsocks IPv4 listener
iptables -t nat -A REDSOCKS -p tcp --dport 443 -j REDIRECT --to-ports "$REDSOCKS_PORT"

iptables -t nat -A OUTPUT -p tcp --dport 443 -j REDSOCKS

echo "==> iptables (IPv4) rules applied"

# ---- ip6tables: block outbound IPv6 :443 so apps fall back to IPv4 ------
# The VPS has no IPv6, so we can't tunnel IPv6 traffic through it.
# Dropping IPv6 :443 causes dual-stack apps to retry over IPv4 (via the proxy).
ip6tables -D OUTPUT -p tcp --dport 443 -j DROP 2>/dev/null || true
ip6tables -A OUTPUT -p tcp --dport 443 -j DROP

echo "==> ip6tables: outbound IPv6 :443 blocked (forces IPv4 fallback via proxy)"

# ---- persist rules across reboots ----------------------------------------
if command -v netfilter-persistent &>/dev/null; then
  netfilter-persistent save
  echo "==> rules persisted via netfilter-persistent"
elif command -v iptables-save &>/dev/null; then
  mkdir -p /etc/iptables
  iptables-save  > /etc/iptables/rules.v4
  ip6tables-save > /etc/iptables/rules.v6
  echo "==> rules saved to /etc/iptables/rules.v{4,6}"
fi

echo ""
echo "==> Setup complete — all outbound :443 routed via $PROXY_HOST:$PROXY_PORT"
echo "    Verify: curl -v https://api.anthropic.com/v1/models (should show mitmproxy cert)"
echo ""
echo "To UNDO (disable proxy routing):"
echo "  sudo bash $(dirname "$0")/local-proxy-teardown.sh"
