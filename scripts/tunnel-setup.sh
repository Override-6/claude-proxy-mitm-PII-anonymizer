#!/usr/bin/env bash
# tunnel-setup.sh — Keep an SSH tunnel to maxou.dev:8080 alive via systemd.
# Run as the user (not root).
set -euxo pipefail

UNIT="claude-proxy-tunnel"
SERVICE_FILE="$HOME/.config/systemd/user/${UNIT}.service"

mkdir -p "$HOME/.config/systemd/user"

cat > "$SERVICE_FILE" <<UNIT
[Unit]
Description=SSH tunnel to claude-mitm-proxy on maxou.dev
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 127.0.0.1:8080:127.0.0.1:8080 -L 127.0.0.1:9999:127.0.0.1:9999 maxou.dev
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

echo PLEASE DONT RUN THIS SCRIPT AS ROOT OR THE FOLLOWING COMMAND WILL FAIL
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT"
echo "==> Tunnel service started. Check: systemctl --user status $UNIT"
