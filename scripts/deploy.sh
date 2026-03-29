#!/usr/bin/env bash
# deploy.sh — Start (or restart) the Docker service on maxou.dev.
# /srv is NFS-mounted on both machines, so no file transfer needed.
set -euo pipefail

DEPLOY_DIR="/srv/Projects/freelance/mateo/claude-pii-proxy"
VPS="maxou.dev"

echo "==> Creating data dirs on VPS ..."
ssh "$VPS" "mkdir -p $DEPLOY_DIR/data/mitmproxy $DEPLOY_DIR/data/cache $DEPLOY_DIR/data/ignore"

echo "==> Building and starting Docker service on $VPS ..."
ssh "$VPS" "cd $DEPLOY_DIR && docker compose up -d --build"

echo ""
echo "==> Waiting for proxy to start ..."
sleep 5

echo "==> Service status:"
ssh "$VPS" "docker compose -f $DEPLOY_DIR/docker-compose.yml ps"

echo ""
echo "==> Done. Proxy running at $VPS:8080"
echo ""
echo "Next: install the CA cert on this machine:"
echo "  sudo bash $(dirname "$0")/install-cert.sh"
