#!/usr/bin/env bash
# install-cert.sh — Install the mitmproxy CA cert into the system trust store.
# Run as root (sudo).
set -euo pipefail

CERT_SRC="data/mitmproxy/mitmproxy-ca-cert.pem"
CERT_DST="/usr/local/share/ca-certificates/mitmproxy-ca.crt"

if [[ ! -f "$CERT_SRC" ]]; then
  echo "ERROR: cert not found at $CERT_SRC"
  echo "Make sure the proxy has started at least once (it generates the cert on first run)."
  exit 1
fi

echo "==> Installing mitmproxy CA cert ..."
cp "$CERT_SRC" "$CERT_DST"
update-ca-certificates

echo "==> Cert installed. Browsers that use the system trust store will now trust mitmproxy."
echo "    Firefox uses its own store — import $CERT_SRC manually if needed."
