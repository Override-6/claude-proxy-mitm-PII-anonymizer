#!/bin/bash
# headscale-setup.sh — Bootstrap headscale + tailscale auth
#
# Usage:
#   ./scripts/headscale-setup.sh [--user <username>]
#
# What it does:
#   1. Ensures headscale and tailscale containers are running
#   2. Creates a headscale user (idempotent)
#   3. Creates a reusable pre-auth key (168h)
#   4. Execs `tailscale up` directly into the running tailscale container
#   5. Execs the post-auth script (iptables + IP forwarding) inside the container
#   6. Waits for the node to appear in headscale
#   7. Approves exit-node routes

set -euo pipefail

HEADSCALE_USER="myuser"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --user) HEADSCALE_USER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

HEADSCALE_URL="${HEADSCALE_URL:-https://vpn.maxou.dev}"
TS_HOSTNAME="${TS_HOSTNAME:-tailscale-exit}"
COMPOSE="docker compose"

cd "$(dirname "$0")/.."

echo "=== Headscale + Tailscale setup ==="
echo "Headscale user : ${HEADSCALE_USER}"
echo "Headscale URL  : ${HEADSCALE_URL}"
echo "Node hostname  : ${TS_HOSTNAME}"
echo ""

# ── 1. Ensure containers are running ────────────────────────────────────────
echo "[1/6] Starting headscale and tailscale containers..."
$COMPOSE up -d headscale tailscale
sleep 3

for svc in headscale tailscale; do
    if ! $COMPOSE ps "$svc" 2>/dev/null | grep -qE "running|Up"; then
        echo "ERROR: $svc container failed to start" >&2
        echo "       docker compose logs $svc" >&2
        exit 1
    fi
done
echo "      both containers are up"

# ── 2. Create headscale user (idempotent) ────────────────────────────────────
echo "[2/6] Ensuring headscale user '${HEADSCALE_USER}' exists..."
if $COMPOSE exec -T headscale headscale users list 2>/dev/null | grep -qE "[[:space:]]${HEADSCALE_USER}[[:space:]]"; then
    echo "      user already exists"
else
    $COMPOSE exec -T headscale headscale users create "${HEADSCALE_USER}"
    echo "      user created"
fi

# ── 3. Generate pre-auth key ─────────────────────────────────────────────────
echo "[3/6] Generating pre-auth key (reusable, 168h)..."
AUTHKEY=$($COMPOSE exec -T headscale headscale preauthkeys create \
    --user "${HEADSCALE_USER}" \
    --reusable \
    --expiration 168h \
    | grep -oE '[a-f0-9]{40,}' | head -1)

if [[ -z "$AUTHKEY" ]]; then
    echo "ERROR: could not extract auth key from headscale output" >&2
    exit 1
fi
echo "      key: ${AUTHKEY:0:8}…"

# ── 4. Run tailscale up inside the running container ─────────────────────────
echo "[4/6] Authenticating tailscale (exec into container)..."
$COMPOSE exec -T tailscale tailscale up \
    --login-server="${HEADSCALE_URL}" \
    --authkey="${AUTHKEY}" \
    --advertise-exit-node \
    --accept-routes \
    --hostname="${TS_HOSTNAME}"
echo "      tailscale up done"

# ── 5. Apply iptables rules inside the container ────────────────────────────
echo "[5/6] Applying iptables + IP forwarding inside tailscale container..."
$COMPOSE exec -T tailscale tailscale-post-auth
echo "      iptables rules applied"

# ── 6. Verify node registration + approve routes ─────────────────────────────
echo "[6/6] Verifying node registration in headscale..."
NODE_FOUND=0
for i in $(seq 1 30); do
    if $COMPOSE exec -T headscale headscale nodes list 2>/dev/null | grep -q "${TS_HOSTNAME}"; then
        NODE_FOUND=1
        break
    fi
    printf "      attempt %d/30…\r" "$i"
    sleep 1
done
echo ""

if [[ $NODE_FOUND -eq 0 ]]; then
    echo "ERROR: node '${TS_HOSTNAME}' did not appear in headscale after 30s" >&2
    echo "       docker compose logs tailscale" >&2
    exit 1
fi
echo "      node '${TS_HOSTNAME}' registered"

echo "      approving exit-node routes..."
ROUTE_IDS=$($COMPOSE exec -T headscale headscale routes list 2>/dev/null \
    | awk '/0\.0\.0\.0\/0|::\/0/ {print $1}')

if [[ -z "$ROUTE_IDS" ]]; then
    echo "      no routes yet — re-run to approve later, or:"
    echo "      docker compose exec headscale headscale routes list"
    echo "      docker compose exec headscale headscale routes enable -r <id>"
else
    while IFS= read -r rid; do
        $COMPOSE exec -T headscale headscale routes enable -r "$rid" 2>/dev/null \
            && echo "      route $rid enabled"
    done <<< "$ROUTE_IDS"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
$COMPOSE exec -T headscale headscale nodes list 2>/dev/null || true
