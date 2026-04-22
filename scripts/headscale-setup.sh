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

echo "      waiting for headscale to be healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:18080/health &>/dev/null; then
        echo "      headscale is healthy"
        break
    fi
    sleep 1
    if [[ $i -eq 30 ]]; then
        echo "ERROR: headscale did not become healthy in 30s" >&2
        exit 1
    fi
done

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
USER_EXISTS=$($COMPOSE exec -T headscale headscale users list -o json 2>/dev/null \
    | python3 -c "import json,sys; users=json.load(sys.stdin); print('yes' if any(u['name']=='${HEADSCALE_USER}' for u in users) else 'no')")
if [[ "$USER_EXISTS" == "yes" ]]; then
    echo "      user already exists"
else
    $COMPOSE exec -T headscale headscale users create "${HEADSCALE_USER}"
    echo "      user created"
fi

# ── 3. Generate pre-auth key ─────────────────────────────────────────────────
echo "[3/6] Generating pre-auth key (reusable, 168h)..."
USER_ID=$($COMPOSE exec -T headscale headscale users list -o json 2>/dev/null \
    | python3 -c "import json,sys; users=json.load(sys.stdin); print(next(u['id'] for u in users if u['name']=='${HEADSCALE_USER}'))")
if [[ -z "$USER_ID" ]]; then
    echo "ERROR: could not find user ID for '${HEADSCALE_USER}'" >&2
    exit 1
fi
echo "      user ID: ${USER_ID}"
AUTHKEY=$($COMPOSE exec -T headscale headscale preauthkeys create \
    --user "${USER_ID}" \
    --reusable \
    --expiration 168h \
    -o json 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['key'])")

if [[ -z "$AUTHKEY" ]]; then
    echo "ERROR: could not extract auth key from headscale output" >&2
    exit 1
fi
echo "      authkey: ${AUTHKEY:0:12}…"

# Save to .env so container self-authenticates on reboot
ENV_FILE="$(pwd)/.env"
touch "$ENV_FILE"
if grep -q "^TS_AUTHKEY=" "$ENV_FILE"; then
    sed -i "s|^TS_AUTHKEY=.*|TS_AUTHKEY=${AUTHKEY}|" "$ENV_FILE"
else
    echo "TS_AUTHKEY=${AUTHKEY}" >> "$ENV_FILE"
fi
echo "      saved to .env"

# ── 4. Run tailscale up inside the running container ─────────────────────────
echo "[4/6] Authenticating tailscale (exec into container)..."
$COMPOSE exec -T tailscale tailscale up \
    --login-server="${HEADSCALE_URL}" \
    --authkey="${AUTHKEY}" \
    --advertise-exit-node \
    --accept-routes \
    --accept-dns=false \
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

# Get node ID
NODE_ID=$($COMPOSE exec -T headscale headscale nodes list -o json 2>/dev/null \
    | python3 -c "import json,sys; nodes=json.load(sys.stdin); print(next(n['id'] for n in nodes if n.get('givenName')==\"${TS_HOSTNAME}\" or n.get('name','').startswith(\"${TS_HOSTNAME}\")))")
echo "      node ID: ${NODE_ID}"

echo "      approving exit-node routes..."
ROUTES=$($COMPOSE exec -T headscale headscale nodes list-routes --identifier "${NODE_ID}" -o json 2>/dev/null \
    | python3 -c "
import json, sys
nodes = json.load(sys.stdin)
node = nodes[0] if isinstance(nodes, list) else nodes
exit_routes = [r for r in node.get('available_routes', []) if r in ('0.0.0.0/0', '::/0')]
print(','.join(exit_routes))
")

if [[ -z "$ROUTES" ]]; then
    echo "      no exit-node routes advertised yet — re-run to approve later"
else
    $COMPOSE exec -T headscale headscale nodes approve-routes \
        --identifier "${NODE_ID}" --routes "${ROUTES}" 2>/dev/null \
        && echo "      routes approved: ${ROUTES}"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
$COMPOSE exec -T headscale headscale nodes list 2>/dev/null || true
