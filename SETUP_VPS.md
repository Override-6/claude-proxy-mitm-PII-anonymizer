# VPS Setup Guide — Headscale + Transparent MITM Proxy

Full setup guide for the Raspberry Pi VPS (`maxou.dev`) acting as Headscale coordination server, Tailscale exit node, and transparent HTTPS proxy.

## Architecture overview

```
Device (any OS)
  │
  │  WireGuard tunnel (Tailscale)
  ▼
maxou.dev (Raspberry Pi)
  ├── Headscale           — VPN coordination server (vpn.maxou.dev:443)
  ├── Tailscale node      — exit node (100.64.0.3)
  ├── redsocks            — bridges transparent interception → forward proxy
  ├── nftables rule       — redirects tailscale0 :80/:443 → redsocks:12345
  └── claude-mitm-proxy   — mitmproxy forward proxy (Docker, port 8080)
         └── PII anonymizer, NER, Presidio
```

**Traffic flow (exit node mode):**
```
Device HTTP/HTTPS
  → WireGuard tunnel → tailscale0
  → nftables REDIRECT → redsocks:12345
  → HTTP CONNECT → mitmproxy:8080
  → PII anonymization
  → internet
```

**Alternative (Linux only, local SSH tunnel mode):**
```
Device HTTP/HTTPS
  → iptables OUTPUT redirect → local redsocks:12345
  → SSH tunnel → maxou.dev:8080
  → mitmproxy:8080
  → PII anonymization
  → internet
```

---

## Part 1 — VPS Setup

### Prerequisites

- Raspberry Pi (or any Debian/Ubuntu server) with a public IP
- Domain pointing to the VPS (e.g. `maxou.dev`)
- Wildcard TLS certificate for `*.maxou.dev` via certbot + nginx (already set up at `/srv/firewall/`)
- Docker + Docker Compose installed

### 1.1 Install Headscale

```bash
wget https://github.com/juanfont/headscale/releases/download/v0.25.1/headscale_0.25.1_linux_amd64.deb
sudo apt install ./headscale_0.25.1_linux_amd64.deb
```

### 1.2 Configure Headscale

Edit `/etc/headscale/config.yaml`:

```yaml
server_url: https://vpn.maxou.dev
listen_addr: 0.0.0.0:18080          # nginx proxies 443 → 18080
metrics_listen_addr: 127.0.0.1:9091 # avoids conflict with other services
ip_prefixes:
  - 100.64.0.0/10
dns:
  magic_dns: true
  base_domain: maxou.corp
```

> **Note:** Headscale listens on port **18080** (not 8080) because mitmproxy uses 8080.

### 1.3 Nginx vhost for Headscale

Create `/srv/firewall/nginx/conf.d/vpn.maxou.dev.conf`:

```nginx
server {
    listen 443 ssl;
    server_name vpn.maxou.dev;

    include /etc/nginx/conf.d/_wildcard_maxou_ssl.inc;

    location / {
        proxy_pass http://host.docker.internal:18080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Reload nginx:
```bash
cd /srv/firewall && docker compose exec nginx nginx -s reload
```

### 1.4 Start Headscale

```bash
sudo systemctl enable --now headscale
# Verify
curl -s https://vpn.maxou.dev/health   # → {"status":"pass"}
```

### 1.5 Create a user and auth key

```bash
sudo headscale users create myuser
sudo headscale preauthkeys create --user myuser --reusable --expiration 168h
# Save the printed key — needed for all device enrollments
```

### 1.6 Install Tailscale on the VPS (exit node)

The VPS must join its own VPN to act as exit node:

```bash
curl -fsSL https://tailscale.com/install.sh | sudo sh

# Enable IP forwarding
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.d/99-tailscale.conf
echo "net.ipv6.conf.all.forwarding=1" | sudo tee -a /etc/sysctl.d/99-tailscale.conf
sudo sysctl -p /etc/sysctl.d/99-tailscale.conf

# Join the VPN and advertise as exit node
sudo tailscale up \
  --login-server=https://vpn.maxou.dev \
  --authkey=<YOUR_PREAUTH_KEY> \
  --advertise-exit-node \
  --accept-routes
```

Approve the exit node routes:
```bash
sudo headscale routes list
sudo headscale routes enable -r 1   # 0.0.0.0/0
sudo headscale routes enable -r 2   # ::/0
```

### 1.7 Deploy the MITM proxy

On the VPS, clone the repo and start the proxy:

```bash
git clone <repo_url> /srv/Projects/freelance/mateo/claude-pii-proxy
cd /srv/Projects/freelance/mateo/claude-pii-proxy
docker compose up -d mitm-proxy
```

Verify it's running:
```bash
docker ps | grep mitm-proxy
# Forward proxy must be reachable:
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080   # → 502 (expected, no target)
```

### 1.8 Set up transparent proxy interception (redsocks + nftables)

Install redsocks:
```bash
sudo apt install -y redsocks
```

Configure `/etc/redsocks.conf`:
```
base {
    log_debug = off;
    log_info = on;
    log = "syslog:daemon";
    daemon = on;
    redirector = iptables;
}

redsocks {
    local_ip = 127.0.0.1;
    local_port = 12345;
    ip = 127.0.0.1;
    port = 8080;
    type = http-connect;
}
```

Start redsocks:
```bash
sudo systemctl enable --now redsocks
```

Apply nftables REDIRECT rule (run as root):
```bash
sudo /usr/local/bin/setup-mitm-redirect.sh
```

The script `/usr/local/bin/setup-mitm-redirect.sh` contains:
```bash
#!/bin/bash
nft delete table ip mitm_redirect 2>/dev/null || true
nft add table ip mitm_redirect
nft add chain ip mitm_redirect PREROUTING '{ type nat hook prerouting priority dstnat; }'
nft add rule ip mitm_redirect PREROUTING iifname tailscale0 tcp dport { 80, 443 } redirect to :12345
```

Enable the systemd service to re-apply the rule on boot:
```bash
sudo systemctl enable mitm-redirect
```

### 1.9 Install the mitmproxy CA certificate (VPS-wide)

The CA cert is generated on first proxy start at `./data/mitmproxy/mitmproxy-ca-cert.pem`.
Keep this file — all devices need it.

---

## Part 2 — Device Setup

All devices need:
1. **Tailscale client** pointing to `https://vpn.maxou.dev`
2. **mitmproxy CA certificate** installed as a trusted root CA
3. Either **exit node enabled** (transparent) or **explicit proxy configured**

### Get the CA certificate

```bash
scp maxou.dev:/srv/Projects/freelance/mateo/claude-pii-proxy/data/mitmproxy/mitmproxy-ca-cert.pem .
```

---

### Linux (Debian / Ubuntu)

**Option A — Transparent proxy via local SSH tunnel (recommended for dev machines)**

This approach avoids exit node and routing all traffic through the VPS.

```bash
# 1. Join the VPN
curl -fsSL https://tailscale.com/install.sh | sudo sh
sudo tailscale up --login-server=https://vpn.maxou.dev --authkey=<KEY>

# 2. Set up persistent SSH tunnel (localhost:8080 → maxou.dev:8080)
bash scripts/tunnel-setup.sh

# 3. Set up transparent proxy (redsocks + iptables redirect :443 → tunnel)
sudo bash scripts/local-proxy-setup.sh

# 4. Install CA cert
sudo cp mitmproxy-ca-cert.pem /usr/local/share/ca-certificates/mitmproxy.crt
sudo update-ca-certificates
```

To disable:
```bash
sudo bash scripts/local-proxy-teardown.sh
```

**Option B — Transparent proxy via exit node**

```bash
# Join VPN and use VPS as exit node
sudo tailscale up \
  --login-server=https://vpn.maxou.dev \
  --authkey=<KEY> \
  --exit-node=100.64.0.3 \
  --exit-node-allow-lan-access

# Install CA cert
sudo cp mitmproxy-ca-cert.pem /usr/local/share/ca-certificates/mitmproxy.crt
sudo update-ca-certificates
```

> **Warning:** Exit node routes ALL traffic through the VPS including SSH sessions. SSH to the VPS using its VPN IP (`ssh 100.64.0.3`) not the public hostname.

---

### macOS

```bash
# 1. Install Tailscale
brew install tailscale
# or: App Store → "Tailscale"

# 2. Join the VPN
sudo tailscale up --login-server=https://vpn.maxou.dev --authkey=<KEY>

# 3. Enable exit node
sudo tailscale up \
  --login-server=https://vpn.maxou.dev \
  --exit-node=100.64.0.3 \
  --exit-node-allow-lan-access
```

**Install CA cert:**
1. Double-click `mitmproxy-ca-cert.pem` → opens Keychain Access
2. Add to **System** keychain
3. Find "mitmproxy" in the list → double-click → Trust → **Always Trust**

**Alternative — explicit proxy (no exit node):**

System Settings → Network → select interface → Proxies:
- HTTPS Proxy: `100.64.0.3` port `8080`

---

### Windows

1. Download Tailscale from [tailscale.com/download](https://tailscale.com/download) and install
2. Open PowerShell as Administrator:

```powershell
tailscale up --login-server=https://vpn.maxou.dev --authkey=<KEY>

# Enable exit node
tailscale up --login-server=https://vpn.maxou.dev --exit-node=100.64.0.3 --exit-node-allow-lan-access
```

**Install CA cert:**
1. Double-click `mitmproxy-ca-cert.pem`
2. Install Certificate → Local Machine → **Trusted Root Certification Authorities**

**Alternative — explicit proxy (no exit node):**

Settings → Network & Internet → Proxy → Manual proxy setup:
- Address: `100.64.0.3`, Port: `8080`

---

### Android

**Join the VPN:**
1. Install **Tailscale** from Play Store
2. Tap ⋮ menu → **Use custom control server** → enter `https://vpn.maxou.dev`
3. Log in with the auth key when prompted

**Enable exit node (transparent proxy):**
1. In Tailscale app → tap on `raspberrypi` in the device list
2. Enable **Use as exit node**

**Install CA cert:**
1. Transfer `mitmproxy-ca-cert.pem` to the device
2. Settings → Security → Install certificate → **CA certificate**
3. Select the `.pem` file and confirm

**Alternative — explicit proxy (no exit node):**

Wi-Fi → long-press network → Modify → Advanced:
- Proxy: **Manual**
- Hostname: `100.64.0.3`, Port: `8080`

> Note: Wi-Fi proxy only applies to that network. For mobile data, use an app like **Drony** or **ProxyDroid**.

---

### iPhone / iPad

**Join the VPN:**
1. Install **Tailscale** from App Store
2. Tap ⋮ menu → **Use custom control server** → enter `https://vpn.maxou.dev`
3. Log in with the auth key

**Enable exit node:**
1. In Tailscale app → tap on `raspberrypi`
2. Enable **Use as exit node**

**Install CA cert:**
1. AirDrop or email `mitmproxy-ca-cert.pem` to the device
2. Settings → General → VPN & Device Management → install the profile
3. Settings → General → About → Certificate Trust Settings → **enable mitmproxy**

**Alternative — explicit proxy (no exit node):**

Settings → Wi-Fi → tap (i) → Configure Proxy → Manual:
- Server: `100.64.0.3`, Port: `8080`

---

## Part 3 — Verification

After setup, verify traffic is being intercepted:

**From any device:**
```bash
# Should show mitmproxy's certificate (issued by "mitmproxy"), not the real cert
curl -v https://api.anthropic.com/v1/models 2>&1 | grep "issuer"
```

**Check proxy logs on the VPS:**
```bash
docker logs claude-mitm-proxy -f
# Should show client connect / server connect lines for your requests
```

**Check redsocks on VPS:**
```bash
sudo journalctl -u redsocks -f
```

---

## Part 4 — Maintenance

### Re-apply nftables rules after reboot

Rules are applied automatically via systemd (`mitm-redirect.service`). To apply manually:
```bash
sudo /usr/local/bin/setup-mitm-redirect.sh
```

### Restart proxy after code changes

```bash
cd /srv/Projects/freelance/mateo/claude-pii-proxy
docker compose restart mitm-proxy
# Re-apply nftables (container IP may have changed)
sudo /usr/local/bin/setup-mitm-redirect.sh
```

### Add a new device

```bash
# On the VPS
sudo headscale preauthkeys create --user myuser --reusable --expiration 168h
# Then follow the relevant device section above
```

### View active VPN nodes

```bash
sudo headscale nodes list
```

### Control the proxy at runtime

```bash
# Dump current anonymization mappings
echo "dump" | nc 127.0.0.1 9999

# Clear all mappings
echo "clear" | nc 127.0.0.1 9999
```
