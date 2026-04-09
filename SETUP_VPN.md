# VPN + Transparent Proxy — Setup Guide

All internet traffic from connected devices is transparently intercepted by mitmproxy on the VPS and anonymised before reaching Claude / Anthropic's API.

```
Device (any OS)
  └─ Tailscale VPN (WireGuard)
       └─ tailscale-exit (maxou.dev, 100.64.0.1)
            └─ iptables DNAT :80/:443 → mitmproxy :8080
                 └─ internet
```

---

## 1. Server — what is already deployed

The VPS (`maxou.dev`) runs four Docker services managed by `docker-compose.yml`:

| Service | Role |
|---|---|
| `headscale` | WireGuard coordination server at `vpn.maxou.dev` |
| `tailscale` | Exit node (100.64.0.1) + iptables DNAT |
| `mitm-proxy` | mitmproxy — transparent interception on :8080 |
| `google-workspace-mcp` | MCP reverse proxy |

To redeploy from scratch on a new VPS:
```bash
ssh maxou.dev
cd /srv/Projects/freelance/mateo/claude-pii-proxy
docker compose up -d
```

---

## 2. Before connecting any device

### 2a. Generate a fresh auth key

Auth keys expire after 7 days by default. Create a long-lived one whenever needed:

```bash
ssh maxou.dev
# User ID 2 = maxime (the user that owns the exit node)
docker exec headscale headscale preauthkeys create -u 2 --reusable --expiration 365d
```

Copy the printed key — you'll use it on every device you connect.

To list existing keys and check expiry:
```bash
docker exec headscale headscale preauthkeys list
```

### 2b. Get the mitmproxy CA certificate

The CA cert must be installed as a trusted root on every device, otherwise HTTPS connections will show certificate errors.

```bash
# From your local machine
scp maxou.dev:/srv/Projects/freelance/mateo/claude-pii-proxy/data/mitmproxy/mitmproxy-ca-cert.pem ~/Downloads/mitmproxy-ca.pem
```

The cert is also committed at `./data/mitmproxy/mitmproxy-ca-cert.pem` in this repo.

> **When to refresh the cert:** mitmproxy only regenerates its CA on first run (if no cert exists). It will not change unless you delete `data/mitmproxy/` and restart. The current cert is valid until **2036**.

---

## 3. Client setup — Ubuntu / Linux

```bash
# 1. Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Join the VPN (replace AUTHKEY with the key from step 2a)
sudo tailscale up \
  --login-server=https://vpn.maxou.dev \
  --authkey=AUTHKEY \
  --accept-dns=true \
  --exit-node=100.64.0.1 \
  --exit-node-allow-lan-access \
  --hostname=my-ubuntu

# 3. Install the CA cert
sudo cp ~/Downloads/mitmproxy-ca.pem /usr/local/share/ca-certificates/mitmproxy-ca.crt
sudo update-ca-certificates

# 4. Verify
curl https://api.anthropic.com/ 2>&1 | head -1          # should not error
curl -sv https://api.anthropic.com/ 2>&1 | grep issuer  # → issuer: CN=mitmproxy
```

To disconnect / stop using the exit node:
```bash
sudo tailscale set --exit-node=
```

---

## 4. Client setup — macOS

```bash
# 1. Install Tailscale CLI
brew install tailscale
sudo tailscaled install-system-daemon

# 2. Join the VPN
sudo tailscale up \
  --login-server=https://vpn.maxou.dev \
  --authkey=AUTHKEY \
  --accept-dns=true \
  --exit-node=100.64.0.1 \
  --exit-node-allow-lan-access \
  --hostname=my-mac

# 3. Install the CA cert system-wide
sudo security add-trusted-cert \
  -d -r trustRoot \
  -k /Library/Keychains/System.keychain \
  ~/Downloads/mitmproxy-ca.pem

# 4. Verify
curl -sv https://api.anthropic.com/ 2>&1 | grep issuer  # → issuer: CN=mitmproxy
```

> **Note — Chrome / Chromium on macOS** uses the system keychain automatically.
> **Firefox** has its own cert store: Preferences → Privacy & Security → Certificates → View Certificates → Authorities → Import.

If you use the **Tailscale macOS app** instead of the CLI:
1. Open the app → hold Option and click the menu bar icon → "Use custom coordination server…"
2. Enter `https://vpn.maxou.dev`
3. The app will prompt you to log in — open the URL it shows in a browser, which will redirect to headscale's auth page.
4. After login, open System Settings → Network → Tailscale → Options → enable "Route all traffic through exit node" and select `tailscale-exit`.

---

## 5. Client setup — Windows

Open **PowerShell as Administrator**:

```powershell
# 1. Install Tailscale (via winget)
winget install tailscale.tailscale

# 2. Join the VPN
tailscale up `
  --login-server=https://vpn.maxou.dev `
  --authkey=AUTHKEY `
  --accept-dns=true `
  --exit-node=100.64.0.1 `
  --exit-node-allow-lan-access `
  --hostname=my-windows

# 3. Install the CA cert into the Windows trusted root store
Import-Certificate `
  -FilePath "$env:USERPROFILE\Downloads\mitmproxy-ca.pem" `
  -CertStoreLocation Cert:\LocalMachine\Root

# 4. Verify (in a new terminal)
curl.exe -sv https://api.anthropic.com/ 2>&1 | Select-String "issuer"
# → issuer: CN=mitmproxy
```

> **Firefox on Windows** also maintains its own cert store — same import step as macOS Firefox above.

> **If curl.exe is not available**, use: `Invoke-WebRequest https://api.anthropic.com/` — a successful (or 401) response without a cert error means it's working.

---

## 6. Client setup — iOS (iPhone / iPad)

### 6a. Install the CA certificate

The iOS Tailscale app cannot route traffic transparently without the cert being trusted first.

1. AirDrop `mitmproxy-ca.pem` to your iPhone, **or** email it to yourself.
2. Tap the attachment — iOS shows "Profile Downloaded".
3. Go to **Settings → General → VPN & Device Management** → tap the "mitmproxy" profile → **Install**.
4. Enter your passcode when prompted.
5. Go to **Settings → General → About → Certificate Trust Settings** → toggle **mitmproxy** to ON.

Step 5 is mandatory — iOS installs the cert but does not trust it automatically.

### 6b. Connect to the VPN

1. Install **Tailscale** from the App Store.
2. Open the app. Before tapping "Log in", tap the **⚙ gear icon** in the bottom right → **Accounts** → **Log in with server URL**.
   - If you don't see that option: on the main "Log in" screen, look for a small "Other login options" or try long-pressing the login button.
3. Enter the control server URL: `https://vpn.maxou.dev`
4. When prompted for an auth key, enter the key from step 2a.
5. Back on the main screen, tap the three-dot menu next to `tailscale-exit` → **Use as exit node**.

### 6c. Verify

Open Safari and navigate to `https://api.anthropic.com/`. It should load (possibly showing a 401/403 from Anthropic, which is fine) without any certificate warning.

---

## 7. Client setup — Android

### 7a. Install the CA certificate

1. Transfer `mitmproxy-ca.pem` to your device (email, USB, Google Drive, etc.).
2. Go to **Settings → Security → More security settings → Install a certificate → CA certificate**.
   - On some Android versions: Settings → Biometrics and security → Other security settings → Install from device storage.
3. Select the `mitmproxy-ca.pem` file and confirm.

> On Android 11+, user-installed CA certs are trusted by most apps but not by apps that enforce certificate pinning (e.g. some banking apps). Claude / Anthropic clients do not pin certificates.

### 7b. Connect to the VPN

1. Install **Tailscale** from the Play Store.
2. Open the app. Before logging in, tap the **⋮ menu** → **Use custom coordination server**.
3. Enter `https://vpn.maxou.dev` and confirm.
4. Tap **Log in** → enter the auth key from step 2a when prompted.
5. In the node list, long-press `tailscale-exit` → **Use as exit node**.

### 7c. Verify

Open Chrome and navigate to `https://api.anthropic.com/`. No certificate warning = working.

---

## 8. DNS

Headscale is configured to push `1.1.1.1` and `8.8.8.8` as DNS resolvers to all clients (`--accept-dns=true` is the default). This means:

- **You do not need to change DNS settings manually** on any platform.
- Tailscale will override the system DNS when the exit node is active, ensuring DNS queries resolve correctly through the VPN.
- MagicDNS is enabled — Tailscale nodes are reachable by hostname on the `maxou.corp` domain (e.g. `tailscale-exit.maxou.corp`).

If a device experiences DNS failures after connecting (rare), set the system DNS manually to `1.1.1.1` before activating the exit node.

---

## 9. Verify interception is working

On any connected device:

```bash
# The issuer should be mitmproxy, not the real CA
curl -sv https://api.anthropic.com/ 2>&1 | grep -i issuer
#  issuer: CN=mitmproxy, O=mitmproxy  ✓

# Check mitmproxy is logging your traffic (on the VPS)
ssh maxou.dev "docker logs claude-mitm-proxy --tail 20 -f"
```

---

## 10. Managing nodes

```bash
# List all connected nodes
ssh maxou.dev "docker exec headscale headscale nodes list"

# Remove a node (use the ID from the list above)
ssh maxou.dev "docker exec headscale headscale nodes delete --identifier <ID>"

# Rename a node
ssh maxou.dev "docker exec headscale headscale nodes rename --identifier <ID> --new-name <name>"

# Expire a node immediately (forces re-auth)
ssh maxou.dev "docker exec headscale headscale nodes expire --identifier <ID>"

# Generate a new auth key (user ID 2 = maxime)
ssh maxou.dev "docker exec headscale headscale preauthkeys create -u 2 --reusable --expiration 365d"
```

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SSL certificate problem: unable to get local issuer certificate` | CA cert not installed or not trusted | Redo step 2b + platform cert install |
| DNS resolution fails after connecting | Exit node active but DNS not pushed | Manually set DNS to `1.1.1.1` |
| `tailscale up` error about non-default flags | Existing state has different settings | Add `--reset` flag to `tailscale up` |
| Auth key expired | Keys have a 7-day default lifetime | Generate a new key (section 2a) |
| mitmproxy not intercepting | mitm-proxy container restarted before tailscale | `docker compose up -d --force-recreate mitm-proxy` on VPS |
| Mobile: cert installed but still untrusted | iOS step 5 (Certificate Trust Settings) skipped | Go to Settings → General → About → Certificate Trust Settings |
