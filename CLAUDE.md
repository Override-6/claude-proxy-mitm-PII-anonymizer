# Claude MITM Proxy — Codebase Guide

## Project Purpose

A MITM proxy that intercepts Claude Desktop / Claude Code HTTPS traffic to anonymize PII in outgoing requests and deanonymize placeholders in responses. Also acts as a reverse proxy in front of local MCP servers, deanonymizing tool call arguments and re-anonymizing tool results.

Runs in Docker. On Windows, WinDivert (`proxifier.py`) provides transparent kernel-level interception; on Linux/Mac, configure the system or application HTTP proxy to point at port 8080.

---

## Architecture

```
Claude Desktop / Claude Code
     │ HTTPS :443
     ▼
proxifier.py  (WinDivert NAT — Windows kernel intercept)
     │ CONNECT tunnel
     ▼
mitmproxy :8080  (forward proxy — Claude API traffic)
     │
     ├── request hook ──► rule_applier.apply_mcp_request_rules()   ← deanon MCP args first
     │                ──► rule_applier.apply_request_rules()
     │                        ├── text_anonymizer  (NER + regex)
     │                        ├── image_anonymizer (OCR + redaction)
     │                        └── claude_system_prompt (inject)
     │
     └── response hook ─► rule_applier.apply_mcp_response_rules()  ← anon MCP results
                       ─► rule_applier.apply_response_rules()
                              └── deanon_chunk (SSE streaming)

mitmproxy :8001  (reverse proxy — MCP server traffic)
     │ deanon request → forward to google-workspace-mcp:8000 → anon response
     ▼
google-workspace-mcp :8000

console.py ──TCP:9999──► control_socket.py  (runtime toggles)
```

---

## File Layout

```
├── CLAUDE.md
├── SYSTEM_PROMPT.md              # Injected into every /v1/messages request
├── rules.json                    # Declarative URL + jq-path anonymization rules
├── console.py                    # Interactive CLI (connects via TCP to proxy)
├── docker-compose.yml            # mitm-proxy + google-workspace-mcp services
├── Dockerfile                    # mitm-proxy image
├── docker/
│   └── google-workspace-mcp/
│       └── Dockerfile            # Clones + runs taylorwilsdon/google_workspace_mcp
├── start.bat                     # Windows launcher: mitmproxy + proxifier + Claude Desktop
├── scripts/
│   ├── push_saved_request.py     # Benchmark: replay saved requests through the anon pipeline
│   └── finetune_ner.py
└── src/
    ├── proxy.py                  # mitmproxy hooks (thin — delegates everything)
    ├── rule_applier.py           # Core pipeline: anonymize requests, deanon responses, MCP rules
    ├── rules.py                  # rules.json loader + jq path parser
    ├── text_anonymizer.py        # Entity detection → placeholder substitution
    ├── image_anonymizer.py       # EasyOCR → entity detect → black-box redaction
    ├── claude_system_prompt.py   # SYSTEM_PROMPT.md injection
    ├── mappings.py               # Bidirectional PII ↔ [TYPE_N] token store
    ├── control_socket.py         # TCP command handler (state toggles)
    ├── event_socket.py           # UDP broadcast to viewer (audit events)
    ├── viewer.py                 # Listens on :8080, logs to ignore/events.jsonl
    └── entity_finder/
        ├── __init__.py           # Entity dataclass + AbstractEntityFinder
        ├── regex_finder.py       # EMAIL and PHONE via compiled regex
        ├── presidio_finder.py    # Presidio structured PII (email, phone, IBAN, SSN, IP…)
        ├── mappings_finder.py    # Re-detects previously-seen entities (FlashText)
        └── ner_finder.py         # NER (PERSON, ORG, LOC) — English model (distilbert-base-uncased-finetuned-conll03)
```

---

## Key Data Flows

### Request anonymization (Claude API)
1. URL matched against `rules.json` `request_rules`
2. Body dispatched by content-type:
   - JSON → `_apply_rule_json` → `_apply_paths` (jq-guided)
   - Multipart → `_apply_rule_multipart` (per-part dispatch)
   - Form-urlencoded → `_apply_rule_urlencoded`
3. Text fields → `text_anonymizer.anonymize_text`
4. Image fields → `image_anonymizer.anonymize_image`
5. `claude_system_prompt.inject` appended after anonymization
6. **Any exception → 502, request never forwarded**

### Entity detection priority (text)
Finders run in order; later-finder entities that overlap an already-accepted span are discarded:
1. `PresidioEntityFinder` — structured PII: EMAIL, PHONE, IBAN, SSN, IP, credit card
2. `NEREntityFinder` — English NER (distilbert-base-uncased-finetuned-conll03): PERSON, ORG, LOC
3. `MappingsEntityFinder` — catches previously-seen entities missed by the above

### Response deanonymization (Claude API)
- **SSE streaming**: `deanon_chunk` callback parses each `data:` JSON line, replaces `[TYPE_N]` tokens at targeted jq paths
- **Non-streaming**: `apply_response_rules` parses full JSON, walks paths, replaces tokens
- Unknown tokens (not in mappings) pass through unchanged — never crash

### MCP reverse proxy flow (port 8001)
Traffic from Claude Desktop/Code to the MCP server goes through port 8001:
1. **Request** → `apply_mcp_request_rules`: deanonymize `[TYPE_N]` tokens → real values before the MCP server sees them (so searches/lookups work)
2. Forward to `google-workspace-mcp:8000`
3. **Response** → `apply_mcp_response_rules`: anonymize real PII in tool results → tokens before Claude sees them

Both steps use the same `Mappings` object as the forward proxy (shared in-process).

Rules for MCP URLs live under `"mcp_rules"` in `rules.json`.

### Anxious filter
Runs after anonymization on URLs in `anxiety_watchlist`. Checks the post-anonymization body for any known-sensitive entity still present (via `MappingsEntityFinder`). Returns 403 if any unredacted entity is found.
- Whitelist: `CLAUDE`, `CLAUDE CODE`, `CLAUDE COWORK`, `ANTHROPIC`
- `{{...}}` exempt spans are stripped from the body before the check — entities inside exempt markers never trigger the filter

### Image OCR pipeline
1. Hash image bytes (SHA256) → check `cache/images/{hash}.json`
2. Cache miss: EasyOCR → group regions into lines → smart merge → entity detection → save cache
3. Cache hit: skip OCR + NER entirely
4. Redaction: precise per-character bbox → black rectangle → white label

---

## Configuration

### rules.json
```json
{
  "exempt_words":       ["claude", ...],
  "anxiety_watchlist":  ["(api|a-api)\\.anthropic\\.com", ...],
  "blocked_urls":       [{ "url_pattern": "regex" }],
  "request_rules": [
    {
      "url_pattern": "regex",
      "sensitive_fields": [".messages[].content[].text", ...] | true
    }
  ],
  "response_rules": [
    {
      "url_pattern": "regex",
      "sensitive_fields": [...],
      "sse_fields": [".delta.text", ...]
    }
  ],
  "mcp_rules": [
    {
      "url_pattern": "http://[^/]+:8001/.*",
      "sensitive_fields": true
    }
  ]
}
```

`mcp_rules` entries apply deanonymization to the **request** and anonymization to the **response** — the opposite of `request_rules` / `response_rules`.

jq path syntax: `.key`, `.key[]` (array expand), `.key[].subkey`

### SYSTEM_PROMPT.md
Written in plain text/Markdown. Re-read from disk on every request — edit without restarting the proxy. Injected into the `system` field of `/v1/messages` requests only. Deduplication prevents double-injection on Claude Desktop's history resend.

---

## Runtime State & Console

```python
state = {
    "anon_enabled":           True,   # master anonymization switch (also gates MCP rules)
    "deanon_enabled":         False,  # response deanonymization (off by default)
    "anxious_enabled":        True,   # 403 on unredacted sensitive data
    "save_images":            True,   # save redacted PNGs to data/redacted_images/
    "system_prompt_enabled":  True,   # inject SYSTEM_PROMPT.md
    "log_requests":           False,  # log raw pre-anon requests to data/requests-sample.jsonl
}
```

Console commands (run `python console.py`):

| Command | Effect |
|---|---|
| `anon on/off` | Toggle request anonymization (also disables MCP rules) |
| `deanon on/off` | Toggle response deanonymization |
| `anxious on/off` | Toggle the anxious filter |
| `save images on/off` | Save redacted images to disk |
| `system prompt on/off` | Toggle system prompt injection |
| `log requests on/off` | Log raw requests (pre-anonymization) to `data/requests-sample.jsonl` |
| `status` | Show all flags |
| `dump` | Print all PII → token mappings |
| `clear` | Wipe mapping table |

---

## Request Logging & Benchmarking

When `log_requests` is enabled, every incoming request is appended to `data/requests-sample.jsonl` **before** anonymization (full headers + body).

To benchmark the anonymization pipeline against saved requests (no network, no real API calls):
```bash
python scripts/push_saved_request.py              # all entries
python scripts/push_saved_request.py --limit 5    # first 5
python scripts/push_saved_request.py --index 2    # single entry
python scripts/push_saved_request.py --dry-run    # list without processing
```

The script imports `rule_applier` directly from `src/`, constructs fake mitmproxy flow objects, and reports per-request timing + a summary. First run is slow (model lazy-loads); subsequent runs benefit from the entity cache.

---

## Mappings

`Mappings` stores a bidirectional PII ↔ placeholder map:
- **Key for lookup**: `sensitive_value.upper()` — case-insensitive dedup (`Alice` and `ALICE` → same token)
- **Stored value**: original casing preserved for deanonymization
- **Token format**: `[TYPE_GLOBALID]` e.g. `[PERSON_0]`, `[EMAIL_3]`

---

## Exempt Markers

Wrap any text in `{{...}}` to prevent anonymization:
- `{{maxime}}` → passes through as `{{maxime}}` (not anonymized, markers preserved)
- Entities that overlap (even partially) with an exempt span are dropped entirely
- The anxious filter also ignores content inside `{{...}}` — it won't 403 on exempt values

---

## Docker Setup

```
docker compose up --build
```

| Service | Port | Purpose |
|---|---|---|
| `mitm-proxy` | 8080 | Forward proxy (Claude API traffic) |
| `mitm-proxy` | 8001 | Reverse proxy (MCP server traffic) |
| `mitm-proxy` | 9999 | Control socket (console.py) |
| `google-workspace-mcp` | 8000 | Google Workspace MCP server (OAuth callback) |

**Required env vars** (`.env` file in project root):
```
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
```

**First-time OAuth login**: visit `http://localhost:8000` in your browser to trigger the Google OAuth flow. Tokens are persisted in `data/google-workspace-mcp/`.

**MCP client config** (Claude Desktop / Claude Code): point the MCP server URL at `http://YOUR_HOST:8001`.

---

## Code Style Constraints

- `proxy.py` stays thin — only mitmproxy hooks, no business logic
- Anonymizer only touches string-type fields — no structural changes to request bodies
- No over-engineering: helpers only when used in multiple places
- Any error in request processing must result in 502 — never forward a partially-processed or unredacted request
- `torch.set_num_threads` called once at module level in `image_anonymizer.py` (PyTorch restriction)

---

## Models & Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| `mitmproxy` | HTTPS proxy framework | Hooks in proxy.py; dual-mode (forward + reverse) |
| `presidio-analyzer` | Structured PII detection | EMAIL, PHONE, IBAN, SSN, IP, credit card |
| `transformers` (`elastic/distilbert-base-uncased-finetuned-conll03-english`) | English NER | Active in `ner_finder.py`; detects PERSON, [LOC_29], LOC |
| `gliner` (`urchade/gliner_multi-v2.1`) | Multilingual NER | Cached locally; not yet wired in |
| `easyocr` | OCR for image redaction | Singleton; English only; CPU mode |
| `pydivert` | WinDivert bindings | Windows + admin only; used by proxifier.py |
| `Pillow` | Image draw/redaction | |
| `torch` | Backend for NER + EasyOCR | Threaded: `set_num_threads(cpu_count)` |

---

## Windows-Specific Notes

- `proxifier.py` requires **Administrator** — WinDivert needs kernel access
- `start.bat` starts everything in order: mitmproxy (port 8080) → wait 18s → proxifier → Claude Desktop
- Proxifier excludes its own PID and mitmproxy's PID to prevent intercept loops
- mitmproxy CA cert must be installed in Windows trust store for HTTPS interception
