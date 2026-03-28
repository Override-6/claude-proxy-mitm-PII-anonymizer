# Claude MITM Proxy — Codebase Guide

## Project Purpose

A Windows MITM proxy that intercepts Claude Desktop HTTPS traffic to anonymize PII in outgoing requests and deanonymize placeholders in responses. Uses WinDivert for transparent kernel-level interception (no app-level proxy config needed).

---

## Architecture

```
Claude Desktop
     │ HTTPS :443
     ▼
proxifier.py  (WinDivert NAT — kernel intercept)
     │ CONNECT tunnel
     ▼
mitmproxy :8080
     │
     ├── request hook ──► rule_applier.apply_request_rules()
     │                        ├── text_anonymizer  (NER + regex)
     │                        ├── image_anonymizer (OCR + redaction)
     │                        └── claude_system_prompt (inject)
     │
     └── response hook ─► rule_applier.apply_response_rules()
                              └── deanon_chunk (SSE streaming)

console.py ──UDP:9999──► control_socket.py  (runtime toggles)
viewer.py  ◄─UDP:8080──  event_socket.py    (audit log)
```

---

## File Layout

```
mitm/
├── CLAUDE.md
├── SYSTEM_PROMPT.md          # Injected into every /v1/messages request
├── rules.json                # Declarative URL + jq-path anonymization rules
├── console.py                # Interactive CLI (connects via UDP to proxy)
├── start.bat                 # Launcher: mitmproxy + proxifier + Claude Desktop
├── cache/images/             # SHA256-keyed OCR result cache (gitignored)
├── ignore/                   # Saved redacted images + event log (gitignored)
└── src/
    ├── proxy.py              # mitmproxy hooks (thin — delegates everything)
    ├── rule_applier.py       # Core pipeline: anonymize requests, deanon responses
    ├── rules.py              # rules.json loader + jq path parser
    ├── text_anonymizer.py    # Entity detection → placeholder substitution
    ├── image_anonymizer.py   # EasyOCR → entity detect → black-box redaction
    ├── claude_system_prompt.py # SYSTEM_PROMPT.md injection
    ├── mappings.py           # Bidirectional PII ↔ [TYPE_N] token store
    ├── control_socket.py     # UDP command handler (state toggles)
    ├── event_socket.py       # UDP broadcast to viewer (audit events)
    ├── viewer.py             # Listens on :8080, logs to ignore/events.jsonl
    └── entity_finder/
        ├── __init__.py       # Entity dataclass + AbstractEntityFinder
        ├── regex_finder.py   # EMAIL and PHONE via compiled regex
        ├── gliner_finder.py  # GLiNER multilingual NER (PERSON, ORG, LOC)
        └── ner_finder.py     # spaCy NER (kept for reference, not active)
```

---

## Key Data Flows

### Request anonymization
1. URL matched against `rules.json` request_rules
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
1. `RegexEntityFinder` — EMAIL, PHONE (high confidence, always wins)
2. `GLiNEREntityFinder` — PERSON, ORG, LOC (fills gaps only)

### Response deanonymization
- **SSE streaming**: `deanon_chunk` callback parses each `data:` JSON line, replaces `[TYPE_N]` tokens at targeted jq paths
- **Non-streaming**: `apply_response_rules` parses full JSON, walks paths, replaces tokens
- Unknown tokens (not in mappings) pass through unchanged — never crash

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
      "sse_fields": [".delta.text", ...]   // optional, SSE-only paths
    }
  ]
}
```

jq path syntax: `.key`, `.key[]` (array expand), `.key[].subkey`

### SYSTEM_PROMPT.md
Written in plain text/Markdown. Re-read from disk on every request — edit without restarting the proxy. Injected into the `system` field of `/v1/messages` requests only. Deduplication prevents double-injection on Claude Desktop's history resend.

---

## Runtime State & Console

```python
state = {
    "anon_enabled":           True,   # master anonymization switch
    "deanon_enabled":         False,  # response deanonymization (off by default)
    "save_images":            False,  # save redacted PNGs to ignore/redacted_images/
    "system_prompt_enabled":  True,   # inject SYSTEM_PROMPT.md
}
```

Console commands (run `python console.py`):

| Command | Effect |
|---|---|
| `anon on/off` | Toggle request anonymization |
| `deanon on/off` | Toggle response deanonymization |
| `save images on/off` | Save redacted images to disk |
| `system prompt on/off` | Toggle system prompt injection |
| `status` | Show all flags |
| `dump` | Print all PII → token mappings |
| `clear` | Wipe mapping table |

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
| `mitmproxy` | HTTPS proxy framework | Hooks in proxy.py |
| `gliner` (`urchade/gliner_multi-v2.1`) | Multilingual NER | Lazy-loaded; ~400MB; handles EN/FR/DE/ES/etc. |
| `easyocr` | OCR for image redaction | Singleton; English only; CPU mode |
| `pydivert` | WinDivert bindings | Windows + admin only; used by proxifier.py |
| `Pillow` | Image draw/redaction | |
| `torch` | Backend for GLiNER + EasyOCR | Threaded: `set_num_threads(cpu_count)` |
| `langdetect` | (unused in active path) | Was used by spaCy multilingual finder |

---

## Windows-Specific Notes

- `proxifier.py` requires **Administrator** — WinDivert needs kernel access
- `start.bat` starts everything in order: mitmproxy (port 8080) → wait 18s → proxifier → Claude Desktop
- Proxifier excludes its own PID and mitmproxy's PID to prevent intercept loops
- mitmproxy CA cert must be installed in Windows trust store for HTTPS interception
