# Claude MITM Proxy — PII Anonymizer

A mitmproxy-based HTTPS proxy that sits between Claude Desktop (or any Anthropic API client) and the Anthropic API. It anonymizes PII in outgoing requests and deanonymizes placeholder tokens in incoming responses, so Claude never sees real sensitive data.

## Architecture

```
Claude Desktop
     │ HTTPS  (port 8080)
     ▼
┌─────────────────────────────────────┐
│   claude-mitm-proxy (mitmproxy)     │  ← docker/proxy.dockerfile
│   proxy/main.py                     │
│                                     │
│  • anonymize request                │  PII → [PERSON_0], [EMAIL_1], …
│  • deanonymize response             │  [PERSON_0] → real value
│  • run NER entity detection         │
└─────────────────────────────────────┘
     │
     ▼
Anthropic API / claude.ai

     ┌────────────────────────────────┐
     │  google-workspace-mcp          │  ← MCP reverse proxy (port 8000)
     │  (reverse proxy to MCP server)  │  deanon args → MCP, re-anon results
     └────────────────────────────────┘

┌────────────────────────────────────────┐
│   validator (background service)       │  ← docker/validator.dockerfile
│                                        │
│  • Read: data/requests-sample.jsonl    │  Run weekly or manually
│  • Evaluate: WikiNEural vs Gemma 4 LLM │
│  • Collect: disagreements dataset      │
│  • Fine-tune: new NER model            │
│  • Test: baseline vs fine-tuned        │
│  • Keep: best-performing model         │
└────────────────────────────────────────┘
```

### Key files — Proxy Service

| File | Purpose |
|------|---------|
| `proxy/main.py` | mitmproxy hooks: `request`, `responseheaders`, `response` |
| `proxy/engine.py` | Core anonymize/deanonymize logic, jq path expansion, SSE streaming |
| `proxy/rules.py` | JSONC parser + dataclasses for `assets/rules.jsonc` |
| `assets/rules.jsonc` | URL pattern rules — which fields to anonymize/deanonymize per endpoint |
| `proxy/entity_finder/` | PII detectors: Presidio (structured PII), NER (names/orgs/locs), Mappings (re-detection) |
| `proxy/image_anonymizer.py` | OCR → entity detection → black-box redaction for images |
| `proxy/cache.py` | In-process entity cache (keyed by text, pruned every 12 h) |
| `proxy/mappings.py` | Bidirectional token ↔ value store, Aho-Corasick automaton for re-detection |
| `proxy/anxious_filter.py` | Post-anonymization check — warns if known sensitive values leaked through |
| `proxy/claude_system_prompt.py` | Injects `assets/SYSTEM_PROMPT.md` into every `/v1/messages` request |
| `proxy/control_socket.py` | TCP socket on port 9999 for runtime commands (`dump`, `clear`) |

### Key files — Validator Service

| File | Purpose |
|------|---------|
| `validator/main.py` | Orchestration: `--collect`, `--finetune`, `--test`, `--all` |
| `validator/evaluator.py` | Use Gemma 4 8B to validate WikiNEural predictions on request data |
| `validator/dataset_builder.py` | Consolidate disagreements into CoNLL-format training data (80/20 split) |
| `validator/trainer.py` | Fine-tune WikiNEural model on collected disagreements |
| `validator/tester.py` | Evaluate baseline and fine-tuned models, keep better, delete worse |
| `tests/validator/` | Unit tests for dataset quality, evaluator logic, disagreement scoring |

### Entity finders (run in order)

1. **PresidioEntityFinder** — emails, phones, credit cards, IBANs, SSNs, IPs (Microsoft Presidio)
2. **NEREntityFinder** — persons, organizations, locations (Babelscape wikineural multilingual NER)
3. **MappingsEntityFinder** — re-detects previously anonymized values using Aho-Corasick (always runs, separate from `finders` list)

## Running

```bash
docker compose up
```

The proxy listens on:
- `:8080` — forward proxy (configure as HTTP/HTTPS proxy in Claude Desktop)
- `:8000` — reverse proxy to `google-workspace-mcp:8000` (MCP tool calls)
- `127.0.0.1:9999` — control socket

### First run

On first start, mitmproxy generates a CA certificate at `data/mitmproxy/mitmproxy-ca-cert.pem`. Install it as a trusted CA on every machine routing traffic through the proxy.

### Environment variables (`.env`)

```
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
```

## Development

### Modifying rules

Edit `assets/rules.jsonc` — it uses `//` line comments and `/* */` block comments (JSONC). Changes are picked up immediately via the Docker volume mount (no rebuild needed).

Rule structure:
```jsonc
{
  "anxious_filter_domains": [".*"],   // URL patterns to run the anxious filter on
  "blocked_urls": [ { "url_pattern": "..." } ],
  "anonymise": {
    "requests":  [ { "url_pattern": "...", "sensitive_fields": [".path.to.field"] } ],
    "responses": [ { "url_pattern": "...", "sensitive_fields": true } ]   // MCP
  },
  "deanonymise": {
    "responses": [ { "url_pattern": "...", "sensitive_fields": [...], "sse_fields": [...] } ]
  }
}
```

`sensitive_fields` is either `true` (all strings in the body) or a list of jq path expressions. Paths are expanded recursively to string leaves (`path(expr | .. | strings)`).

### Modifying the system prompt

Edit `assets/SYSTEM_PROMPT.md`. Injected live — no rebuild needed.

### Rebuilding after code changes

Source is mounted as a volume, so Python changes take effect on next request. Dependency changes require a rebuild:

```bash
docker compose build mitm-proxy validator
docker compose up
```

## Validator (Fine-tuning Pipeline)

The validator runs in a separate container, processing saved requests to collect disagreements, fine-tune the NER model, and automatically select the best version.

### Running the validator

**One-time: collect disagreements from the past N requests**
```bash
docker compose run validator poetry run python validator/main.py --collect --limit 100
```

**Fine-tune the model on collected data**
```bash
docker compose run validator poetry run python validator/main.py --finetune
```

**Evaluate both models and keep the better one**
```bash
docker compose run validator poetry run python validator/main.py --test
```

**Full pipeline (collect → finetune → test → select best)**
```bash
docker compose run validator poetry run python validator/main.py --all --limit 100
```

### Weekly automation

To run the validator weekly, set up a cron job on your host:
```cron
# Every Sunday at 2am, collect disagreements and fine-tune
0 2 * * 0 cd /path/to/project && docker compose run -d validator poetry run python validator/main.py --all --limit 200
```

Or use a scheduler inside the Docker container (e.g., APScheduler).

### Testing the validator

Run unit tests for dataset quality and evaluator logic:
```bash
poetry run pytest tests/validator/ -v

# Test specific component
poetry run pytest tests/validator/test_dataset_builder.py -v
poetry run pytest tests/validator/test_evaluator.py -v
```

What the validator checks:
- ✅ **No empty texts** in training data
- ✅ **No duplicate entities** in final dataset
- ✅ **Valid entity types** (PERSON, ORG, LOC, EMAIL, etc.)
- ✅ **Proper 80/20 train/test split**
- ✅ **Gemma-validated ground truth** (removes false positives, adds missed entities)
- ✅ **Model improvement** (keeps new model only if F1 score improves)

### How it works

**Entity Caching (Performance Optimization)**

The proxy logs extracted entities to `data/requests-entities.jsonl` as it anonymizes requests. The validator reuses these cached entities instead of re-running expensive NER extraction.

1. **Proxy** (during normal operation):
   - In `proxy/engine.py:_apply_paths`, after extracting entities, calls `entity_cache_log.log_extracted_entities()`
   - Saves: `request_url`, `field_path`, `text`, `entities`
   - This happens in real-time with minimal overhead

2. **Evaluator** (during validation):
   - Reads `data/requests-sample.jsonl` (pre-anonymization request log)
   - For each text field, checks `data/requests-entities.jsonl` for cached entities
   - If found: uses cached entities (fast path, avoids NER re-run)
   - If not found: runs WikiNEural NER (fallback for old requests)
   - Sends to Gemma 4 5B for validation: "Are these entities correct? What did we miss?"
   - Computes disagreement score (false positives cost 0.3, missed entities cost 0.7)

2. **Dataset Builder** (`validator/dataset_builder.py`):
   - Consolidates samples with disagreement > 0
   - Removes false positives flagged by Gemma
   - Adds missed entities Gemma discovered
   - Creates 80/20 train/test split in CoNLL format

3. **Trainer** (`validator/trainer.py`):
   - Fine-tunes `Babelscape/wikineural-multilingual-ner` on collected dataset
   - Uses HuggingFace Transformers trainer
   - Saves checkpoint to `models/wikineural_finetuned_YYYYMMDD_HHMMSS/`

4. **Tester** (`validator/tester.py`):
   - Evaluates baseline (original WikiNEural) on held-out test set
   - Evaluates fine-tuned model on same test set
   - If fine-tuned F1 > baseline F1: promote fine-tuned to production, archive baseline
   - If fine-tuned F1 ≤ baseline F1: delete fine-tuned, keep baseline

## Testing / benchmarking

Replay saved requests through the pipeline without a running proxy or network:

```bash
# Process up to 10 saved requests
poetry run python scripts/push_saved_request.py --limit 10

# Process a single request by index
poetry run python scripts/push_saved_request.py --index 0

# Save the anonymized output
poetry run python scripts/push_saved_request.py --index 0 --output /tmp/out.json

# List available requests without processing
poetry run python scripts/push_saved_request.py --dry-run
```

Requests are logged (before anonymization) to `data/requests-sample.jsonl` when `save_requests=True` in `ProxyOptions`.

## Runtime control

```bash
# Dump current token ↔ value mappings
echo "dump" | nc 127.0.0.1 9999

# Clear all mappings (next request starts fresh)
echo "clear" | nc 127.0.0.1 9999
```

## Data directories

| Path | Contents |
|------|---------|
| `data/mitmproxy/` | mitmproxy CA cert and state |
| `data/requests-sample.jsonl` | Raw request log (pre-anonymization) |
| `data/requests-entities.jsonl` | **NEW** Extracted entities per request (proxy → validator) |
| `data/cache/images/` | Image OCR result cache (SHA-256 keyed JSON) |
| `data/ignore/` | Debug dumps (last 4xx/5xx body, anxious filter trigger) |
| `data/hf_cache/` | HuggingFace model weights (NER, Gemma 4, EasyOCR) |
| `data/validator/` | Validator output: disagreements, training data, test data |
| `data/validator/disagreements.jsonl` | Raw Gemma-evaluated samples (FP/FN info) |
| `data/validator/training_data.jsonl` | Gemma-corrected training set (80% of samples) |
| `data/validator/test_data.jsonl` | Held-out test set (20% of samples) |
| `models/` | NER model checkpoints |
| `models/baseline/` | Current production WikiNEural model (updated after successful fine-tune) |
| `models/baseline_archive/` | Previous baseline (kept for rollback if needed) |
| `models/wikineural_finetuned_YYYYMMDD_HHMMSS/` | Fine-tuned model checkpoint (deleted if not better) |

Some json files in the data dir are huge, one huge object per line so grepping isnt always the best idea. remember that you have access to `jq` command utility.