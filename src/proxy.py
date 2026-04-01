"""
MITM Proxy for intercepting Claude Desktop traffic.
- Anonymizes PII in outgoing prompts (request)
- Deanonymizes placeholders in Claude's streamed responses (SSE)
- Streams responses through without buffering (no slowdown)
- Logs anonymized traffic to JSONL
"""
import json
import os
from typing import Tuple

from mitmproxy import http

from control_socket import init_control_socket
from entity_finder import Entity
from entity_finder.mappings_finder import MappingsEntityFinder
from mappings import Mappings, REDACTED_REGEX
from rule_applier import apply_request_rules, apply_response_rules, apply_mcp_request_rules, apply_mcp_response_rules, make_deanon_chunk, rules, state
from text_anonymizer import start_cache_prune_task, _EXEMPT_RE


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

async def running():
    await init_control_socket(state)
    start_cache_prune_task()

_ANXIOUS_FILTER_WHITELIST = frozenset(["CLAUDE", "CLAUDE CODE", "CLAUDE COWORK", "ANTHROPIC"])

def anxious_filter(mappings: Mappings, request_body: str) -> Tuple[bool, list[Entity]]:
    # Strip {{...}} exempt spans so entities intentionally left unredacted don't trigger the filter
    filtered_body = _EXEMPT_RE.sub('', request_body)
    entities = MappingsEntityFinder().find_entities(filtered_body, mappings)
    entities = [entity for entity in entities if entity.text.upper() not in _ANXIOUS_FILTER_WHITELIST and not REDACTED_REGEX.fullmatch(entity.text)]
    return len(entities) > 0, entities

def trigger_anxious_filter(url: str, flow: http.HTTPFlow, entities: list[Entity]):
    # flow.response = http.Response.make(
    #     403,
    #     "Blocked by proxy: request contains unredacted sensible data.",
    #     {"Content-Type": "text/plain"},
    # )

    print("[proxy] Anxious filter triggered !")
    print(f"[proxy] Request {url} contained {len(entities)} unmasked sensible entities.")
    print(f"[proxy] entities are ", set(e.text for e in entities))
    # print("[proxy] Dropping request and returning 403 instead.")
    dump_path = "/app/ignore/last_anxious_filter.json"
    os.makedirs("/app/ignore", exist_ok=True)
    with open(dump_path, "w") as f:
        f.write(flow.request.get_content().decode("utf-8"))
    print(f"[proxy] Full request body saved to {dump_path}")
    print("")




_REQUESTS_LOG = "/app/data/requests-sample.jsonl"


def _log_request(flow: http.HTTPFlow) -> None:
    """Append the raw request (headers + body) to the JSONL sample log."""
    try:
        content = flow.request.get_content()
        body_str = content.decode("utf-8", errors="replace") if content else ""
        try:
            body = json.loads(body_str)
        except (json.JSONDecodeError, ValueError):
            body = body_str
        entry = {
            "url": flow.request.pretty_url,
            "method": flow.request.method,
            "headers": dict(flow.request.headers),
            "body": body,
        }
        os.makedirs(os.path.dirname(_REQUESTS_LOG), exist_ok=True)
        with open(_REQUESTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[proxy] Could not log request: {e}")


async def request(flow: http.HTTPFlow):
    # Force server to send uncompressed responses so our stream callback
    # receives plain text instead of gzip/br bytes.
    flow.request.headers.pop("accept-encoding", None)

    # Block blacklisted URLs before any processing
    url = flow.request.pretty_url.split("?")[0]
    for blocked in rules.blocked_urls:
        if blocked.url_pattern.fullmatch(url):
            print(f"[proxy] BLOCKED: {url}")
            flow.response = http.Response.make(
                403,
                "Blocked by proxy: this endpoint is not allowed.",
                {"Content-Type": "text/plain"},
            )
            return

    # Log raw request BEFORE anonymization if enabled
    if state["log_requests"]:
        _log_request(flow)

    # MCP reverse proxy: deanonymize before forwarding to the MCP server.
    # no need to pass anxious checks
    if apply_mcp_request_rules(flow, rules.mcp_rules):
        return

    # Anonymize before broadcasting so the viewer sees the redacted body.
    await apply_request_rules(flow, rules.request_rules)

    if state["anxious_enabled"] and any(pattern.match(url) for pattern in rules.anxiety_watchlist):
        body = flow.request.get_content().decode("utf-8", errors="ignore")
        triggered, entities = anxious_filter(state["mappings"], body)
        if triggered:
            trigger_anxious_filter(url, flow, entities)
            


async def responseheaders(flow: http.HTTPFlow):
    """Enable streaming for SSE responses so they're not buffered."""
    content_type = flow.response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        return

    # Find matching response rule to get targeted SSE fields
    url = flow.request.pretty_url.split("?")[0]
    sse_fields = None
    for rule in rules.response_rules:
        if rule.url_pattern.fullmatch(url) and rule.sse_fields:
            sse_fields = rule.sse_fields
            break

    flow.response.stream = make_deanon_chunk(sse_fields)


async def response(flow: http.HTTPFlow):
    # Broadcast raw (still-anonymized) response to viewer
    content = flow.response.get_content()
    body = content.decode("utf-8", errors="ignore") if content else None

    if flow.response.status_code >= 400:
        req_content = flow.request.get_content()
        req_body = req_content.decode("utf-8", errors="ignore") if req_content else ""
        print(f"[proxy] {flow.response.status_code} from {flow.request.pretty_url}")
        print(f"[proxy] Response: {body}")
        try:
            # Save full body to file for inspection
            dump_path = "/app/ignore/last_400_body.json"
            os.makedirs("/app/ignore", exist_ok=True)
            with open(dump_path, "w") as f:
                f.write(req_body)
            print(f"[proxy] Full request body saved to {dump_path}")
        except Exception as e:
            print(f"[proxy] Could not parse request body: {e}")

    # MCP reverse proxy: anonymize tool results before they reach Claude.
    await apply_mcp_response_rules(flow, rules.mcp_rules)

    # Deanonymize non-streaming responses before they reach the client
    apply_response_rules(flow, rules.response_rules)
