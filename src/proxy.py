"""
MITM Proxy for intercepting Claude Desktop traffic.
- Anonymizes PII in outgoing prompts (request)
- Deanonymizes placeholders in Claude's streamed responses (SSE)
- Streams responses through without buffering (no slowdown)
- Logs anonymized traffic to JSONL
"""
import time

from mitmproxy import http

from control_socket import init_control_socket
from event_socket import get_event_socket, init_event_socket
from rule_applier import apply_request_rules, apply_response_rules, make_deanon_chunk, rules, state


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

async def running():
    await init_event_socket()
    await init_control_socket(state)


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
            socket = get_event_socket()
            socket.broadcast({"type": "blocked", "url": flow.request.pretty_url})
            return

    t0 = time.time()
    # Anonymize before broadcasting so the viewer sees the redacted body.
    anonymized = await apply_request_rules(flow, rules.request_rules)
    t1 = time.time()

    socket = get_event_socket()
    content = flow.request.get_content()
    body = content.decode("utf-8", errors="ignore") if content else None

    event: dict = {
        "type": "request",
        "url": flow.request.pretty_url,
        "method": flow.request.method,
        "body": body,
    }
    if anonymized:
        event["process_time"] = t1 - t0
        print("[anonymizer] Process time: " + str(t1 - t0) + " seconds")

    socket.broadcast(event)


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
    socket = get_event_socket()
    content = flow.response.get_content()
    body = content.decode("utf-8", errors="ignore") if content else None

    socket.broadcast({
        "type": "response",
        "url": flow.response.status_code,
        "body": body,
    })

    # Deanonymize non-streaming responses before they reach the client
    apply_response_rules(flow, rules.response_rules)
