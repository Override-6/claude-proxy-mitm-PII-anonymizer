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
from text_anonymizer import start_cache_prune_task


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

async def running():
    await init_event_socket()
    await init_control_socket(state)
    start_cache_prune_task()


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

    if flow.response.status_code >= 400:
        import json as _json, os as _os
        req_content = flow.request.get_content()
        req_body = req_content.decode("utf-8", errors="ignore") if req_content else ""
        print(f"[proxy] {flow.response.status_code} from {flow.request.pretty_url}")
        print(f"[proxy] Response: {body}")
        try:
            req_json = _json.loads(req_body)
            print(f"[proxy] Request keys: {list(req_json.keys())}")
            system = req_json.get("system")
            if isinstance(system, list):
                print(f"[proxy] system: {len(system)} blocks, types={[b.get('type') for b in system]}, has_cache_control={[bool(b.get('cache_control')) for b in system]}")
            elif isinstance(system, str):
                print(f"[proxy] system: string len={len(system)}")
            msgs = req_json.get("messages", [])
            print(f"[proxy] messages: {len(msgs)} messages")
            for i, m in enumerate(msgs[-3:], len(msgs) - 3):
                c = m.get("content")
                if isinstance(c, list):
                    print(f"[proxy] messages[{i}] role={m.get('role')} content_types={[b.get('type') for b in c]}")
                else:
                    print(f"[proxy] messages[{i}] role={m.get('role')} content=string len={len(c or '')}")
            # Save full body to file for inspection
            dump_path = "/app/ignore/last_400_body.json"
            _os.makedirs("/app/ignore", exist_ok=True)
            with open(dump_path, "w") as f:
                f.write(req_body)
            print(f"[proxy] Full request body saved to {dump_path}")
        except Exception as e:
            print(f"[proxy] Could not parse request body: {e}")

    socket.broadcast({
        "type": "response",
        "url": flow.response.status_code,
        "body": body,
    })

    # Deanonymize non-streaming responses before they reach the client
    apply_response_rules(flow, rules.response_rules)
