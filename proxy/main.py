"""
MITM Proxy for intercepting Claude Desktop traffic.
- Anonymizes PII in outgoing prompts (requests)
- Deanonymizes placeholders in Claude's streamed responses (SSE)
- MCP reverse proxy: deanonymizes Claude args → MCP server, re-anonymizes tool results → Claude
"""
import asyncio
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mitmproxy import http

import proxy.engine as engine
from proxy.anxious_filter import anxious_filter, trigger_anxious_filter
from proxy.cache import start_cache_prune_task
from proxy.control_socket import init_control_socket
from proxy.engine import DLPProxy, ProxyOptions, make_deanon_chunk
from proxy.entity_cache_log import init_entities_log
from proxy.entity_finder.ner_finder import NEREntityFinder
from proxy.entity_finder.presidio_finder import PresidioEntityFinder
from proxy.mappings import Mappings
from proxy.rules import load_rules

proxy = DLPProxy(
    mappings=Mappings(),
    rules=load_rules(),
    finders=[
        PresidioEntityFinder(),
        NEREntityFinder(),
    ],
    options=ProxyOptions(
        anxious_filter=True,
        save_redacted_images=True,
        inject_system_prompt=True,
        save_requests=True,
    ),
)

# Thread pool for offloading blocking anonymisation work
_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def running():
    init_entities_log()  # Initialize entity cache log for validator
    await init_control_socket(proxy)
    start_cache_prune_task()


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

# Use relative paths from repo root
repo_root = Path(__file__).parent.parent
_REQUESTS_LOG = repo_root / "data" / "requests-sample.jsonl"
_IGNORE_DIR = repo_root / "data" / "ignore"


def _log_request(flow: http.HTTPFlow) -> None:
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


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

async def request(flow: http.HTTPFlow):
    try:
        # Force uncompressed responses so stream callbacks receive plain text
        flow.request.headers.pop("accept-encoding", None)

        url = flow.request.pretty_url.split("?")[0]

        # Block blacklisted URLs
        for blocked in proxy.rules.blocked_urls:
            if blocked.url_pattern.fullmatch(url):
                print(f"[proxy] BLOCKED: {url}")
                flow.response = http.Response.make(
                    403,
                    "Blocked by proxy: this endpoint is not allowed.",
                    {"Content-Type": "text/plain"},
                )
                return

        if proxy.options.save_requests:
            _log_request(flow)

        # MCP reverse proxy: Claude sends redacted args → deanonymize before forwarding
        if any(r.url_pattern.fullmatch(url) for r in proxy.rules.anonymise_responses):
            try:
                content = flow.request.get_content()
                new_content = await engine.deanonymize_message(proxy, content)
                if new_content is not None:
                    flow.request.set_content(new_content)
            except Exception as e:
                print(f"[proxy] Error deanonymizing MCP request: {e}")
                traceback.print_exc()
                flow.response = http.Response.make(
                    502,
                    "Bad Gateway: proxy processing error",
                    {"Content-Type": "text/plain"},
                )
                return  # DO NOT forward the request
            return  # skip anonymization and anxious filter for MCP requests

        # Anonymize outgoing requests in a thread to avoid blocking other requests
        content = flow.request.get_content()
        try:
            loop = asyncio.get_event_loop()
            new_content = await loop.run_in_executor(
                _executor,
                lambda: asyncio.run(engine.anonymize_message(
                    proxy, flow.request.headers, content, url, proxy.rules.anonymise_requests
                ))
            )
            if new_content is not None:
                flow.request.set_content(new_content)
        except Exception as e:
            print(f"[proxy] Error anonymizing request: {e}")
            traceback.print_exc()
            flow.response = http.Response.make(
                502,
                "Bad Gateway: proxy processing error",
                {"Content-Type": "text/plain"},
            )
            return  # DO NOT forward the request

        # Anxious filter: warn if known sensitive values still present after anonymization
        try:
            if proxy.options.anxious_filter and any(p.match(url) for p in proxy.rules.anxious_filter_domains):
                body = flow.request.get_content().decode("utf-8", errors="ignore")
                triggered, entities = anxious_filter(proxy.mappings, body)
                if triggered:
                    trigger_anxious_filter(url, flow, entities)
        except Exception as e:
            print(f"[proxy] Error in anxious filter: {e}")
            traceback.print_exc()
            # Don't block request on anxious filter error, just log it

    except Exception as e:
        print(f"[proxy] Unexpected error in request processing: {e}")
        traceback.print_exc()
        flow.response = http.Response.make(
            502,
            "Bad Gateway: proxy processing error",
            {"Content-Type": "text/plain"},
        )


async def responseheaders(flow: http.HTTPFlow):
    """Enable streaming for SSE responses so chunks are processed incrementally."""
    try:
        if "text/event-stream" not in flow.response.headers.get("content-type", ""):
            return

        url = flow.request.pretty_url.split("?")[0]
        sse_fields = None
        for rule in proxy.rules.deanonymise_responses:
            if rule.url_pattern.fullmatch(url) and rule.sse_fields:
                sse_fields = rule.sse_fields
                break

        flow.response.stream = make_deanon_chunk(proxy, sse_fields)
    except Exception as e:
        print(f"[proxy] Error in responseheaders: {e}")
        traceback.print_exc()


async def response(flow: http.HTTPFlow):
    try:
        url = flow.request.pretty_url.split("?")[0]
        content = flow.response.get_content()

        if flow.response.status_code >= 400:
            body = content.decode("utf-8", errors="ignore") if content else ""
            req_content = flow.request.get_content()
            req_body = req_content.decode("utf-8", errors="ignore") if req_content else ""
            print(f"[proxy] {flow.response.status_code} from {flow.request.pretty_url}")
            print(f"[proxy] Response: {body}")
            try:
                _IGNORE_DIR.mkdir(parents=True, exist_ok=True)
                dump_path = _IGNORE_DIR / f"last_{flow.response.status_code}_body.json"
                with open(dump_path, "w") as f:
                    f.write(req_body)
                print(f"[proxy] Full request body saved to {dump_path}")
            except Exception as e:
                print(f"[proxy] Could not save error body: {e}")

        # MCP reverse proxy: anonymize tool results before they reach Claude
        if any(r.url_pattern.fullmatch(url) for r in proxy.rules.anonymise_responses):
            try:
                loop = asyncio.get_event_loop()
                new_content = await loop.run_in_executor(
                    _executor,
                    lambda: asyncio.run(engine.anonymize_message(
                        proxy, flow.response.headers, content, url, proxy.rules.anonymise_responses
                    ))
                )
                if new_content is not None:
                    flow.response.set_content(new_content)
            except Exception as e:
                print(f"[proxy] Error anonymizing response: {e}")
                traceback.print_exc()
                # Still send response but log error
            return

        # Deanonymize non-streaming responses before they reach the client
        try:
            new_content = await engine.deanonymize_message(proxy, content)
            if new_content is not None:
                flow.response.set_content(new_content)
        except Exception as e:
            print(f"[proxy] Error deanonymizing response: {e}")
            traceback.print_exc()
            # Still send response but log error
    except Exception as e:
        print(f"[proxy] Unexpected error in response processing: {e}")
        traceback.print_exc()
