"""
MITM Proxy for intercepting Claude Desktop traffic.
- Anonymizes PII in outgoing prompts (requests)
- Deanonymizes placeholders in Claude's streamed responses (SSE)
- MCP reverse proxy: deanonymizes Claude args → MCP server, re-anonymizes tool results → Claude
"""
import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mitmproxy import http
from mitmproxy.proxy.layers.tls import ClientHelloData

import proxy.engine as engine
from proxy.anxious_filter import anxious_filter, trigger_anxious_filter
from proxy.cache import start_cache_prune_task
from proxy.engine import DLPProxy, ProxyOptions, make_deanon_chunk
from proxy.entity_cache_log import init_entities_log
from proxy.entity_finder.ner_finder import NEREntityFinder
from proxy.entity_finder.presidio_finder import PresidioEntityFinder
from proxy.mappings import Mappings
from proxy.rules import load_rules, ProxyRules

log = logging.getLogger("proxy")

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
# TLS passthrough — only intercept hosts that appear in rules.jsonc
# ---------------------------------------------------------------------------

def _build_allowed_host_patterns(rules: ProxyRules) -> list[re.Pattern]:
    """
    Extract host-only regex patterns from every URL pattern in rules.
    Only HTTPS patterns matter — HTTP flows are never TLS-intercepted.
    """
    host_regexes: set[str] = set()
    all_patterns = (
        [r.url_pattern.pattern for r in rules.anonymise_requests]
        + [r.url_pattern.pattern for r in rules.anonymise_responses]
        + [r.url_pattern.pattern for r in rules.deanonymise_responses]
        + [r.url_pattern.pattern for r in rules.blocked_urls]
        + [p.pattern for p in rules.known_safe_routes]
    )
    for pat in all_patterns:
        if pat.startswith("https://"):
            host_part = pat[len("https://"):]
            # Take the regex up to (but not including) the first literal slash
            host_regex = host_part.split("/")[0]
            host_regexes.add(host_regex)
    return [re.compile(h) for h in host_regexes]


_allowed_host_patterns = _build_allowed_host_patterns(proxy.rules)


async def tls_clienthello(data: ClientHelloData) -> None:
    """Pass through TLS for any host not covered by rules.jsonc."""
    sni = data.client_hello.sni
    if not sni:
        return  # No SNI → unknown host, intercept (safe default)
    for pattern in _allowed_host_patterns:
        if pattern.fullmatch(sni):
            return  # Whitelisted → intercept normally
    # Host not in any rule → transparent passthrough (no cert substitution)
    data.ignore_connection = True
    # print(f"[proxy] TLS passthrough (not in rules): {sni}")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def running():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("proxy").setLevel(logging.DEBUG)
    init_entities_log()  # Initialize entity cache log for validator
    # await init_control_socket(proxy)
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
        if not content:
            return
        # Splice the raw body bytes directly into the JSONL record without
        # parsing them into Python objects.  Parsing a 500 MB JSON body creates
        # a 2-4 GB Python dict — the single largest allocation before NER runs.
        meta = json.dumps({
            "url": flow.request.pretty_url,
            "method": flow.request.method,
            "headers": dict(flow.request.headers),
        }, ensure_ascii=False)
        body_str = content.decode("utf-8", errors="replace")
        os.makedirs(os.path.dirname(_REQUESTS_LOG), exist_ok=True)
        with open(_REQUESTS_LOG, "a", encoding="utf-8") as f:
            # meta ends with "}" — insert "body" before the closing brace.
            f.write(meta[:-1] + ', "body": ' + body_str + '}\n')
    except Exception as e:
        log.error("Could not log request: %s", e)


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
                log.warning("BLOCKED: %s", url)
                flow.response = http.Response.make(
                    403,
                    "Blocked by proxy: this endpoint is not allowed.",
                    {"Content-Type": "text/plain"},
                )
                return

        # Default-deny: intercepted hosts must match an explicit rule or be
        # whitelisted in `known_safe_routes`. Otherwise the URL carries unknown
        # payload shape and may leak PII we can't see — reject with 403.
        if not proxy.rules.matches_any_rule(url):
            log.warning("BLOCKED (unknown route, not in rules or known_safe_routes): %s", url)
            flow.response = http.Response.make(
                403,
                "Blocked by proxy: route not covered by anonymisation rules. "
                "Add it to assets/rules.jsonc (anonymise.requests or known_safe_routes) if legitimate.",
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
                log.error("Error deanonymizing MCP request: %s", e, exc_info=True)
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
            log.error("Error anonymizing request: %s", e, exc_info=True)
            flow.response = http.Response.make(
                502,
                "Bad Gateway: proxy processing error",
                {"Content-Type": "text/plain"},
            )
            return  # DO NOT forward the request

        # Anxious filter: last line of defence — if a known sensitive value is
        # still present after anonymisation, refuse to forward. Fail CLOSED:
        # any exception inside the filter is treated as "unable to verify" and
        # blocks the request, because silently forwarding on filter failure
        # defeats the purpose of having the filter at all.
        try:
            if proxy.options.anxious_filter and any(p.match(url) for p in proxy.rules.anxious_filter_domains):
                body = flow.request.get_content().decode("utf-8", errors="ignore")
                triggered, entities = anxious_filter(proxy.mappings, body)
                if triggered:
                    trigger_anxious_filter(url, flow, entities)
                    flow.response = http.Response.make(
                        502,
                        "Bad Gateway: anxious filter detected unmasked sensitive values "
                        "after anonymisation; request refused.",
                        {"Content-Type": "text/plain"},
                    )
                    return
        except Exception as e:
            log.error("Error in anxious filter — blocking request (fail closed): %s", e, exc_info=True)
            flow.response = http.Response.make(
                502,
                "Bad Gateway: anxious filter failed; request refused (fail closed).",
                {"Content-Type": "text/plain"},
            )
            return

    except Exception as e:
        log.error("Unexpected error in request processing: %s", e, exc_info=True)
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
        log.error("Error in responseheaders: %s", e, exc_info=True)


async def response(flow: http.HTTPFlow):
    try:
        url = flow.request.pretty_url.split("?")[0]
        content = flow.response.get_content()

        if flow.response.status_code >= 400:
            body = content.decode("utf-8", errors="ignore") if content else ""
            req_content = flow.request.get_content()
            req_body = req_content.decode("utf-8", errors="ignore") if req_content else ""
            log.warning("HTTP %d from %s — %s", flow.response.status_code, flow.request.pretty_url, body[:200])
            try:
                _IGNORE_DIR.mkdir(parents=True, exist_ok=True)
                dump_path = _IGNORE_DIR / f"last_{flow.response.status_code}_body.json"
                with open(dump_path, "w") as f:
                    f.write(req_body)
                log.debug("Full request body saved to %s", dump_path)
            except Exception as e:
                log.error("Could not save error body: %s", e)

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
                log.error("Error anonymizing response: %s", e, exc_info=True)
                # Still send response but log error
            return

        # Deanonymize non-streaming text/JSON responses before they reach the client
        # Skip binary content types — decoding them as UTF-8 corrupts the data
        response_ct = flow.response.headers.get("content-type", "")
        is_text = any(t in response_ct for t in ("application/json", "text/"))
        if is_text:
            try:
                new_content = await engine.deanonymize_message(proxy, content)
                if new_content is not None:
                    flow.response.set_content(new_content)
            except Exception as e:
                log.error("Error deanonymizing response: %s", e, exc_info=True)
                # Still send response but log error
    except Exception as e:
        log.error("Unexpected error in response processing: %s", e, exc_info=True)
