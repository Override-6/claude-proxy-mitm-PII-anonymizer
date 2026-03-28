"""
Rule application logic: anonymize requests and deanonymize responses.

Separated from proxy.py (which only contains mitmproxy hooks) so this
module can be tested and imported independently.
"""

import asyncio
import base64
import json
import os
import re
import time

from mitmproxy import http
from mitmproxy.http import Request

import claude_system_prompt
import image_anonymizer
import text_anonymizer
from event_socket import get_event_socket
from mappings import Mappings
from rules import load_rules, RequestRule

mappings = Mappings()
rules = load_rules()

_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "ignore", "redacted_images")

# Mutable state shared with the control socket
state = {
    "anon_enabled": True,
    "deanon_enabled": False,
    "save_images": False,
    "system_prompt_enabled": True,
    "mappings": mappings,
}


def _save_image(image_bytes: bytes, ocr_text: str, image_hash: str) -> None:
    os.makedirs(_IMAGES_DIR, exist_ok=True)
    stem = image_hash[:24]
    img_path = os.path.join(_IMAGES_DIR, f"{stem}.png")
    if os.path.exists(img_path):
        return
    txt_path = os.path.join(_IMAGES_DIR, f"{stem}.txt")
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)
    print(f"[image_anonymizer] Saved → {img_path}")


# ---------------------------------------------------------------------------
# Deanonymization helpers
# ---------------------------------------------------------------------------

def _deanon_sse_targeted(text: str, sse_fields: list[list[str]]) -> str:
    """Parse SSE data lines as JSON and deanonymize only targeted fields."""
    lines = text.split("\n")
    result = []
    for line in lines:
        if line.startswith("data: "):
            payload = line[6:]
            try:
                data = json.loads(payload)
                data = _apply_deanon_paths(data, sse_fields)
                result.append("data: " + json.dumps(data, ensure_ascii=False))
            except (json.JSONDecodeError, ValueError):
                # Partial or non-JSON data line — fall back to regex
                result.append("data: " + text_anonymizer.deanonymize_text(payload, mappings))
        else:
            result.append(line)
    return "\n".join(result)


def make_deanon_chunk(sse_fields: list[list[str]] | None):
    """Return a stream callback that deanonymizes SSE chunks."""
    def deanon_chunk(chunk: bytes) -> bytes:
        text = chunk.decode("utf-8", errors="replace")

        socket = get_event_socket()
        socket.broadcast({
            "type": "sse_event",
            "text": text,
        })

        if not state["deanon_enabled"]:
            return chunk

        if sse_fields:
            text = _deanon_sse_targeted(text, sse_fields)
        else:
            text = text_anonymizer.deanonymize_text(text, mappings)

        return text.encode("utf-8")
    return deanon_chunk


def _deanonymize_recursive(value):
    """Deanonymize all strings in *value*, recursing into lists and dicts."""
    if isinstance(value, str):
        return text_anonymizer.deanonymize_text(value, mappings)
    if isinstance(value, list):
        return [_deanonymize_recursive(item) for item in value]
    if isinstance(value, dict):
        return {k: _deanonymize_recursive(v) for k, v in value.items()}
    return value


def _deanonymize_at_path(data, segments: list[str]):
    """Walk *data* along jq-style *segments*, deanonymize at the leaf."""
    if not segments:
        return _deanonymize_recursive(data)

    seg, rest = segments[0], segments[1:]

    if seg == "[]":
        if isinstance(data, list):
            return [_deanonymize_at_path(item, rest) for item in data]
        return data

    if isinstance(data, dict) and seg in data:
        result = dict(data)
        result[seg] = _deanonymize_at_path(data[seg], rest)
        return result
    return data


def _apply_deanon_paths(data, sensitive_fields):
    """Apply deanonymization along parsed jq paths (or True)."""
    if sensitive_fields is True:
        return _deanonymize_recursive(data)
    for segments in sensitive_fields:
        data = _deanonymize_at_path(data, segments)
    return data


# ---------------------------------------------------------------------------
# Anonymization helpers
# ---------------------------------------------------------------------------

async def _anonymize_recursive(value):
    """Anonymize *value* recursively.

    Dicts with a known Anthropic content-block 'type' are dispatched to a
    dedicated handler that knows exactly which fields to process and with
    which anonymizer.  Unknown dicts fall back to generic key-by-key
    recursion.  Raw strings are passed through text_anonymizer.
    """
    if isinstance(value, str):
        return await text_anonymizer.anonymize_text(value, mappings)
    if isinstance(value, list):
        return [await _anonymize_recursive(item) for item in value]
    if isinstance(value, dict):
        handler = _BLOCK_HANDLERS.get(value.get("type"))
        if handler:
            return await handler(value)
        return {k: await _anonymize_recursive(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Per-block-type handlers
# Each handler is responsible for exactly the fields that may contain PII
# and routes them to the right anonymizer.  Fields not mentioned are copied
# through unchanged (no unnecessary recursion into binary/structural data).
# ---------------------------------------------------------------------------

async def _block_text(block: dict) -> dict:
    """{"type":"text", "text": "..."}"""
    return {**block, "text": await text_anonymizer.anonymize_text(block.get("text", ""), mappings)}


async def _block_image(block: dict) -> dict:
    """{"type":"image", "source": {"type": "base64"|"url"|"file", ...}}"""
    source = block.get("source") or {}
    source_type = source.get("type")

    if source_type == "base64":
        image_bytes = base64.b64decode(source["data"])
        redacted_bytes, ocr_text = await asyncio.to_thread(
            image_anonymizer.anonymize_image, image_bytes, mappings
        )
        # image_anonymizer always outputs PNG — update media_type to match
        if state["save_images"]:
            _save_image(redacted_bytes, ocr_text, image_anonymizer._image_hash(image_bytes))
        new_source = {**source,
                      "data": base64.b64encode(redacted_bytes).decode("ascii"),
                      "media_type": "image/png"}
        return {**block, "source": new_source}

    # url / file sources: image lives outside the request — pass through
    return block


async def _block_document(block: dict) -> dict:
    """{"type":"document", "source": {"type": "text"|"base64"|"url"|"file", ...}}"""
    source = block.get("source") or {}
    source_type = source.get("type")

    if source_type == "text":
        anon = await text_anonymizer.anonymize_text(source.get("data", ""), mappings)
        return {**block, "source": {**source, "data": anon}}

    # base64 / url / file: binary or remote — pass through
    return block


async def _block_tool_use(block: dict) -> dict:
    """{"type":"tool_use", "input": {...}}  — anonymize the input object."""
    return {**block, "input": await _anonymize_recursive(block.get("input") or {})}


async def _block_tool_result(block: dict) -> dict:
    """{"type":"tool_result", "content": str | [...]}"""
    content = block.get("content")
    if isinstance(content, str):
        anon = await text_anonymizer.anonymize_text(content, mappings)
    elif isinstance(content, list):
        anon = [await _anonymize_recursive(item) for item in content]
    else:
        anon = content
    return {**block, "content": anon}


async def _block_skip(block: dict) -> dict:
    """thinking / redacted_thinking — leave intact (cryptographic signature)."""
    return block


_BLOCK_HANDLERS = {
    "text":               _block_text,
    "image":              _block_image,
    "document":           _block_document,
    "tool_use":           _block_tool_use,
    "tool_result":        _block_tool_result,
    "thinking":           _block_skip,
    "redacted_thinking":  _block_skip,
}


async def _anonymize_at_path(data, segments: list[str], string_only: bool = False):
    """Walk *data* along jq-style *segments*, anonymize at the leaf.

    string_only=True  — the path has more-specific sub-paths in the same rule
                        (e.g. .messages[].content has .messages[].content[].text).
                        Only string leaves are anonymized; list/dict leaves are
                        skipped to avoid double-processing with the sub-paths.

    string_only=False — no sub-paths exist for this path, so list/dict leaves
                        are handed to _anonymize_recursive (e.g. tool_result
                        .content, tool_use .input).
    """
    if not segments:
        if isinstance(data, str):
            return await text_anonymizer.anonymize_text(data, mappings)
        if string_only:
            return data
        return await _anonymize_recursive(data)

    seg, rest = segments[0], segments[1:]

    if seg == "[]":
        if isinstance(data, list):
            return [await _anonymize_at_path(item, rest, string_only) for item in data]
        return data

    if isinstance(data, dict) and seg in data:
        result = dict(data)
        result[seg] = await _anonymize_at_path(data[seg], rest, string_only)
        return result
    return data


async def _apply_paths(data, sensitive_fields):
    """Apply a list of parsed jq paths (or True) to *data*.

    Pre-computes which paths are prefixes of other paths in the same rule and
    passes string_only=True for those, preventing double-processing.
    """
    if sensitive_fields is True:
        return await _anonymize_recursive(data)

    # A path has sub-paths if any other path in the rule extends it.
    has_subpaths = {
        i
        for i, si in enumerate(sensitive_fields)
        if any(
            i != j and len(sj) > len(si) and sj[:len(si)] == si
            for j, sj in enumerate(sensitive_fields)
        )
    }

    for i, segments in enumerate(sensitive_fields):
        data = await _anonymize_at_path(data, segments, string_only=(i in has_subpaths))
    return data


# ---------------------------------------------------------------------------
# Body-format handlers
# ---------------------------------------------------------------------------

async def _apply_rule_json(request: Request, rule: RequestRule, url: str = ""):
    raw = request.get_content()
    if not raw:
        return
    body = json.loads(raw)
    original = json.dumps(body, ensure_ascii=False)
    body = await _apply_paths(body, rule.sensitive_fields)
    if state["system_prompt_enabled"]:
        body = claude_system_prompt.inject(body, url)
    new_content = json.dumps(body, ensure_ascii=False)
    if new_content != original:
        print(f"[anonymizer] Body modified (original length {len(original)} → {len(new_content)})")
    else:
        print(f"[anonymizer] WARNING: body unchanged after anonymization ({len(original)} chars)")
    request.set_content(new_content.encode("utf-8"))


async def _apply_rule_multipart(request: Request, rule: RequestRule):
    """Process multipart bodies by splitting on the boundary and modifying part
    content in-place, keeping per-part headers (filename, Content-Type) intact."""
    ct = request.headers.get("content-type", "")
    boundary_match = re.search(r'boundary=([^\s;]+)', ct)
    if not boundary_match:
        return
    boundary = boundary_match.group(1).strip('"')
    delimiter = ("--" + boundary).encode()

    raw = request.get_content()
    parts = raw.split(delimiter)
    # parts[0]  = preamble (empty)
    # parts[1:-1] = actual parts
    # parts[-1]  = "--\r\n" epilogue

    new_parts = [parts[0]]

    for part in parts[1:-1]:
        sep = b"\r\n\r\n"
        sep_pos = part.find(sep)
        if sep_pos == -1:
            new_parts.append(part)
            continue

        part_headers_bytes = part[:sep_pos]
        content_with_suffix = part[sep_pos + 4:]

        # The trailing \r\n belongs to the boundary delimiter, not the content
        if content_with_suffix.endswith(b"\r\n"):
            content_bytes = content_with_suffix[:-2]
            suffix = b"\r\n"
        else:
            content_bytes = content_with_suffix
            suffix = b""

        part_headers_text = part_headers_bytes.decode("utf-8", errors="replace")
        name_match = re.search(r'name="([^"]*)"', part_headers_text, re.IGNORECASE)
        field_name = name_match.group(1) if name_match else ""

        # Decide whether and how to process this part
        if rule.sensitive_fields is True:
            sub_paths = None  # anonymize everything
        elif isinstance(rule.sensitive_fields, list):
            matching = [segs[1:] for segs in rule.sensitive_fields if segs and segs[0] == field_name]
            if not matching:
                new_parts.append(part)
                continue
            sub_paths = matching
        else:
            new_parts.append(part)
            continue

        # Check if this part is an image — route to image_anonymizer
        ct_match = re.search(r'content-type:\s*(\S+)', part_headers_text, re.IGNORECASE)
        part_ct = ct_match.group(1).rstrip(";") if ct_match else ""
        if part_ct.startswith("image/"):
            redacted_bytes, ocr_text = await asyncio.to_thread(
                image_anonymizer.anonymize_image, content_bytes, mappings
            )
            # image_anonymizer always outputs PNG — rewrite Content-Type header
            if state["save_images"]:
                _save_image(redacted_bytes, ocr_text, image_anonymizer._image_hash(content_bytes))
            updated_headers = re.sub(
                r'(content-type:\s*)\S+',
                r'\g<1>image/png',
                part_headers_bytes.decode("utf-8", errors="replace"),
                flags=re.IGNORECASE,
            ).encode("utf-8")
            new_parts.append(updated_headers + sep + redacted_bytes + suffix)
            continue

        try:
            content_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            new_parts.append(part)  # binary — skip
            continue

        try:
            parsed = json.loads(content_text)
            if sub_paths is None:
                anonymized = await _anonymize_recursive(parsed)
            else:
                anonymized = parsed
                for segs in sub_paths:
                    anonymized = await _anonymize_at_path(anonymized, segs)
            new_content = json.dumps(anonymized, ensure_ascii=False).encode("utf-8")
        except (json.JSONDecodeError, ValueError):
            anonymized_text = await text_anonymizer.anonymize_text(content_text, mappings)
            new_content = anonymized_text.encode("utf-8")

        new_parts.append(part_headers_bytes + sep + new_content + suffix)

    new_parts.append(parts[-1])
    request.set_content(delimiter.join(new_parts))


async def _apply_rule_urlencoded(request: Request, rule: RequestRule):
    form = request.urlencoded_form

    parsed = {}
    for key in list(form.keys()):
        try:
            parsed[key] = json.loads(form[key])
        except (json.JSONDecodeError, ValueError):
            parsed[key] = form[key]

    parsed = await _apply_paths(parsed, rule.sensitive_fields)

    for key in list(form.keys()):
        if key not in parsed:
            continue
        val = parsed[key]
        form[key] = val if isinstance(val, str) \
            else json.dumps(val, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def apply_request_rules(flow: http.HTTPFlow, request_rules: list[RequestRule]) -> bool:
    if not state["anon_enabled"]:
        return False

    request = flow.request
    url = request.pretty_url.split("?")[0]
    matched_rule = None
    for rule in request_rules:
        if rule.url_pattern.fullmatch(url):
            matched_rule = rule
            break

    if not matched_rule:
        return False

    print(f"[anonymizer] Rule matched: {matched_rule.url_pattern.pattern} for {url}")
    content_type = request.headers.get("content-type", "")

    try:
        if "multipart/form-data" in content_type:
            await _apply_rule_multipart(request, matched_rule)
        elif "application/x-www-form-urlencoded" in content_type:
            await _apply_rule_urlencoded(request, matched_rule)
        else:
            await _apply_rule_json(request, matched_rule, url)
        print(f"[anonymizer] Done processing {url}")
    except Exception as e:
        import traceback
        print(f"[anonymizer] ERROR anonymizing {request.pretty_url}: {e}")
        traceback.print_exc()
        flow.response = http.Response.make(
            502,
            f"Proxy anonymization failed: {e}",
            {"Content-Type": "text/plain"},
        )
    return True

def apply_response_rules(flow: http.HTTPFlow, response_rules: list[RequestRule]):
    if not state["deanon_enabled"]:
        return

    url = flow.request.pretty_url.split("?")[0]
    matched_rule = None
    for rule in response_rules:
        if rule.url_pattern.fullmatch(url):
            matched_rule = rule
            break

    if not matched_rule:
        return

    content_type = flow.response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return  # SSE already handled by deanon_chunk

    content = flow.response.get_content()
    if not content:
        return

    try:
        body = json.loads(content)
        body = _apply_deanon_paths(body, matched_rule.sensitive_fields)
        flow.response.set_content(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    except Exception as e:
        print(f"[deanonymizer] Error processing response for {flow.request.pretty_url}: {e}")
