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

import jq as _jq

from mitmproxy import http
from mitmproxy.http import Request

import claude_system_prompt
import image_anonymizer
import text_anonymizer
from mappings import Mappings
from rules import load_rules, RequestRule

mappings = Mappings()
rules = load_rules()

_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "redacted_images")

# Mutable state shared with the control socket
state = {
    "anon_enabled": True,
    "deanon_enabled": False,
    "anxious_enabled": True,
    "save_images": True,
    "system_prompt_enabled": True,
    "mappings": mappings,
}


def _log_anonymization_diff(original: str, new_content: str) -> None:
    if new_content == original:
        print(f"[anonymizer] WARNING: body unchanged after anonymization ({len(original)} chars)")
        return
    orig_tokens = set(re.findall(r'\[[A-Z_]+_\d+\]', original))
    new_tokens = set(re.findall(r'\[[A-Z_]+_\d+\]', new_content))
    injected = new_tokens - orig_tokens
    if not injected:
        print("[anonymizer] Body changed (system prompt injected, no new tokens)")
        return
    print(f"[anonymizer] Tokens introduced: {injected}")
    for tok in list(injected)[:10]:
        idx = new_content.find(tok)
        print(f"[anonymizer] '{tok}' context: ...{new_content[max(0,idx-80):idx+len(tok)+80]}...")


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
    """Return a stream callback that deanonymizes SSE chunks.

    Buffers incomplete redacted tokens (e.g. ``[PER``) across chunk boundaries
    so that a token split like ``[PER`` + ``SON_0]`` is reassembled before the
    regex tries to match it.
    """
    # Pattern that detects an incomplete token at the end of a chunk:
    # an opening '[' followed by uppercase/underscore/digits but no closing ']'.
    _PARTIAL_TOKEN_RE = re.compile(r'\[[A-Z_0-9]+$')

    leftover = ""

    def deanon_chunk(chunk: bytes) -> bytes:
        nonlocal leftover
        text = chunk.decode("utf-8", errors="replace")

        if not state["deanon_enabled"]:
            return chunk

        # Prepend any leftover from the previous chunk
        text = leftover + text
        leftover = ""

        # Check if the chunk ends with an incomplete token
        m = _PARTIAL_TOKEN_RE.search(text)
        if m:
            leftover = text[m.start():]
            text = text[:m.start()]

        if not text:
            return b""

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


def _normalize_path(raw_path: list) -> list:
    """Collapse jq slice descriptor objects into plain integer indices.

    jq's path() represents a slice like [1:] as {"start": 1, "end": null}
    followed by an integer index into that slice.  We flatten these pairs
    into a single absolute index so _get_at / _set_at only deal with
    strings (dict keys) and ints (list indices).
    """
    result = []
    i = 0
    while i < len(raw_path):
        key = raw_path[i]
        if isinstance(key, dict) and "start" in key:
            start = key.get("start") or 0
            if i + 1 < len(raw_path) and isinstance(raw_path[i + 1], int):
                result.append(start + raw_path[i + 1])
                i += 2
                continue
        result.append(key)
        i += 1
    return result


def _get_at(data, path: list):
    for key in path:
        data = data[key]
    return data


def _set_at(data, path: list, value):
    if not path:
        return value
    key, rest = path[0], path[1:]
    if isinstance(data, list):
        result = list(data)
        result[key] = _set_at(result[key], rest, value)
        return result
    result = dict(data)
    result[key] = _set_at(result[key], rest, value)
    return result


def _apply_deanon_paths(data, sensitive_fields):
    """Apply deanonymization at jq path expressions (or recursively if True)."""
    if sensitive_fields is True:
        return _deanonymize_recursive(data)
    for expr in sensitive_fields:
        try:
            raw_paths = _jq.all(f'path({expr})', data)
        except Exception:
            continue
        for raw_path in raw_paths:
            path = _normalize_path(raw_path)
            try:
                value = _get_at(data, path)
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(value, str):
                data = _set_at(data, path, text_anonymizer.deanonymize_text(value, mappings))
            elif isinstance(value, (list, dict)):
                data = _set_at(data, path, _deanonymize_recursive(value))
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


async def _apply_paths(data, sensitive_fields):
    """Anonymize every value reached by each jq path expression (or all if True).

    For each expression, jq resolves it to a set of concrete paths.  String
    leaves are passed to text_anonymizer; list/dict leaves to _anonymize_recursive
    (which handles typed Anthropic content blocks).  Because jq iterates only
    actual elements in the data, there is no double-processing risk from
    overlapping path expressions.
    """
    if sensitive_fields is True:
        return await _anonymize_recursive(data)

    for expr in sensitive_fields:
        try:
            raw_paths = _jq.all(f'path({expr})', data)
        except Exception:
            continue
        for raw_path in raw_paths:
            path = _normalize_path(raw_path)
            try:
                value = _get_at(data, path)
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(value, str):
                new_value = await text_anonymizer.anonymize_text(value, mappings)
            elif isinstance(value, (list, dict)):
                new_value = await _anonymize_recursive(value)
            else:
                continue
            data = _set_at(data, path, new_value)
    return data


# ---------------------------------------------------------------------------
# Text collection helpers (for pre-warming entity cache across all fields)
# ---------------------------------------------------------------------------

def _collect_strings(value) -> list[str]:
    """Recursively collect all string values, skipping image/binary blocks."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_collect_strings(item))
        return result
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type == "image":
            return []
        if block_type == "text":
            t = value.get("text", "")
            return [t] if t else []
        if block_type == "document":
            source = value.get("source", {})
            if source.get("type") == "text":
                d = source.get("data", "")
                return [d] if d else []
            return []
        if block_type in ("thinking", "redacted_thinking"):
            return []
        result = []
        for v in value.values():
            result.extend(_collect_strings(v))
        return result
    return []


def _collect_texts_for_fields(data, sensitive_fields) -> list[str]:
    """Collect all text strings reachable via jq path expressions."""
    if sensitive_fields is True:
        return _collect_strings(data)
    texts = []
    for expr in sensitive_fields:
        try:
            raw_paths = _jq.all(f'path({expr})', data)
        except Exception:
            continue
        for raw_path in raw_paths:
            path = _normalize_path(raw_path)
            try:
                value = _get_at(data, path)
            except (KeyError, IndexError, TypeError):
                continue
            texts.extend(_collect_strings(value))
    return texts


# ---------------------------------------------------------------------------
# Body-format handlers
# ---------------------------------------------------------------------------

async def _apply_rule_json(request: Request, rule: RequestRule, url: str = ""):
    raw = request.get_content()
    if not raw:
        return
    body = json.loads(raw)
    original = json.dumps(body, ensure_ascii=False)
    # Pre-warm: run finder[0] on ALL texts, then finder[1], etc. before per-field processing.
    await text_anonymizer.prewarm_cache(_collect_texts_for_fields(body, rule.sensitive_fields), mappings)
    body = await _apply_paths(body, rule.sensitive_fields)
    if state["system_prompt_enabled"]:
        body = claude_system_prompt.inject(body, url)
    new_content = json.dumps(body, ensure_ascii=False)
    _log_anonymization_diff(original, new_content)
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

    # Pre-warm: collect all text strings from all parts, then run finders in batch.
    # Wrap each part as {field_name: parsed} so sensitive_fields paths work directly.
    all_texts: list[str] = []
    for part in parts[1:-1]:
        sep_pos = part.find(b"\r\n\r\n")
        if sep_pos == -1:
            continue
        part_headers_text = part[:sep_pos].decode("utf-8", errors="replace")
        ct_match = re.search(r'content-type:\s*(\S+)', part_headers_text, re.IGNORECASE)
        part_ct = (ct_match.group(1).rstrip(";") if ct_match else "")
        if part_ct.startswith("image/"):
            continue
        content_bytes = part[sep_pos + 4:]
        if content_bytes.endswith(b"\r\n"):
            content_bytes = content_bytes[:-2]
        try:
            content_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        try:
            parsed = json.loads(content_text)
            name_match = re.search(r'name="([^"]*)"', part_headers_text, re.IGNORECASE)
            field_name = name_match.group(1) if name_match else ""
            wrapped = {field_name: parsed} if rule.sensitive_fields is not True else parsed
            all_texts.extend(_collect_texts_for_fields(wrapped, rule.sensitive_fields))
        except (json.JSONDecodeError, ValueError):
            all_texts.append(content_text)
    await text_anonymizer.prewarm_cache(all_texts, mappings)

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
            if rule.sensitive_fields is True:
                anonymized = await _anonymize_recursive(parsed)
            else:
                # Wrap as {field_name: ...} so sensitive_fields paths resolve correctly,
                # then unwrap after applying.
                wrapped = {field_name: parsed}
                wrapped = await _apply_paths(wrapped, rule.sensitive_fields)
                anonymized = wrapped[field_name]
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

    await text_anonymizer.prewarm_cache(_collect_texts_for_fields(parsed, rule.sensitive_fields), mappings)
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

    t0 = time.time()
    print(f"[anonymizer] Rule matched: {matched_rule.url_pattern.pattern} for {url}")
    content_type = request.headers.get("content-type", "")

    try:
        if "multipart/form-data" in content_type:
            await _apply_rule_multipart(request, matched_rule)
        elif "application/x-www-form-urlencoded" in content_type:
            await _apply_rule_urlencoded(request, matched_rule)
        else:
            await _apply_rule_json(request, matched_rule, url)
        t1 = time.time()
        print(f"[anonymizer] Done processing {url}.")
        print(f"[anonymizer] Took {t1 - t0} seconds to process.")
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
