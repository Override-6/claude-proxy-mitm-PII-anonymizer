import base64
import json
import logging
import re
import time
from dataclasses import dataclass

import jq
from mitmproxy.http import Headers

from proxy import cache
from proxy import claude_system_prompt
from proxy.entity_finder import AbstractEntityFinder, Entity
from proxy.entity_finder.mappings_finder import MappingsEntityFinder
from proxy.mappings import Mappings
from proxy.rules import ProxyRules, AnonymiseRule

log = logging.getLogger(__name__)

REDACTED_REGEX = re.compile(r'\[([A-Z_]+)_[0-9]+]')

_mappings_finder = MappingsEntityFinder()
# Compiled jq programs keyed by frozenset of expressions — reused across requests.
_compiled_expand: dict[tuple, object] = {}


@dataclass
class ProxyOptions:
    anxious_filter: bool = True
    save_redacted_images: bool = True
    inject_system_prompt: bool = True
    save_requests: bool = False


@dataclass
class DLPProxy:
    mappings: Mappings
    rules: ProxyRules
    finders: list[AbstractEntityFinder]
    options: ProxyOptions


def expand_paths(obj: dict, expressions: list[str]) -> list[list[str | int]]:
    """Find all paths to string leaves matching any of the given jq expressions.

    All expressions are merged into a single jq call so the full document is
    serialised through jq's C heap only ONCE per request.
    The compiled jq program is cached for the lifetime of the
    process so compilation cost is paid only once per unique rule set.
    """
    key = tuple(expressions)
    if key not in _compiled_expand:
        if len(expressions) == 1 and expressions[0] == ".":
            prog_src = "[path(.. | strings)] | unique"
        else:
            # Combine all expressions into one generator inside path() so jq
            # traverses the document once.
            combined = ", ".join(f"({e} | .. | strings)?" for e in expressions)
            prog_src = f"[path({combined})] | unique"
        _compiled_expand[key] = jq.compile(prog_src)

    try:
        result = _compiled_expand[key].input(obj).first()
        return result if result else []
    except (ValueError, StopIteration):
        return []


_MISSING = object()


def get_values(obj: dict, paths: list[list]) -> list:
    """Extract values at the given paths using pure-Python traversal.

    Returns the raw va
    lue when the path resolves, or the _MISSING sentinel when
    it doesn't. Callers must distinguish "missing" from "empty string" — writing
    back to a missing path via set_values would CREATE the field (see the
    thinking-block bug where jq's path() returns paths to fields that don't
    exist on the object).
    """
    result = []
    for path in paths:
        try:
            val = obj
            for key in path:
                val = val[key]
            result.append(val)
        except (KeyError, IndexError, TypeError):
            result.append(_MISSING)
    return result


def set_values(obj: dict, values: list[tuple[list, str]]) -> dict:
    for path, value in values:
        if not path:
            continue
        try:
            target = obj
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        except (KeyError, IndexError, TypeError):
            pass
    return obj


def _add_non_overlapping(target: list[Entity], candidates: list[Entity]) -> None:
    """Extend target with candidates that don't overlap any entity already in target.

    The naive pattern — checking each candidate against every accepted entity —
    is O(M*K) where M = len(candidates) and K = len(target).  For large texts
    (MappingsEntityFinder runs on ALL values every request) this becomes the
    dominant cost.

    This implementation sorts the combined set by start position and keeps the
    first entity when spans overlap, giving O((M+K) log(M+K)) total.
    """
    if not candidates:
        return
    if not target:
        target.extend(sorted(candidates, key=lambda e: e.start))
        return

    combined = sorted(target + candidates, key=lambda e: e.start)
    target.clear()
    for entity in combined:
        if not target or entity.start >= target[-1].end:
            target.append(entity)


def redact_entities(text: str, entities: list[Entity], mappings: Mappings) -> str:
    if not entities:
        return text
    parts = []
    prev = 0
    for entity in sorted(entities, key=lambda ent: ent.start):
        s, e = entity.start, entity.end
        parts.append(text[prev:s])
        parts.append(mappings.get_or_set_redacted_text(text[s:e], entity.type))
        prev = e
    parts.append(text[prev:])
    return ''.join(parts)


async def _apply_paths(proxy: DLPProxy, data: dict, sensitive_fields: list[str] | bool, url: str = None) -> dict:
    """Anonymize every string value reached by each jq path expression."""
    from entity_cache_log import log_extracted_entities

    expressions = ["."] if sensitive_fields is True else sensitive_fields

    paths = expand_paths(data, expressions)
    if not paths:
        return data
    raw_values = get_values(data, paths)

    # Coerce to strings for the entity pipeline, but remember which paths had
    # a real string value so we only write those back (avoid creating fields
    # at paths that don't actually exist — see deanon thinking-block bug).
    values = [v if isinstance(v, str) else "" for v in raw_values]
    writable = [isinstance(v, str) for v in raw_values]

    # Parallel list of entity lists indexed the same as paths/values.
    # Avoids using paths as dict keys (jq paths can contain dicts for complex expressions).
    cached_results = [cache.get_cached_entities_of_text(v) for v in values]
    entities_list: list[list[Entity]] = [
        list(c) if c is not None else [] for c in cached_results
    ]

    non_cached_indices = [i for i, c in enumerate(cached_results) if c is None]
    non_cached_values = [values[i] for i in non_cached_indices]
    # Run all finders on non-cached texts
    if non_cached_values:
        for finder in proxy.finders:
            for list_idx, entities in zip(non_cached_indices,
                                          finder.find_entities_batch(non_cached_values, proxy.mappings)):
                _add_non_overlapping(entities_list[list_idx], entities)

    # Run MappingsEntityFinder on all texts to catch re-occurrences of known sensitive values
    for i, entities in enumerate(_mappings_finder.find_entities_batch(values, proxy.mappings)):
        _add_non_overlapping(entities_list[i], entities)

    updates: list[tuple[list, str]] = []
    for path_list, value, entities, ok in zip(paths, values, entities_list, writable):
        if not ok:
            continue
        cache.set_cached_entities(value, entities)

        # Log extracted entities for validator reuse (convert path list to string)
        if url and entities:
            path_str = str(path_list)
            log_extracted_entities(url, path_str, value, entities)

        updates.append((path_list, redact_entities(value, entities, proxy.mappings)))

    return set_values(data, updates)


async def _anonymize_base64_images(proxy: DLPProxy, data: dict) -> dict:
    """Find and redact PII in base64-encoded images in the JSON structure.

    Walks through the JSON looking for content blocks with type: "image" and
    source.type: "base64", decodes the base64 data, anonymizes using
    anonymize_image(), and re-encodes.
    """
    from image_anonymizer import anonymize_image

    def walk(obj):
        """Recursively walk the data structure to find and process image blocks."""
        if isinstance(obj, dict):
            # Check if this is an image content block with base64 source
            if (obj.get("type") == "image" and
                    isinstance(obj.get("source"), dict) and
                    obj["source"].get("type") == "base64"):

                source = obj["source"]
                b64_data = source.get("data", "")
                if b64_data:
                    try:
                        # Decode base64 → anonymize → re-encode
                        image_bytes = base64.b64decode(b64_data)
                        anonymized_bytes, _ = anonymize_image(image_bytes, proxy)
                        source["data"] = base64.b64encode(anonymized_bytes).decode("ascii")
                        media_type = source.get("media_type", "image/jpeg")
                        log.debug(
                            f"[anonymizer] Anonymized base64 image ({media_type}): "
                            f"{len(image_bytes)} → {len(anonymized_bytes)} bytes"
                        )
                    except Exception as e:
                        log.warning(f"[anonymizer] Failed to anonymize base64 image: {e}")
                        raise e

            else:
                # Recurse into all values
                for value in obj.values():
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return data


_REDACTED_RE_BYTES = re.compile(rb'\[[A-Z_]+_\d+\]')
_REDACTED_RE_STR = re.compile(r'\[[A-Z_]+_\d+\]')


def _log_anonymization_diff(raw: bytes, new_content: str) -> None:
    # Scan for tokens on the raw bytes directly — avoids keeping a separate
    # re-serialised 'original' string alongside the parsed body dict.
    orig_tokens = {m.group(0).decode() for m in _REDACTED_RE_BYTES.finditer(raw)}
    new_tokens = set(_REDACTED_RE_STR.findall(new_content))
    injected = new_tokens - orig_tokens
    if len(new_content) == len(raw) and not injected:
        print(f"[anonymizer] WARNING: body unchanged after anonymization ({len(raw)} bytes)")
        return
    if not injected:
        print("[anonymizer] Body changed (system prompt injected, no new tokens)")
        return
    print(f"[anonymizer] Tokens introduced: {injected}")
    for tok in list(injected)[:10]:
        idx = new_content.find(tok)
        print(f"[anonymizer] '{tok}' context: ...{new_content[max(0, idx - 80):idx + len(tok) + 80]}...")


async def _apply_rule_json(proxy: DLPProxy, raw: bytes, sensitive_fields: list[str] | bool, url: str) -> bytes:
    if not raw or not raw.strip():
        return raw
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    # Do NOT keep a json.dumps(body) 'original' here — for large conversations
    # that extra string is 1× body size in memory alongside the parsed dict
    # (3-5×) and the final new_content (1×), pushing peak to 6-7× body size.

    body = await _anonymize_base64_images(proxy, body)
    body = await _apply_paths(proxy, body, sensitive_fields, url=url)

    if proxy.options.inject_system_prompt:
        body = claude_system_prompt.inject(body, url)

    new_content = json.dumps(body, ensure_ascii=False)
    _log_anonymization_diff(raw, new_content)
    return new_content.encode("utf-8", errors="ignore")


async def _anonymize_multipart(proxy: DLPProxy, content: bytes, content_type: str) -> bytes:
    """Anonymize image parts in a multipart/form-data body."""
    from image_anonymizer import anonymize_image

    boundary_match = re.search(r'boundary=([^\s;]+)', content_type)
    if not boundary_match:
        return content

    boundary = boundary_match.group(1).strip('"').encode()
    delimiter = b'--' + boundary
    raw_parts = content.split(delimiter)

    result_parts = [raw_parts[0]]  # preamble (usually empty)

    for raw_part in raw_parts[1:]:
        # Closing boundary suffix "--\r\n"
        if raw_part.lstrip(b'\r\n').startswith(b'--'):
            result_parts.append(raw_part)
            continue

        # Each part starts with \r\n before the headers
        part_content = raw_part[2:] if raw_part.startswith(b'\r\n') else raw_part
        sep = part_content.find(b'\r\n\r\n')
        if sep == -1:
            result_parts.append(raw_part)
            continue

        part_headers = part_content[:sep]
        part_body_with_crlf = part_content[sep + 4:]

        if re.search(rb'(?i)content-type:\s*image/', part_headers):
            # Strip trailing \r\n (part separator before next delimiter)
            if part_body_with_crlf.endswith(b'\r\n'):
                part_body = part_body_with_crlf[:-2]
                suffix = b'\r\n'
            else:
                part_body = part_body_with_crlf
                suffix = b''
            try:
                anonymized_body, _ = anonymize_image(part_body, proxy)
                result_parts.append(b'\r\n' + part_headers + b'\r\n\r\n' + anonymized_body + suffix)
            except Exception as e:
                print(f"[anonymizer] WARNING: image anonymization failed! rethrowing error")
                # result_parts.append(raw_part)
                raise e
        else:
            result_parts.append(raw_part)

    return delimiter.join(result_parts)


async def anonymize_message(proxy: DLPProxy, headers: Headers, content: bytes | None, url: str,
                            rules: list[AnonymiseRule]) -> bytes | None:
    if not content:
        return None

    url = url.split("?")[0]
    matched_rule = next((r for r in rules if r.url_pattern.fullmatch(url)), None)
    if not matched_rule:
        return None

    t0 = time.time()
    print(f"[anonymizer] Rule matched: {matched_rule.url_pattern.pattern} for {url}")
    content_type = headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        content = await _anonymize_multipart(proxy, content, content_type)
    else:
        content = await _apply_rule_json(proxy, content, matched_rule.sensitive_fields, url)

    print(f"[anonymizer] Done in {time.time() - t0:.3f}s — {url}")

    return content


def make_deanon_chunk(proxy: DLPProxy, sse_fields: list[str] | None):
    """Return a mitmproxy stream callback that deanonymizes tokens in SSE chunks."""

    def _deanon_str(s: str) -> str:
        return REDACTED_REGEX.sub(lambda m: proxy.mappings.get_sensitive_value(m.group(0)), s)

    def deanon_chunk(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="ignore")
        out = []
        for line in text.split("\n"):
            if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                try:
                    obj = json.loads(line[6:])
                    if sse_fields:
                        # Direct path lookup — no recursive expansion, sse_fields point to scalars
                        paths_expr = "[" + ", ".join(f"path({f})" for f in sse_fields) + "]"
                        paths = jq.compile(paths_expr).input(obj).first()
                        if paths:
                            values = get_values(obj, paths)
                            updates = [(p, _deanon_str(v)) for p, v in zip(paths, values) if isinstance(v, str)]
                            obj = set_values(obj, updates)
                    else:
                        obj = json.loads(_deanon_str(json.dumps(obj, ensure_ascii=False)))
                    line = "data: " + json.dumps(obj, ensure_ascii=False)
                except Exception:
                    line = _deanon_str(line)
            out.append(line)
        return "\n".join(out).encode("utf-8", errors="ignore")

    return deanon_chunk


async def deanonymize_message(proxy: DLPProxy, content: bytes | None) -> bytes | None:
    if not content:
        return content
    text = content.decode("utf-8", errors="ignore")
    chars = list(text)
    for match in reversed(list(REDACTED_REGEX.finditer(text))):
        sensitive_value = proxy.mappings.get_sensitive_value(match.group(0))
        chars[match.start():match.end()] = list(sensitive_value)
    return "".join(chars).encode("utf-8", errors="ignore")
