import base64
import json
import logging
import re
import time
from dataclasses import dataclass

import jq
from mitmproxy.http import Headers

import claude_system_prompt
import cache
from entity_finder.mappings_finder import MappingsEntityFinder
from mappings import Mappings
from entity_finder import AbstractEntityFinder, Entity
from rules import ProxyRules, AnonymiseRule, DeanonymiseRule

log = logging.getLogger(__name__)

REDACTED_REGEX = re.compile(r'\[([A-Z_]+)_[0-9]+]')

_mappings_finder = MappingsEntityFinder()


@dataclass
class ProxyOptions:
    anxious_filter: bool = True
    save_redacted_images: bool = True
    inject_system_prompt: bool = True
    save_requests: bool = True


@dataclass
class DLPProxy:
    mappings: Mappings
    rules: ProxyRules
    finders: list[AbstractEntityFinder]
    options: ProxyOptions


def expand_paths(obj: dict, expressions: list[str]) -> list[list[str | int]]:
    combined = "[" + ", ".join(f"path({e} | .. | strings)" for e in expressions) + "]"
    try:
        return jq.compile(combined).input(obj).first()
    except (ValueError, StopIteration):
        return []


def get_values(obj: dict, paths: list[list]) -> list[str]:
    try:
        return jq.compile('[.[] | . as $p | $obj | getpath($p)]', args={'obj': obj}).input(paths).first()
    except (ValueError, StopIteration):
        return []


def set_values(obj: dict, values: list[tuple[list, str]]) -> dict:
    for path, value in values:
        try:
            obj = jq.compile('setpath($p; $v)', args={'p': path, 'v': value}).input(obj).first()
        except (ValueError, StopIteration):
            pass
    return obj


def overlaps(entity: Entity, accepted: list[Entity]) -> bool:
    """Return True if *entity* overlaps any already-accepted entity."""
    return any(entity.start < a.end and entity.end > a.start for a in accepted)


def redact_entities(text: str, entities: list[Entity], mappings: Mappings) -> str:
    chars = list(text)
    for entity in sorted(entities, key=lambda ent: ent.start, reverse=True):
        s, e = entity.start, entity.end
        replacement = mappings.get_or_set_redacted_text(text[s:e], entity.type)
        chars[s:e] = list(replacement)
    return ''.join(chars)


async def _apply_paths(proxy: DLPProxy, data: dict, sensitive_fields: list[str] | bool, url: str = None) -> dict:
    """Anonymize every string value reached by each jq path expression."""
    from entity_cache_log import log_extracted_entities

    expressions = ["."] if sensitive_fields is True else sensitive_fields
    paths = expand_paths(data, expressions)
    if not paths:
        return data
    values = get_values(data, paths)

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
            for list_idx, entities in zip(non_cached_indices, finder.find_entities_batch(non_cached_values, proxy.mappings)):
                entities_list[list_idx].extend(
                    e for e in entities if not overlaps(e, entities_list[list_idx])
                )

    # Run MappingsEntityFinder on all texts to catch re-occurrences of known sensitive values
    for i, entities in enumerate(_mappings_finder.find_entities_batch(values, proxy.mappings)):
        entities_list[i].extend(e for e in entities if not overlaps(e, entities_list[i]))

    updates: list[tuple[list, str]] = []
    for path_list, value, entities in zip(paths, values, entities_list):
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
            else:
                # Recurse into all values
                for value in obj.values():
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return data


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
        print(f"[anonymizer] '{tok}' context: ...{new_content[max(0, idx - 80):idx + len(tok) + 80]}...")


async def _apply_rule_json(proxy: DLPProxy, raw: bytes, sensitive_fields: list[str] | bool, url: str) -> bytes:
    body = json.loads(raw)
    original = json.dumps(body, ensure_ascii=False)

    # Anonymize base64 images in content blocks
    body = await _anonymize_base64_images(proxy, body)

    # Anonymize text fields (with URL for entity logging)
    body = await _apply_paths(proxy, body, sensitive_fields, url=url)

    if proxy.options.inject_system_prompt:
        body = claude_system_prompt.inject(body, url)

    new_content = json.dumps(body, ensure_ascii=False)
    _log_anonymization_diff(original, new_content)
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
                print(f"[anonymizer] WARNING: image anonymization failed: {e}")
                result_parts.append(raw_part)
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
