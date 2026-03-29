"""
PII detection using GLiNER (multilingual) + regex for realtime proxy use.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import List

from entity_finder import Entity
from entity_finder.ner_finder import NEREntityFinder
from entity_finder.regex_finder import RegexEntityFinder
from event_socket import get_event_socket
from mappings import Mappings

log = logging.getLogger(__name__)

# Limit concurrent GLiNER inference to 1.
# On Raspberry Pi 4 (ARMv8.0, 3.7 GB RAM) running GLiNER simultaneously in
# multiple asyncio.to_thread calls causes OOM kills and SIGILL crashes.
# The semaphore is only acquired on a cache miss — cache hits skip it entirely
# so large conversation histories (many repeated blocks) don't queue needlessly.
_gliner_sem = asyncio.Semaphore(1)
_entity_cache: dict[str, tuple[Entity, ...]] = {}
_entity_cache_hits: dict[str, int] = {}

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
PHONE_REGEX = re.compile(
    r'(?<!\d)'
    r'(?:'
    # International +CC: compact (+33761647274) or with separators (+33 7 61 64 72 74)
    r'\+\d{1,3}(?:\d{6,12}|[\s\-.]?\(?\d{1,4}\)?(?:[\s\-.]?\d{2,4}){1,3})'
    r'|'
    # Local 0-prefix: compact (0761647274) or with separators (06 12 34 56 78)
    r'0\d{1,4}(?:\d{6,10}|(?:[\s\-.]?\d{2,4}){2,4})'
    r'|'
    # Parenthesized area code: (NXX) NXX-XXXX
    r'\(\d{2,4}\)[\s\-.]?\d{3,4}[\s\-.]?\d{4}'
    r'|'
    # Consistent-separator groups without prefix: NXX-NXX-XXXX or NXX NXX XXXX
    # Backreference ensures the same separator is used throughout
    r'\d{2,4}([\s\-])\d{3,4}\1\d{3,4}'
    r')'
    r'(?!\d)'
)

_regex_finder = RegexEntityFinder([
    (EMAIL_REGEX, "EMAIL"),
    (PHONE_REGEX, "PHONE"),
])
_ner_finder = NEREntityFinder()

REDACTED_REGEX = re.compile(r'\[[A-Z_]+_[0-9]+]')
_EXEMPT_RE = re.compile(r'\{\{(.*?)\}\}', re.DOTALL)


def _strip_exempt_markers(text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Remove {{...}} markers from *text*, returning (clean_text, exempt_spans).

    exempt_spans are (start, end) positions in clean_text that must not be
    anonymized — they correspond to content the user explicitly marked.
    """
    parts: list[str] = []
    exempt_spans: list[tuple[int, int]] = []
    pos = 0
    clean_pos = 0

    for m in _EXEMPT_RE.finditer(text):
        before = text[pos:m.start()]
        parts.append(before)
        clean_pos += len(before)

        content = m.group(1)
        exempt_spans.append((clean_pos, clean_pos + len(content)))
        parts.append(content)
        clean_pos += len(content)

        pos = m.end()

    parts.append(text[pos:])
    return "".join(parts), exempt_spans


def _run_entity_detection(text: str) -> tuple[Entity, ...]:
    """Run all finders on *text* and return deduplicated entities.

    Finders run in priority order; later-finder entities overlapping an
    already-accepted span are discarded (prevents NER from grabbing email
    local-parts as PERSON, etc.).
    """
    regex_entities = _regex_finder.find_entities(text)

    if not text.strip():
        return tuple(regex_entities)

    accepted: list[Entity] = list(regex_entities)
    for entity in _ner_finder.find_entities(text):
        if not any(entity.start < a.end and entity.end > a.start for a in accepted):
            accepted.append(entity)
    return tuple(accepted)


def _prune_entity_cache():
    """Remove cache entries hit fewer than 3 times since the last prune."""
    cold = [key for key, hits in _entity_cache_hits.items() if hits < 3]
    for key in cold:
        _entity_cache.pop(key, None)
        _entity_cache_hits.pop(key, None)
    if cold:
        log.info("[cache] Pruned %d cold entries (< 3 hits). %d remain.", len(cold), len(_entity_cache))
    else:
        log.info("[cache] Prune run: all %d entries are warm.", len(_entity_cache))
    # Reset counters for the next period
    for key in _entity_cache_hits:
        _entity_cache_hits[key] = 0


async def _cache_prune_loop():
    """Run _prune_entity_cache every 12 h, aligned to 00:00 or 12:00 UTC."""
    while True:
        now = datetime.now(timezone.utc)
        # Next boundary: either midnight or noon, whichever comes first
        next_hour = 12 if now.hour < 12 else 24  # 24 → wraps to 00:00 next day
        seconds_until = (
            (next_hour - now.hour) * 3600
            - now.minute * 60
            - now.second
        )
        await asyncio.sleep(seconds_until)
        _prune_entity_cache()


def start_cache_prune_task():
    """Schedule the cache pruning loop. Call once after the event loop starts."""
    asyncio.get_event_loop().create_task(_cache_prune_loop())


def detect_entities(text: str) -> List[Entity]:
    if text not in _entity_cache:
        _entity_cache[text] = _run_entity_detection(text)
    return list(_entity_cache[text])



def detect_entities_batch(texts: List[str]) -> List[List[Entity]]:
    """Detect entities across multiple texts using batched NER + regex.

    Used by image_anonymizer to process all OCR lines in one pass.
    """
    # Regex first (higher priority)
    regex_results: list[list[Entity]] = [
        _regex_finder.find_entities(t) for t in texts
    ]

    # NER: batch — single model pass for all texts
    ner_results: list[list[Entity]] = _ner_finder.find_entities_batch(texts)

    # Merge: NER entities overlapping a regex match are discarded
    merged = []
    for regex_ents, ner_ents in zip(regex_results, ner_results):
        accepted = list(regex_ents)
        for entity in ner_ents:
            if not any(entity.start < a.end and entity.end > a.start for a in accepted):
                accepted.append(entity)
        merged.append(accepted)
    return merged


async def anonymize_text(text: str, mappings: Mappings) -> str:
    clean_text, exempt_spans = _strip_exempt_markers(text)

    # Cache hit: return instantly, no semaphore needed.
    # Cache miss: serialize GLiNER inference — prevents concurrent PyTorch calls
    # that cause OOM / SIGILL on Raspberry Pi 4 (ARMv8.0, 3.7 GB RAM).
    if clean_text in _entity_cache:
        _entity_cache_hits[clean_text] = _entity_cache_hits.get(clean_text, 0) + 1
        entities = _entity_cache[clean_text]
    else:
        async with _gliner_sem:
            entities = await asyncio.to_thread(_run_entity_detection, clean_text)
            _entity_cache[clean_text] = entities
            _entity_cache_hits[clean_text] = 0

    # Drop entities that overlap (even partially) with an {{...}} marker
    active = [
        e for e in entities
        if not any(e.start < ee and e.end > es for es, ee in exempt_spans)
    ]

    redacted_entities = [(ent, mappings.get_redacted_text(ent.text, ent.type)) for ent in active]

    socket = get_event_socket()
    socket.broadcast({
        "type": "anonymize_text",
        "entities": [{
            "entity": ent.text,
            "redacted": redacted_text,
        } for ent, redacted_text in redacted_entities],
    })

    if not active and not exempt_spans:
        return clean_text

    chars = list(clean_text)

    # Process all operations right-to-left so earlier positions stay valid:
    #  - redactions: replace entity span with redacted token
    #  - exempt rewrap: re-insert {{...}} around exempt content
    ops = (
        [(ent.start, ent.end, redacted) for ent, redacted in redacted_entities] +
        [(s, e, None) for s, e in exempt_spans]
    )
    for s, e, replacement in sorted(ops, key=lambda x: x[0], reverse=True):
        if replacement is not None:
            chars[s:e] = list(replacement)
        else:
            chars[s:e] = list("{{") + chars[s:e] + list("}}")

    return "".join(chars)


def deanonymize_text(text: str, mappings: Mappings) -> str:
    chars = list(text)
    for match in reversed(list(REDACTED_REGEX.finditer(text))):
        sensitive_value = mappings.get_sensitive_value(match.group(0))
        chars[match.start():match.end()] = list(sensitive_value)

    return "".join(chars)
