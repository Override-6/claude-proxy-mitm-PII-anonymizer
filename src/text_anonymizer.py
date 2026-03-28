"""
PII detection using GLiNER (multilingual) + regex for realtime proxy use.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import List

from entity_finder import Entity
from entity_finder.ner_finder import NEREntityFinder
from entity_finder.regex_finder import RegexEntityFinder
from event_socket import get_event_socket
from mappings import Mappings

log = logging.getLogger(__name__)

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

finders = [_regex_finder, _ner_finder]

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


@lru_cache(maxsize=2048)
def _detect_entities_cached(text: str) -> tuple[Entity, ...]:
    """Cached wrapper around entity detection.

    Finders are applied in priority order (regex first, NER second).
    Entities from later finders are dropped when they overlap with an entity
    already accepted by an earlier finder — prevents NER from detecting the
    local-part of an email address as a PERSON, for example.

    Claude Desktop resends the full conversation history with every turn, so
    most text blocks are identical across requests. The LRU cache makes those
    repeat calls instant without re-running NER.
    """
    accepted: list[Entity] = []
    for finder in finders:
        for entity in finder.find_entities(text):
            if not any(entity.start < a.end and entity.end > a.start for a in accepted):
                accepted.append(entity)
    return tuple(accepted)


def detect_entities(text: str) -> List[Entity]:
    return list(_detect_entities_cached(text))


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

    # Run entity detection in a thread so spaCy doesn't block the event loop
    entities = await asyncio.to_thread(_detect_entities_cached, clean_text)

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
