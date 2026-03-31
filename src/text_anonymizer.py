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
from entity_finder.mappings_finder import MappingsEntityFinder
from entity_finder.ner_finder import NEREntityFinder
from entity_finder.presidio_finder import PresidioEntityFinder
from mappings import Mappings, REDACTED_REGEX

log = logging.getLogger(__name__)

# Limit concurrent GLiNER inference to 1.
# On Raspberry Pi 4 (ARMv8.0, 3.7 GB RAM) running GLiNER simultaneously in
# multiple asyncio.to_thread calls causes OOM kills and SIGILL crashes.
# The semaphore is only acquired on a cache miss — cache hits skip it entirely
# so large conversation histories (many repeated blocks) don't queue needlessly.
_gliner_sem = asyncio.Semaphore(1)
_entity_cache: dict[str, tuple[Entity, ...]] = {}
_entity_cache_hits: dict[str, int] = {}

# Priority order: finders earlier in the list win on overlapping spans.
# Presidio handles structured PII (email, phone, credit card, IBAN, SSN, IP).
# GLiNER handles semantic NER (person names, organizations, locations).
# MappingsEntityFinder catches any previously-seen entities missed above.
_entity_finders = [
    PresidioEntityFinder(),
    NEREntityFinder(),
    MappingsEntityFinder()
]

_EXEMPT_RE = re.compile(r'\{\{(.*?)\}\}', re.DOTALL)


def _no_overlap(entity: Entity, accepted: list[Entity]) -> bool:
    """Return True if *entity* does not overlap any already-accepted entity."""
    return not any(entity.start < a.end and entity.end > a.start for a in accepted)


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


def _run_entity_detection(text: str, mappings: Mappings) -> tuple[Entity, ...]:
    """Run all finders on *text* and return deduplicated entities.

    Finders run in priority order (_entity_finders); later-finder entities
    overlapping an already-accepted span are discarded (prevents NER from
    grabbing email local-parts as PERSON, etc.).
    """
    if not text.strip():
        return ()

    accepted: list[Entity] = []
    for finder in _entity_finders:
        for entity in finder.find_entities(text, mappings):
            if _no_overlap(entity, accepted):
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


def detect_entities(text: str, mappings: Mappings) -> List[Entity]:
    if text not in _entity_cache:
        _entity_cache[text] = _run_entity_detection(text, mappings)
    return list(_entity_cache[text])


async def prewarm_cache(texts: List[str], mappings: Mappings) -> None:
    """Pre-populate entity cache for a batch of texts using finder-first ordering.

    Runs finder[0] on ALL texts, then finder[1] on ALL texts, etc.  This is the
    batch counterpart of _run_entity_detection and ensures GLiNER (and other
    heavy finders) process all texts in a single pass rather than one at a time.

    Already-cached texts are skipped so repeated calls are cheap.
    """
    clean_texts: List[str] = []
    for text in texts:
        if not text.strip():
            continue
        clean, _ = _strip_exempt_markers(text)
        if clean.strip() and clean not in _entity_cache:
            clean_texts.append(clean)

    if not clean_texts:
        return

    per_text_accepted: list[list[Entity]] = [[] for _ in clean_texts]

    def _run_sequential():
        # Run each finder in order and commit new entities to mappings.kp between
        # finders so that MappingsEntityFinder can find entities detected earlier.
        for finder in _entity_finders:
            batch_results = finder.find_entities_batch(clean_texts, mappings)
            for i, entities in enumerate(batch_results):
                for entity in entities:
                    accepted = per_text_accepted[i]
                    if _no_overlap(entity, accepted):
                        accepted.append(entity)
                        mappings.get_or_set_redacted_text(entity.text, entity.type)

    async with _gliner_sem:
        await asyncio.to_thread(_run_sequential)

    for i, clean in enumerate(clean_texts):
        _entity_cache[clean] = tuple(per_text_accepted[i])
        _entity_cache_hits[clean] = 0


def detect_entities_batch(texts: List[str], mappings: Mappings) -> List[List[Entity]]:
    """Detect entities across multiple texts, respecting _entity_finders priority order.

    Used by image_anonymizer to process all OCR lines in one pass.
    """
    per_finder = [finder.find_entities_batch(texts, mappings) for finder in _entity_finders]
    merged = []
    for i in range(len(texts)):
        accepted: list[Entity] = []
        for finder_results in per_finder:
            for entity in finder_results[i]:
                if _no_overlap(entity, accepted):
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
            entities = await asyncio.to_thread(_run_entity_detection, clean_text, mappings)
            _entity_cache[clean_text] = entities
            _entity_cache_hits[clean_text] = 0

    # Drop entities that overlap (even partially) with an {{...}} marker
    active = [
        e for e in entities
        if not any(e.start < ee and e.end > es for es, ee in exempt_spans)
    ]

    redacted_entities = [(ent, mappings.get_or_set_redacted_text(ent.text, ent.type)) for ent in active]

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
