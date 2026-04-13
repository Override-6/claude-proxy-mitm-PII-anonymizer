import logging
import os
import re
from pathlib import Path
from typing import Tuple

from mitmproxy import http

from proxy.entity_finder import Entity
from proxy.entity_finder.mappings_finder import MappingsEntityFinder
from proxy.mappings import Mappings, REDACTED_REGEX

# Relative paths from repo root
repo_root = Path(__file__).parent.parent
_IGNORE_DIR = repo_root / "data" / "ignore"

# Strip {{...}} spans that are intentionally left unredacted
_EXEMPT_RE = re.compile(r'\{\{.*?\}\}', re.DOTALL)

_ANXIOUS_FILTER_WHITELIST = frozenset(["CLAUDE", "CLAUDE CODE", "CLAUDE COWORK", "ANTHROPIC"])

log = logging.getLogger(__name__)

_mappings_finder = MappingsEntityFinder()


def anxious_filter(mappings: Mappings, request_body: str) -> Tuple[bool, list[Entity]]:
    filtered_body = _EXEMPT_RE.sub('', request_body)
    all_entities = next(_mappings_finder.find_entities_batch([filtered_body], mappings), [])
    entities = [
        e for e in all_entities
        if e.text.upper() not in _ANXIOUS_FILTER_WHITELIST and not REDACTED_REGEX.fullmatch(e.text)
    ]
    return len(entities) > 0, entities


def trigger_anxious_filter(url: str, flow: http.HTTPFlow, entities: list[Entity]):
    entity_texts = {e.text for e in entities}
    log.warning("ANXIOUS FILTER: %s — %d unmasked entities: %s", url, len(entities), entity_texts)
    try:
        _IGNORE_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = _IGNORE_DIR / "last_anxious_filter.json"
        with open(dump_path, "w") as f:
            f.write(flow.request.get_content().decode("utf-8", errors="replace"))
        log.debug("Anxious filter dump saved to %s", dump_path)
    except Exception as e:
        log.error("Could not save anxious filter dump: %s", e)
