import os
import re
from pathlib import Path
from typing import Tuple

from mitmproxy import http

from .entity_finder import Entity
from .entity_finder.mappings_finder import MappingsEntityFinder
from .mappings import Mappings, REDACTED_REGEX

# Relative paths from repo root
repo_root = Path(__file__).parent.parent
_IGNORE_DIR = repo_root / "data" / "ignore"

# Strip {{...}} spans that are intentionally left unredacted
_EXEMPT_RE = re.compile(r'\{\{.*?\}\}', re.DOTALL)

_ANXIOUS_FILTER_WHITELIST = frozenset(["CLAUDE", "CLAUDE CODE", "CLAUDE COWORK", "ANTHROPIC"])

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
    print("[proxy] Anxious filter triggered!")
    print(f"[proxy] Request {url} contained {len(entities)} unmasked sensitive entities.")
    print(f"[proxy] Entities: {set(e.text for e in entities)}")
    try:
        _IGNORE_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = _IGNORE_DIR / "last_anxious_filter.json"
        with open(dump_path, "w") as f:
            f.write(flow.request.get_content().decode("utf-8", errors="replace"))
        print(f"[proxy] Full request body saved to {dump_path}")
    except Exception as e:
        print(f"[proxy] Could not save anxious filter dump: {e}")
