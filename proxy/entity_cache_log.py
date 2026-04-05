"""
Log extracted entities per request for validator reuse.

When the proxy extracts entities during anonymization, save them so the validator
doesn't have to re-run expensive entity extraction.
"""

import json
import logging
from pathlib import Path
from typing import List

from proxy.entity_finder import Entity

log = logging.getLogger(__name__)

# Use relative paths from repo root
repo_root = Path(__file__).parent.parent
_ENTITIES_LOG = repo_root / "data" / "requests-entities.jsonl"


def init_entities_log():
    """Initialize the entities log file."""
    _ENTITIES_LOG.parent.mkdir(parents=True, exist_ok=True)


def log_extracted_entities(request_url: str, text_field_path: str, field_text: str, entities: List[Entity]):
    """
    Log extracted entities for a text field.

    Args:
        request_url: The request URL (for deduplication)
        text_field_path: jq path to the field (e.g., ".messages[0].content")
        field_text: The original text that was analyzed
        entities: List of Entity objects extracted
    """
    try:
        entry = {
            "request_url": request_url,
            "field_path": text_field_path,
            "text": field_text,
            "entities": [
                {
                    "text": e.text,
                    "type": e.type,
                    "start": e.start,
                    "end": e.end
                }
                for e in entities
            ]
        }

        with open(_ENTITIES_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception as e:
        log.warning(f"Failed to log extracted entities: {e}")


def get_cached_entities(request_url: str, text_field_path: str, field_text: str) -> List[dict] | None:
    """
    Look up previously extracted entities for this field.

    Returns:
        List of entity dicts if found, None if not cached
    """
    try:
        if not _ENTITIES_LOG.exists():
            return None

        # Simple linear search (could be optimized with indexing for large logs)
        with open(_ENTITIES_LOG, "r") as f:
            for line in f:
                entry = json.loads(line)
                if (entry["request_url"] == request_url and
                    entry["field_path"] == text_field_path and
                    entry["text"] == field_text):
                    return entry["entities"]

        return None

    except Exception as e:
        log.warning(f"Failed to look up cached entities: {e}")
        return None
