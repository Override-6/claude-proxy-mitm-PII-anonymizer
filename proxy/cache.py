import asyncio
import logging
from datetime import datetime, timezone

from proxy.entity_finder import Entity

_entity_cache: dict[str, list[Entity]] = {}
_entity_cache_hits: dict[str, int] = {}
log = logging.getLogger(__name__)


def get_cached_entities_of_text(text: str) -> list[Entity] | None:
    value = _entity_cache.get(text, None)
    if value is None:
        return None

    _entity_cache_hits[text] = _entity_cache_hits.get(text, 0) + 1
    return value

def set_cached_entities(text: str, entities: list[Entity]):
    _entity_cache[text] = entities


def clear_entity_cache():
    _entity_cache.clear()
    _entity_cache_hits.clear()

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
