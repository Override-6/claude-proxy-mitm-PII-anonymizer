from typing import List, Generator

from ..mappings import Mappings
from . import AbstractEntityFinder, Entity


class MappingsEntityFinder(AbstractEntityFinder):
    """
    Finds every occurrence of any known sensitive value as a substring using
    Aho-Corasick (O(n + m + z)). Case-insensitive, no word-boundary restrictions —
    matches inside compound identifiers like mcp__google-workspace__auth.
    This finder is the last line of defence against sensitive data leaking through.
    """

    def find_entities_batch(self, texts: List[str], mappings: Mappings) -> Generator[list[Entity], None, None]:
        automaton = mappings.build_automaton()
        if not len(automaton):
            return None

        for text in texts:
            text_lower = text
            entities: List[Entity] = []
            for end_idx, length in automaton.iter(text_lower):
                start = end_idx - length + 1
                matched = text[start:end_idx + 1]  # original casing from source text
                entity_type = mappings.get_redacted_text_type(matched)
                entities.append(Entity(matched, entity_type, start, end_idx + 1))
            yield entities

        return None
