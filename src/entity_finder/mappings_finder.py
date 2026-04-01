from typing import List

from entity_finder import AbstractEntityFinder, Entity
from mappings import Mappings


class MappingsEntityFinder(AbstractEntityFinder):
    """
    Finds every occurrence of any known sensitive value as a substring using
    Aho-Corasick (O(n + m + z)). Case-insensitive, no word-boundary restrictions —
    matches inside compound identifiers like mcp__google-workspace__auth.
    This finder is the last line of defence against sensitive data leaking through.
    """

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        automaton = mappings.build_automaton()
        if not len(automaton):
            return []

        text_lower = text.lower()
        entities: List[Entity] = []
        for end_idx, length in automaton.iter(text_lower):
            start = end_idx - length + 1
            matched = text[start:end_idx + 1]  # original casing from source text
            entity_type = mappings.get_redacted_text_type(matched)
            entities.append(Entity(matched, entity_type, start, end_idx + 1))
        return entities
