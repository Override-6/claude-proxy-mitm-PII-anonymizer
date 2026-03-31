from typing import List

from entity_finder import AbstractEntityFinder, Entity
from mappings import Mappings

whitelisted_entities = frozenset(["CLAUDE", "ANTHROPIC"])

class MappingsEntityFinder(AbstractEntityFinder):
    """
    Uses flashtext to get any occurrence in text of known entities that could have been missed by preceding entity finders.
    This finder is the latest rempart to sensitive data leak.
    """
    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        matches = mappings.kp.extract_keywords(text, span_info=True)
        return [Entity(match[0], mappings.get_redacted_text_type(match[0]), match[1], match[2]) for match in matches if match[0] not in whitelisted_entities]