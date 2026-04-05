from re import Pattern
from typing import List, Tuple

from proxy.mappings import Mappings
from proxy.entity_finder import AbstractEntityFinder, Entity


class RegexEntityFinder(AbstractEntityFinder):
    patterns: List[Tuple[Pattern, str]]

    def __init__(self, patterns: List[Tuple[Pattern, str]]):
        self.patterns = patterns

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        return [
            Entity(match.group(0), label, match.start(), match.end())
            for pattern, label in self.patterns
            for match in pattern.finditer(text)
        ]