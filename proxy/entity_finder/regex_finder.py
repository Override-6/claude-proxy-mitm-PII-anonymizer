from re import Pattern
from typing import List, Tuple, Generator

from proxy.entity_finder import AbstractEntityFinder, Entity
from proxy.mappings import Mappings


class RegexEntityFinder(AbstractEntityFinder):
    patterns: List[Tuple[Pattern, str]]

    def __init__(self, patterns: List[Tuple[Pattern, str]]):
        self.patterns = patterns

    def find_entities_batch(self, texts: List[str], mappings: Mappings) -> Generator[List[Entity], None, None]:
        for text in texts:
            yield [
                Entity(match.group(0), label, match.start(), match.end())
                for pattern, label in self.patterns
                for match in pattern.finditer(text)
            ]
