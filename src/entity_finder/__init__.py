from abc import ABC
from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    text: str
    type: str
    # start and end position in the text they were extracted from
    start: int
    end: int


class AbstractEntityFinder(ABC):

    def find_entities(self, text: str) -> List[Entity]:
        ...
