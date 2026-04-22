from abc import ABC
from dataclasses import dataclass
from typing import List, Generator

from ..mappings import Mappings


@dataclass
class Entity:
    text: str
    type: str
    # start and end position in the text they were extracted from
    start: int
    end: int


class AbstractEntityFinder(ABC):

    def find_entities_batch(self, texts: List[str], mappings: Mappings) -> Generator[List[Entity], None, None]:
        ...
