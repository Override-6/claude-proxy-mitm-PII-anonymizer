"""
Supervised NER entity finder (case-insensitive).

Input text is lowercased before being passed to the model to match the
lowercased training distribution.  Character offsets are unchanged by
lower(), so span extraction against the original text is still correct.

To switch models: update _MODEL_NAME.
"""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from mappings import Mappings
from . import AbstractEntityFinder, Entity

_MODEL_NAME = "models/xlm-roberta-ner"

_LABEL_MAP = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORG",    "B-ORG": "ORG",    "I-ORG": "ORG",
    "LOC": "LOC",    "B-LOC": "LOC",    "I-LOC": "LOC",
}

_MIN_ENTITY_CHARS = 4
_MAX_ENTITY_CHARS = 50
_MIN_SCORE = 0.80


class NEREntityFinder(AbstractEntityFinder):

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        print(f"Loading NER model ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=0 if torch.cuda.is_available() else -1,
        )
        print(f"NER model loaded! ({model_name})")

    def _to_entities(self, groups: list, text: str, text_offset: int = 0) -> List[Entity]:
        seen: set[tuple[int, int]] = set()
        result: List[Entity] = []
        for g in groups:
            if g.get("score", 1.0) < _MIN_SCORE:
                continue
            label = _LABEL_MAP.get(g["entity_group"])
            if label is None:
                continue
            s = g["start"]
            e = g["end"]
            # Must sit at word boundaries in the original text.
            if s > 0 and text[s - 1].isalnum():
                continue
            if e < len(text) and text[e].isalnum():
                continue
            # Skip if immediately adjacent to _ or . (part of an identifier or path)
            if s > 0 and text[s - 1] in ("_", "."):
                continue
            if e < len(text) and text[e] in ("_", "."):
                continue
            start = s + text_offset
            end = e + text_offset
            span_text = text[s:e].strip()
            if not span_text or not (_MIN_ENTITY_CHARS <= len(span_text) <= _MAX_ENTITY_CHARS):
                continue
            if span_text.startswith(f"{label}_"):
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append(Entity(span_text, label, start, end))
        return result

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        groups = self._pipe(text.lower())
        return self._to_entities(groups, text)

    def find_entities_batch(self, texts: List[str], mappings: Mappings = None) -> List[List[Entity]]:
        if not texts:
            return []
        results = self._pipe([t.lower() for t in texts])
        if texts and results and not isinstance(results[0], list):
            results = [results]
        return [self._to_entities(groups, text) for groups, text in zip(results, texts)]
