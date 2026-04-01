"""
Supervised NER entity finder (case-insensitive).

Uses elastic/distilbert-base-uncased-finetuned-conll03-english as an interim
model while a multilingual uncased model is being fine-tuned.  Because the
tokenizer is uncased (do_lower_case=True), capitalization is invisible to the
model — "charlie kirk" and "Charlie Kirk" are detected identically.

To switch to the fine-tuned multilingual model later, update _MODEL_NAME and
_LABEL_MAP (if label names differ).
"""

from __future__ import annotations

from typing import List

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from mappings import Mappings
from . import AbstractEntityFinder, Entity

_MODEL_NAME = "elastic/distilbert-base-uncased-finetuned-conll03-english"

_LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
}

_MIN_ENTITY_CHARS = 3
_MAX_ENTITY_CHARS = 50

_ORG_BLACKLIST = frozenset(["CAT"])


class NEREntityFinder(AbstractEntityFinder):

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        print(f"Loading NER model ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )
        print(f"NER model loaded! ({model_name})")

    def _to_entities(self, groups: list, text: str, text_offset: int = 0) -> List[Entity]:
        seen: set[tuple[int, int]] = set()
        result: List[Entity] = []
        for g in groups:
            label = _LABEL_MAP.get(g["entity_group"])
            if label is None:
                continue
            s = g["start"]
            e = g["end"]
            # Entity must sit at word boundaries in the original text.
            if s > 0 and text[s - 1].isalnum():
                continue
            if e < len(text) and text[e].isalnum():
                continue
            start = s + text_offset
            end = e + text_offset
            span_text = text[s:e].strip()
            if not span_text or not (_MIN_ENTITY_CHARS <= len(span_text) <= _MAX_ENTITY_CHARS):
                continue
            # Skip already-redacted text
            if span_text.startswith(f"{label}_"):
                continue
            if label == "ORG" and span_text.upper() in _ORG_BLACKLIST:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append(Entity(span_text, label, start, end))
        return result

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        groups = self._pipe(text)
        return self._to_entities(groups, text)

    def find_entities_batch(self, texts: List[str], mappings: Mappings = None) -> List[List[Entity]]:
        if not texts:
            return []
        results = self._pipe(texts)
        # pipe returns a flat list for single input, list of lists for multiple
        if texts and results and not isinstance(results[0], list):
            results = [results]
        return [self._to_entities(groups, text) for groups, text in zip(results, texts)]
