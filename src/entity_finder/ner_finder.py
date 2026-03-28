"""
GLiNER-based multilingual entity finder.

Uses urchade/gliner_multi-v2.1 — a zero-shot multilingual NER model that
handles French, English, and other languages without separate per-language
models.  Labels are mapped to the same type names used by the rest of the
pipeline (PERSON, ORG, LOC, GPE, EMAIL, PHONE).
"""

from __future__ import annotations

from typing import List

from gliner import GLiNER

from . import AbstractEntityFinder, Entity

_MODEL_NAME = "urchade/gliner_multi-v2.1"

# Labels to request from GLiNER — use natural-language descriptions that the
# model understands, mapped to our internal type names.
_GLINER_LABELS = ["person", "organization", "location"]
_LABEL_MAP = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "LOC",
}

_MIN_ENTITY_CHARS = 3
_MAX_ENTITY_CHARS = 50


class NEREntityFinder(AbstractEntityFinder):

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        print(f"Loading GLiNER model ({model_name})...")
        self._model = GLiNER.from_pretrained(model_name)
        print(f"GLiNER model loaded! ({model_name})")

    def _to_entities(self, raw: list, text_offset: int = 0) -> List[Entity]:
        seen: set[tuple[int, int]] = set()
        result: List[Entity] = []
        for r in raw:
            label = _LABEL_MAP.get(r["label"], r["label"].upper())
            start = r["start"] + text_offset
            end = r["end"] + text_offset
            span_text = r["text"]
            if not (_MIN_ENTITY_CHARS <= len(span_text) <= _MAX_ENTITY_CHARS):
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append(Entity(span_text, label, start, end))
        return result

    def find_entities(self, text: str) -> List[Entity]:
        raw = self._model.predict_entities(text, _GLINER_LABELS)
        return self._to_entities(raw)

    def find_entities_batch(self, texts: List[Entity]) -> List[List[Entity]]:
        results = self._model.batch_predict_entities(texts, _GLINER_LABELS)
        return [self._to_entities(raw) for raw in results]
