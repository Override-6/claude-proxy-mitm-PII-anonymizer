"""
GLiNER-based multilingual entity finder.

Uses urchade/gliner_multi-v2.1 — a zero-shot multilingual NER model that
handles French, English, and other languages without separate per-language
models.  Labels are mapped to the same type names used by the rest of the
pipeline (PERSON, ORG, LOC).

To prevent false positives like "electrical engineer" being flagged as PERSON,
we feed the model *negative labels* ("job title", "profession") alongside the
real labels.  GLiNER then assigns ambiguous spans to the negative label instead
of PERSON — the model does the disambiguation, not a hardcoded word list.
"""

from __future__ import annotations

from typing import List

from gliner import GLiNER

from mappings import Mappings
from . import AbstractEntityFinder, Entity

_MODEL_NAME = "urchade/gliner_multi-v2.1"

# Labels to request from GLiNER.  "job title" and "profession" act as negative
# labels — entities assigned to them are discarded, preventing misclassification
# of roles/professions as PERSON.
_GLINER_LABELS = [
    "person name", "organization", "location",
    "job title", "profession",
]

# Only these labels produce Entity results; the rest are negative / disambiguation labels.
_LABEL_MAP = {
    "person name": "PERSON",
    "organization": "ORG",
    "location": "LOC",
}

_MIN_ENTITY_CHARS = 3
_MAX_ENTITY_CHARS = 50

_ENTITY_WHITELIST = frozenset(["ANTHROPIC", "CLAUDE"])


class NEREntityFinder(AbstractEntityFinder):

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        print(f"Loading GLiNER model ({model_name})...")
        self._model = GLiNER.from_pretrained(model_name)
        print(f"GLiNER model loaded! ({model_name})")

    def _to_entities(self, raw: list, text_offset: int = 0) -> List[Entity]:
        seen: set[tuple[int, int]] = set()
        result: List[Entity] = []
        for r in raw:
            label = _LABEL_MAP.get(r["label"])
            if label is None:
                # Negative label (job title, profession, etc.) — discard
                continue
            start = r["start"] + text_offset
            end = r["end"] + text_offset
            span_text: str = r["text"]
            if not (_MIN_ENTITY_CHARS <= len(span_text) <= _MAX_ENTITY_CHARS):
                continue
            # the NER will sometimes match already redacted text. Should be ignored.
            if span_text.startswith(f"{label}_") or span_text in _ENTITY_WHITELIST:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append(Entity(span_text, label, start, end))
        return result

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        raw = self._model.predict_entities(text, _GLINER_LABELS, threshold=0.5)
        return self._to_entities(raw)

    def find_entities_batch(self, texts: List[str], mappings: Mappings = None) -> List[List[Entity]]:
        results = self._model.batch_predict_entities(texts, _GLINER_LABELS, threshold=0.5)
        return [self._to_entities(raw) for raw in results]
