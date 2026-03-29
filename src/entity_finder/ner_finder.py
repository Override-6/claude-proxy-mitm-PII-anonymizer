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

# Words that are never actual person names — GLiNER at lower thresholds tends to
# misclassify these as PERSON.  A PERSON entity is rejected if ALL of its tokens
# (lowercased, punctuation stripped) are in this set.
_PERSON_STOPWORDS = frozenset({
    # Pronouns
    "i", "you", "he", "she", "they", "we", "it",
    "me", "him", "her", "them", "us",
    # Possessives
    "my", "your", "his", "their", "our", "its",
    # Articles / determiners
    "the", "a", "an",
    # Generic role / profession words
    "user", "users", "client", "clients", "customer", "customers",
    "person", "people", "someone", "anyone", "everyone", "nobody", "whoever",
    "student", "students", "learner", "beginners", "beginner", "novice", "expert",
    "engineer", "engineers", "developer", "developers", "programmer", "programmers",
    "designer", "designers", "architect", "architects", "analyst", "analysts",
    "consultant", "specialist", "technician",
    "manager", "director", "lead", "owner", "operator", "supervisor",
    "senior", "junior", "mid", "principal", "staff",
    "colleague", "coworker", "teammate", "member", "contributor",
    "assistant", "agent", "bot", "model", "llm", "ai",
    "admin", "administrator", "moderator", "reviewer",
    "author", "writer", "reader", "viewer",
    # Tech-specific common words often misclassified
    "software", "hardware", "frontend", "backend", "fullstack", "devops",
    "coder", "hacker", "researcher", "scientist",
})


def _looks_like_name(text: str) -> bool:
    """Return True if *text* could be an actual person name.

    Rejects entities composed entirely of pronouns, role words, or other
    common non-name terms.  A real name must contain at least one token that
    isn't in the stopword set (e.g. "charlie", "Kirk", "Alice").
    """
    tokens = [w.lower().strip("'s,.!?:\"()[]{}") for w in text.split()]
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return False
    return any(t not in _PERSON_STOPWORDS for t in tokens)


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
            if label == "PERSON" and not _looks_like_name(span_text):
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append(Entity(span_text, label, start, end))
        return result

    def find_entities(self, text: str) -> List[Entity]:
        raw = self._model.predict_entities(text, _GLINER_LABELS, threshold=0.5)
        return self._to_entities(raw)

    def find_entities_batch(self, texts: List[Entity]) -> List[List[Entity]]:
        results = self._model.batch_predict_entities(texts, _GLINER_LABELS, threshold=0.5)
        return [self._to_entities(raw) for raw in results]
