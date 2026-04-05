"""
Presidio-based PII entity finder.

Uses Microsoft Presidio AnalyzerEngine for high-precision detection of
structured PII: credit cards, IBANs, SSNs, IP addresses, and more.
Replaces hand-rolled regex patterns with production-grade recognizers.
"""

from __future__ import annotations

from typing import List, Generator

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerResult

from ..mappings import Mappings
from . import AbstractEntityFinder, Entity

# Entity types to detect.  Presidio supports dozens — we pick the ones relevant
# to a privacy-focused proxy intercepting LLM traffic.
_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_SSN",
    # French NIR is not built-in — covered by a custom recognizer below.
]

# Map Presidio entity type names to our shorter internal names.
_TYPE_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_SSN": "SSN",
    "FR_NIR": "SSN",
}

# IPs that are clearly not PII (loopback, unspecified, broadcast).
_IGNORE_IPS = frozenset({
    "0.0.0.0", "127.0.0.1", "255.255.255.255",
    "localhost", "::1",
})

# Minimum confidence score to accept a Presidio result.
_MIN_SCORE = 0.4


def _build_analyzer() -> AnalyzerEngine:
    """Build a Presidio AnalyzerEngine with optional custom recognizers."""
    from presidio_analyzer import PatternRecognizer, Pattern

    analyzer = AnalyzerEngine()

    # French NIR (Numéro de Sécurité Sociale): 1 or 2, then 12 digits, then 2-digit key
    fr_nir = PatternRecognizer(
        supported_entity="FR_NIR",
        patterns=[
            Pattern(
                "FR_NIR",
                r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
                0.6,
            )
        ],
    )
    analyzer.registry.add_recognizer(fr_nir)
    return analyzer


class PresidioEntityFinder(AbstractEntityFinder):
    def __init__(self) -> None:
        print("Loading Presidio analyzer...")
        self._analyzer = _build_analyzer()
        self._batch_analyzer = BatchAnalyzerEngine(self._analyzer)
        self._entities = _ENTITIES + ["FR_NIR"]
        print("Presidio analyzer loaded!")

    def _to_entities(self, results: List[RecognizerResult], text: str) -> List[Entity]:
        out: List[Entity] = []
        for r in results:
            entity_type = _TYPE_MAP.get(r.entity_type, r.entity_type)
            if r.score < _MIN_SCORE:
                continue
            span_text = text[r.start:r.end]
            # Skip non-PII IPs
            if entity_type == "IP_ADDRESS" and span_text in _IGNORE_IPS:
                continue
            # Skip phone matches embedded inside a larger token (e.g. "haiku-4-5-20251001")
            if entity_type == "PHONE" and r.start > 0 and not text[r.start - 1].isspace():
                continue
            out.append(Entity(span_text, entity_type, r.start, r.end))
        return out

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        results = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language="en",
        )
        return self._to_entities(results, text)

    def find_entities_batch(self, texts: List[str], mappings: Mappings) -> List[List[Entity]]:
        if not texts:
            return None
        batch_results = self._batch_analyzer.analyze_iterator(
            texts,
            language="en",
            entities=self._entities,
            batch_size=32,
        )
        for idx, results in enumerate(batch_results):
            yield self._to_entities(results, texts[idx])

        return None
