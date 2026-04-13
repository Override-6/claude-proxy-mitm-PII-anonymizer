"""
Supervised NER entity finder.

To switch models: update _MODEL_NAME.
"""

from __future__ import annotations

import threading
import time
from typing import List, Generator

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

import logging

from proxy.mappings import Mappings
from proxy.entity_finder import AbstractEntityFinder, Entity

log = logging.getLogger(__name__)

_MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

_LABEL_MAP = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORG", "B-ORG": "ORG", "I-ORG": "ORG",
    "LOC": "LOC", "B-LOC": "LOC", "I-LOC": "LOC",
}

_MIN_ENTITY_CHARS = 4
_MAX_ENTITY_CHARS = 50
_MIN_SCORE = 0.80

# Token-based chunking — guarantees we stay under the model's 512-token limit
# regardless of script (CJK, base64, minified JSON, ...).
_CHUNK_MAX_TOKENS = 450
_CHUNK_OVERLAP_TOKENS = 50


class NEREntityFinder(AbstractEntityFinder):

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        log.info("Loading NER model (%s)...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, truncation=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._tokenizer = tokenizer
        self._pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )
        self._lock = threading.Lock()
        log.info("NER model loaded (%s)", model_name)

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into overlapping chunks bounded by *tokens*, not characters.

        Uses the tokenizer's overflow mechanism so each chunk is guaranteed to
        fit in the model's 512-token window — independent of script (CJK,
        base64, minified JSON, long unbroken strings, ...). Returns
        (chunk_text, char_offset_in_original) pairs.
        """
        if not text:
            return []

        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            add_special_tokens=False,
            truncation=True,
            max_length=_CHUNK_MAX_TOKENS,
            stride=_CHUNK_OVERLAP_TOKENS,
        )

        chunks: list[tuple[str, int]] = []
        for offsets in encoding["offset_mapping"]:
            real = [(s, e) for s, e in offsets if not (s == 0 and e == 0)]
            if not real:
                continue
            start_char = real[0][0]
            end_char = real[-1][1]
            chunks.append((text[start_char:end_char], start_char))
        return chunks

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

    def find_entities_batch(self, texts: List[str], mappings: Mappings = None) -> Generator[List[Entity], None, None]:
        """
        Multi-text inference. The pipeline streams chunks from every document
        without materialising the full corpus in memory. Peak RAM is
        proportional to one batch (32 chunks), not the entire input.
        Yields one List[Entity] per input text, in the original order.
        """
        if not texts:
            return

        char_count = sum([len(text) for text in texts])
        t0 = time.time()

        # chunk_meta[i] = (text_idx, chunk_text, offset_in_text)
        chunk_meta: list[tuple[int, str, int]] = []

        for text_idx, text in enumerate(texts):
            for chunk_text, offset in self._chunk_text(text):
                chunk_meta.append((text_idx, chunk_text, offset))
        # Accumulate entities per text across all their chunks.
        # seen_spans deduplicates entities from overlapping chunk regions.
        accumulated: list[list[Entity]] = [[] for _ in texts]
        seen_spans: list[set[tuple[int, int]]] = [set() for _ in texts]

        if not chunk_meta:
            return None

        try:
            def _gen():
                for _, chunk_text, _ in chunk_meta:
                    yield chunk_text

            with self._lock:
                for chunk_i, groups in enumerate(self._pipe(_gen(), batch_size=32)):
                    text_idx, chunk_text, offset = chunk_meta[chunk_i]
                    for entity in self._to_entities(groups, chunk_text, text_offset=offset):
                        key = (entity.start, entity.end)
                        if key not in seen_spans[text_idx]:
                            seen_spans[text_idx].add(key)
                            accumulated[text_idx].append(entity)
        finally:
            t1 = time.time()
            log.debug("NER %.3fs over %d chars", t1 - t0, char_count)

        for entities in accumulated:
            yield entities

        return None
