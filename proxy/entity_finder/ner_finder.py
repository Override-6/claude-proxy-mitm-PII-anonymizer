"""
Supervised NER entity finder (case-insensitive).

Input text is lowercased before being passed to the model to match the
lowercased training distribution.  Character offsets are unchanged by
lower(), so span extraction against the original text is still correct.

To switch models: update _MODEL_NAME.
"""

from __future__ import annotations

import time
from typing import List, Generator

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ..mappings import Mappings
from . import AbstractEntityFinder, Entity

_MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

_LABEL_MAP = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORG", "B-ORG": "ORG", "I-ORG": "ORG",
    "LOC": "LOC", "B-LOC": "LOC", "I-LOC": "LOC",
}

_MIN_ENTITY_CHARS = 4
_MAX_ENTITY_CHARS = 50
_MIN_SCORE = 0.80

# ~1000 chars stays well under 512 tokens for most text
_CHUNK_MAX_CHARS = 1000
_CHUNK_OVERLAP_CHARS = 100


class NEREntityFinder(AbstractEntityFinder):

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        print(f"Loading NER model ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, truncation=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )
        print(f"NER model loaded! ({model_name})")

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into overlapping chunks of ~_CHUNK_MAX_CHARS characters.
        Cuts at word boundaries. Returns list of (chunk, offset_in_original).
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + _CHUNK_MAX_CHARS, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunks.append((text[start:end], start))
            if end >= len(text):
                break
            start = end - _CHUNK_OVERLAP_CHARS
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

        # chunk_meta[i] is set by _gen() before the pipeline consumes chunk i,
        # so it is always populated when the pipeline result for chunk i arrives.
        chunk_meta: list[tuple[str, int]] = []  # (original_idx, chunk_text, offset)

        def _gen():
            for text in texts:
                for chunk_text, offset in self._chunk_text(text):
                    chunk_meta.append((chunk_text, offset))
                    yield chunk_text

        try:
            for chunk_i, groups in enumerate(self._pipe(_gen(), batch_size=32)):
                chunk_text, offset = chunk_meta[chunk_i]
                yield self._to_entities(groups, chunk_text, text_offset=offset)
        finally:
            t1 = time.time()

            print(f"[ner] operation took {t1 - t0} seconds for ner over {char_count} chars ({chunk_i + 1} chunk)")
