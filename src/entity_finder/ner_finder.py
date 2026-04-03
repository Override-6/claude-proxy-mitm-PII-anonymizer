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

_MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

_LABEL_MAP = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORG",    "B-ORG": "ORG",    "I-ORG": "ORG",
    "LOC": "LOC",    "B-LOC": "LOC",    "I-LOC": "LOC",
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

    def find_entities(self, text: str, mappings: Mappings) -> List[Entity]:
        """
        Single-text inference. Uses a generator so the pipeline streams chunks
        through the GPU in batches rather than tokenising all at once.
        """
        chunks = self._chunk_text(text)
        # chunk_meta is populated lazily inside the generator before each yield,
        # so by the time enumerate() returns result i, chunk_meta[i] exists.
        chunk_meta: list[tuple[str, int]] = []

        def _gen():
            for chunk_text, offset in chunks:
                chunk_meta.append((chunk_text, offset))
                yield chunk_text

        seen: set[tuple[int, int]] = set()
        all_entities: List[Entity] = []
        for i, groups in enumerate(self._pipe(_gen(), batch_size=32)):
            chunk_text, offset = chunk_meta[i]
            for entity in self._to_entities(groups, chunk_text, text_offset=offset):
                key = (entity.start, entity.end)
                if key not in seen:
                    seen.add(key)
                    all_entities.append(entity)
        return all_entities

    def find_entities_batch(self, texts: List[str], mappings: Mappings = None) -> List[List[Entity]]:
        """
        Multi-text inference. A single generator streams every chunk from every
        document through the pipeline without ever materialising the full corpus
        in memory. Peak RAM is proportional to one batch (32 chunks), not the
        entire input.
        """
        if not texts:
            return []

        clean = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not clean:
            return [[] for _ in texts]

        # Metadata is appended inside the generator before each yield, so
        # chunk_meta[i] is guaranteed to exist when result i is consumed below.
        chunk_meta: list[tuple[int, str, int]] = []  # (original_idx, chunk_text, offset)

        def _gen():
            for original_idx, text in clean:
                for chunk_text, offset in self._chunk_text(text):
                    chunk_meta.append((original_idx, chunk_text, offset))
                    yield chunk_text

        output: List[List[Entity]] = [[] for _ in texts]
        seen_per_text: list[set] = [set() for _ in texts]

        for i, groups in enumerate(self._pipe(_gen(), batch_size=32)):
            original_idx, chunk_text, offset = chunk_meta[i]
            for entity in self._to_entities(groups, chunk_text, text_offset=offset):
                key = (entity.start, entity.end)
                if key not in seen_per_text[original_idx]:
                    seen_per_text[original_idx].add(key)
                    output[original_idx].append(entity)

        return output