"""
Dataset builder: Consolidate Gemma-validated samples into NER training data.

Converts disagreement samples into CoNLL-2003 BIO format for fine-tuning.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple

log = logging.getLogger(__name__)


class DatasetBuilder:
    """Build training dataset from Gemma-validated disagreements."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_file = self.output_dir / "disagreements.jsonl"
        self.training_file = self.output_dir / "training_data.jsonl"
        self.test_file = self.output_dir / "test_data.jsonl"

        self.samples = []

    def add_sample(self, sample: dict):
        """Add a disagreement sample to the dataset."""
        self.samples.append(sample)

    def _convert_to_bio_format(self, text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Convert text + entity list to BIO (Begin-Inside-Outside) format.

        Returns:
            (tokens, labels) — aligned token/label sequences
        """
        # Simple whitespace tokenization
        tokens = text.split()

        # Build character-level map of entities
        char_to_label = {}
        for entity in entities:
            ent_type = entity["type"]
            start = text.find(entity["text"])
            if start >= 0:
                end = start + len(entity["text"])
                for i in range(start, end):
                    char_to_label[i] = ent_type

        # Assign BIO labels to tokens
        labels = []
        char_idx = 0

        for token in tokens:
            # Skip whitespace
            while char_idx < len(text) and text[char_idx].isspace():
                char_idx += 1

            if char_idx >= len(text):
                break

            # Get entity type for first character of token
            ent_type = char_to_label.get(char_idx)
            token_end = char_idx + len(token)

            # Check if this is a new entity or continuation
            if ent_type:
                # Check if previous token had same entity
                if labels and labels[-1] != "O" and labels[-1].endswith(ent_type):
                    labels.append(f"I-{ent_type}")
                else:
                    labels.append(f"B-{ent_type}")
            else:
                labels.append("O")

            char_idx = token_end

        return tokens, labels

    _VALID_ENTITY_TYPES = {"PERSON", "ORG", "LOC", "EMAIL", "PHONE", "MISC"}
    _NON_NAME_PATTERNS = re.compile(
        r"@|https?://|www\.|^\d{1,3}(\.\d{1,3}){3}|^\d+$|\.\.\."
    )
    _TYPE_ALIASES = {
        "PER": "PERSON",
        "ORGANIZATION": "ORG",
        "COMPANY": "ORG",
        "LOCATION": "LOC",
        "GPE": "LOC",
        "PLACE": "LOC",
    }

    def _normalize_type(self, entity_type: str) -> str | None:
        """Return canonical type (PERSON/ORG/LOC) or None if unsupported."""
        t = entity_type.upper()
        t = self._TYPE_ALIASES.get(t, t)
        return t if t in self._VALID_ENTITY_TYPES else None


    def _build_ground_truth(self, sample: dict) -> dict:
        """
        Build ground truth labels from Gemma evaluation.

        Merges WikiNEural predictions with Gemma corrections.
        Only keeps PERSON, ORG, LOC entities that appear in the text.
        """
        text = sample["text"]
        gemma_eval = sample.get("gemma_eval", {})

        # Start with WikiNEural entities — filter to valid types and verify in text, deduplicate
        seen = set()
        entities = []
        for ent_text, ent_type in sample.get("wikineural", []):
            normalized = self._normalize_type(ent_type)
            if normalized and ent_text in text and not self._NON_NAME_PATTERNS.search(ent_text):
                key = (ent_text, normalized)
                if key not in seen:
                    seen.add(key)
                    entities.append({"text": ent_text, "type": normalized})

        # Remove false positives
        fp_list = gemma_eval.get("false_positives", [])
        fp_texts = set()
        for fp in fp_list:
            if isinstance(fp, dict):
                fp_texts.add(fp.get("text", ""))
            else:
                fp_texts.add(str(fp))
        entities = [e for e in entities if e["text"] not in fp_texts]

        # Add missed entities — validate text presence and type
        for missed in gemma_eval.get("missed_entities", []):
            missed_text = missed.get("text") if isinstance(missed, dict) else str(missed)
            raw_type = missed.get("type", "") if isinstance(missed, dict) else ""
            missed_type = self._normalize_type(raw_type)
            if not missed_type:
                log.debug(f"Skipping missed entity with invalid type {raw_type!r}: {missed_text!r}")
                continue
            # Only apply the non-name pattern filter for name-type entities (PERSON/ORG/LOC).
            # EMAIL and PHONE legitimately contain characters like @ that the pattern rejects.
            if missed_type in {"PERSON", "ORG", "LOC"} and self._NON_NAME_PATTERNS.search(missed_text):
                log.debug(f"Skipping non-name missed entity: {missed_text!r}")
                continue
            if missed_text not in text:
                log.debug(f"Skipping missed entity not in text: {missed_text!r}")
                continue
            if not any(e["text"] == missed_text for e in entities):
                entities.append({"text": missed_text, "type": missed_type})

        return {
            "text": text,
            "entities": entities,
            "source": "gemma-validated"
        }

    def finalize(self):
        """
        Process all samples and generate training/test split.
        """
        log.info(f"Finalizing dataset with {len(self.samples)} samples...")

        # Save raw disagreements
        with open(self.raw_file, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")

        log.info(f"Raw disagreements saved to {self.raw_file}")

        # Build ground truth (Gemma-corrected labels)
        training_samples = []
        for sample in self.samples:
            ground_truth = self._build_ground_truth(sample)
            training_samples.append(ground_truth)

        # Split: 80% train, 20% test
        split_idx = int(len(training_samples) * 0.8)
        train_data = training_samples[:split_idx]
        test_data = training_samples[split_idx:]

        # Save training data (CoNLL format in JSONL)
        with open(self.training_file, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        log.info(f"Training data: {len(train_data)} samples → {self.training_file}")

        # Save test data
        with open(self.test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        log.info(f"Test data: {len(test_data)} samples → {self.test_file}")

        return self.training_file, self.test_file
