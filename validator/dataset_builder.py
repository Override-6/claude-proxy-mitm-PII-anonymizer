"""
Dataset builder: Consolidate Gemma-validated samples into NER training data.

Converts disagreement samples into CoNLL-2003 BIO format for fine-tuning.
"""

import json
import logging
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

    def _build_ground_truth(self, sample: dict) -> dict:
        """
        Build ground truth labels from Gemma evaluation.

        Merges WikiNEural predictions with Gemma corrections.
        """
        text = sample["text"]
        gemma_eval = sample.get("gemma_eval", {})

        # Start with WikiNEural entities
        entities = [
            {"text": ent[0], "type": ent[1]}
            for ent in sample.get("wikineural", [])
        ]

        # Remove false positives (can be list of strings or list of dicts)
        fp_list = gemma_eval.get("false_positives", [])
        fp_texts = set()
        for fp in fp_list:
            if isinstance(fp, dict):
                fp_texts.add(fp.get("text", ""))
            else:
                fp_texts.add(str(fp))
        entities = [e for e in entities if e["text"] not in fp_texts]

        # Add missed entities (normalize to dict format)
        for missed in gemma_eval.get("missed_entities", []):
            missed_text = missed.get("text") if isinstance(missed, dict) else missed
            missed_type = missed.get("type", "PER") if isinstance(missed, dict) else "PER"
            # Avoid duplicates
            if not any(e["text"] == missed_text for e in entities):
                entities.append({
                    "text": missed_text,
                    "type": missed_type
                })

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
