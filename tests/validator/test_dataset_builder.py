"""Tests for dataset builder and data quality."""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, '/app')
from validator.dataset_builder import DatasetBuilder


class TestDatasetBuilder:
    """Test dataset construction and validation."""

    def test_add_sample(self):
        """Test adding samples to the dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            sample = {
                "text": "John Smith works at Acme Corp",
                "wikineural": [("John Smith", "PERSON"), ("Acme Corp", "ORG")],
                "gemma_eval": {
                    "correct": True,
                    "false_positives": [],
                    "missed_entities": []
                },
                "disagreement_score": 0.0
            }

            builder.add_sample(sample)
            assert len(builder.samples) == 1

    def test_convert_to_bio_format(self):
        """Test BIO format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            text = "John Smith works at Acme Corp in New York"
            entities = [
                {"text": "John Smith", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORG"},
                {"text": "New York", "type": "LOC"},
            ]

            tokens, labels = builder._convert_to_bio_format(text, entities)

            assert len(tokens) == len(labels)
            assert tokens[0] == "John"
            assert labels[0] in ("B-PERSON", "O")  # Should mark John as PERSON

    def test_build_ground_truth_merges_annotations(self):
        """Test that ground truth properly merges WikiNEural + Gemma corrections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            sample = {
                "text": "User John Smith contacted support at support@example.com",
                "wikineural": [
                    ("User", "PERSON"),  # False positive
                    ("John Smith", "PERSON"),  # Correct
                ],
                "gemma_eval": {
                    "correct": False,
                    "false_positives": ["User"],  # Gemma flags the FP
                    "missed_entities": [
                        {"text": "support@example.com", "type": "EMAIL"}  # Missed
                    ]
                }
            }

            ground_truth = builder._build_ground_truth(sample)

            # Should remove "User" false positive
            entity_texts = [e["text"] for e in ground_truth["entities"]]
            assert "User" not in entity_texts
            assert "John Smith" in entity_texts

            # Should add missed email
            assert "support@example.com" in entity_texts

    def test_finalize_creates_train_test_split(self):
        """Test that finalization creates proper train/test split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            # Add 10 samples
            for i in range(10):
                sample = {
                    "text": f"Sample {i} with John Smith",
                    "wikineural": [("John Smith", "PERSON")],
                    "gemma_eval": {
                        "correct": True,
                        "false_positives": [],
                        "missed_entities": []
                    },
                    "disagreement_score": 0.0
                }
                builder.add_sample(sample)

            train_file, test_file = builder.finalize()

            assert train_file.exists()
            assert test_file.exists()

            # Check split (80/20)
            with open(train_file) as f:
                train_count = sum(1 for _ in f)
            with open(test_file) as f:
                test_count = sum(1 for _ in f)

            assert train_count == 8
            assert test_count == 2

    def test_dataset_quality_no_empty_texts(self):
        """Test that generated dataset has no empty texts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            # Add samples with valid texts
            for i in range(5):
                sample = {
                    "text": f"Text {i}: John Smith works at Company {i}",
                    "wikineural": [("John Smith", "PERSON"), (f"Company {i}", "ORG")],
                    "gemma_eval": {"correct": True, "false_positives": [], "missed_entities": []},
                    "disagreement_score": 0.0
                }
                builder.add_sample(sample)

            train_file, test_file = builder.finalize()

            # Verify no empty texts in output
            with open(train_file) as f:
                for line in f:
                    item = json.loads(line)
                    assert item["text"], "Found empty text in training data"
                    assert item["entities"] is not None

    def test_dataset_quality_entity_type_validity(self):
        """Test that all entity types are valid NER types."""
        valid_types = {"PERSON", "ORG", "LOC", "MISC", "EMAIL", "PHONE"}

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            sample = {
                "text": "John Smith at Acme Corp in NYC",
                "wikineural": [
                    ("John Smith", "PERSON"),
                    ("Acme Corp", "ORG"),
                    ("NYC", "LOC"),
                ],
                "gemma_eval": {
                    "correct": True,
                    "false_positives": [],
                    "missed_entities": [
                        {"text": "contact@acme.com", "type": "EMAIL"}
                    ]
                },
                "disagreement_score": 0.0
            }

            builder.add_sample(sample)
            ground_truth = builder._build_ground_truth(sample)

            for entity in ground_truth["entities"]:
                assert entity["type"] in valid_types, f"Invalid type: {entity['type']}"

    def test_dataset_quality_no_duplicate_entities(self):
        """Test that same entity doesn't appear multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))

            sample = {
                "text": "John Smith met John Smith",
                "wikineural": [
                    ("John Smith", "PERSON"),
                    ("John Smith", "PERSON"),  # Duplicate
                ],
                "gemma_eval": {
                    "correct": True,
                    "false_positives": [],
                    "missed_entities": []
                },
                "disagreement_score": 0.0
            }

            builder.add_sample(sample)
            ground_truth = builder._build_ground_truth(sample)

            # Should not have duplicates when merging
            entity_texts = [e["text"] for e in ground_truth["entities"]]
            assert entity_texts.count("John Smith") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
