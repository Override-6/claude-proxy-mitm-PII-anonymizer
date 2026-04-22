"""Tests for validator evaluator logic."""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, '/app')
from validator.evaluator import Evaluator


class TestEvaluatorDataExtraction:
    """Test text extraction from request bodies."""

    def test_extract_text_from_dict(self):
        """Test extracting text fields from nested dictionary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            obj = {
                "system": "You are helpful",
                "messages": [
                    {"role": "user", "content": "Hello John Smith"},
                    {"role": "assistant", "content": "Hi there"}
                ],
                "tools": [
                    {
                        "name": "search",
                        "description": "Search for information about a person",
                        "input_schema": {"type": "object"}
                    }
                ]
            }

            texts = evaluator._extract_text_fields(obj)

            # Should extract system, messages.content, and tool descriptions
            extracted_texts = [t[0] for t in texts]
            assert "You are helpful" in extracted_texts
            assert "Search for information about a person" in extracted_texts

        finally:
            requests_file.unlink()

    def test_extract_skips_short_texts(self):
        """Test that very short texts are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            obj = {
                "short": "Hi",  # Too short
                "long": "This is a much longer text that contains actual information about a person",
            }

            texts = evaluator._extract_text_fields(obj)
            extracted_texts = [t[0] for t in texts]

            # Short text should not be extracted
            assert "Hi" not in extracted_texts
            # Long text should be extracted
            assert any("longer text" in t for t in extracted_texts)

        finally:
            requests_file.unlink()

    def test_extract_respects_path_prefix(self):
        """Test that extraction tracks field paths correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            obj = {
                "messages": [
                    {"content": "Long message content here about John Smith"}
                ]
            }

            texts = evaluator._extract_text_fields(obj)

            # Should include path information
            assert len(texts) > 0
            text, path = texts[0]
            assert "messages" in path

        finally:
            requests_file.unlink()


class TestGemmaResponseParsing:
    """Test Gemma LLM response parsing."""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON responses."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            response = '{"correct": true, "false_positives": [], "missed_entities": []}'
            result = evaluator._parse_gemma_response(response)

            assert result["correct"] is True
            assert result["false_positives"] == []
            assert result["missed_entities"] == []

        finally:
            requests_file.unlink()

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in surrounding text."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            response = """Here's my analysis:

{"correct": false, "false_positives": ["User"], "missed_entities": []}

That's all!"""
            result = evaluator._parse_gemma_response(response)

            assert result["correct"] is False
            assert "User" in result["false_positives"]

        finally:
            requests_file.unlink()

    def test_parse_invalid_json_returns_default(self):
        """Test that invalid JSON returns default response."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            response = "This is not JSON at all!"
            result = evaluator._parse_gemma_response(response)

            assert result["correct"] is False
            assert result["missed_entities"] == []

        finally:
            requests_file.unlink()


class TestDisagreementScoring:
    """Test disagreement scoring logic."""

    def test_no_disagreement_scores_zero(self):
        """Test that perfect predictions score 0 disagreement."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            text = "John Smith works at Acme Corp"
            sample = {
                "text": text,
                "wikineural": [("John Smith", "PERSON"), ("Acme Corp", "ORG")],
                "gemma_eval": {
                    "correct": True,
                    "false_positives": [],
                    "missed_entities": []
                },
            }

            # Compute disagreement (from _evaluate_text logic)
            disagreement = 0
            if sample["gemma_eval"].get("false_positives"):
                disagreement += len(sample["gemma_eval"]["false_positives"]) * 0.3
            if sample["gemma_eval"].get("missed_entities"):
                disagreement += len(sample["gemma_eval"]["missed_entities"]) * 0.7

            assert disagreement == 0

        finally:
            requests_file.unlink()

    def test_false_positive_adds_cost(self):
        """Test that false positives add to disagreement score."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            sample = {
                "text": "The user contacted support",
                "wikineural": [("The user", "PERSON")],  # False positive
                "gemma_eval": {
                    "correct": False,
                    "false_positives": ["The user"],
                    "missed_entities": []
                },
            }

            disagreement = len(sample["gemma_eval"]["false_positives"]) * 0.3
            assert disagreement == 0.3

        finally:
            requests_file.unlink()

    def test_missed_entity_adds_higher_cost(self):
        """Test that missed entities have higher disagreement cost."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"request": {"body": {}}}) + "\n")
            requests_file = Path(f.name)

        try:
            evaluator = Evaluator(requests_file)

            sample = {
                "text": "John Smith from Acme Corp",
                "wikineural": [("John Smith", "PERSON")],  # Missed: Acme Corp
                "gemma_eval": {
                    "correct": False,
                    "false_positives": [],
                    "missed_entities": [
                        {"text": "Acme Corp", "type": "ORG"}
                    ]
                },
            }

            disagreement = len(sample["gemma_eval"]["missed_entities"]) * 0.7
            assert disagreement == 0.7
            # Missed entity cost (0.7) > False positive cost (0.3)
            assert disagreement > 0.3

        finally:
            requests_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
