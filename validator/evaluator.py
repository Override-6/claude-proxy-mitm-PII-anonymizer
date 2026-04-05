"""
Evaluator: Use Gemma 4 8B to validate WikiNEural NER predictions.

Compares predictions and generates ground truth labels for fine-tuning dataset.
Uses pre-extracted entities from data/requests-entities.jsonl when available.
"""

import json
import logging
from pathlib import Path
from typing import Generator

from transformers import pipeline

from proxy.entity_cache_log import get_cached_entities

log = logging.getLogger(__name__)


class Evaluator:
    """Use Gemma 4 to validate WikiNEural entity extraction."""

    def __init__(self, entities_file: Path):
        self.entities_file = entities_file

        if not self.entities_file.exists():
            raise FileNotFoundError(f"Entity cache not found: {self.entities_file}")

        log.info("Loading TinyLlama-1.1B for evaluation (GPU)...")
        import torch
        self.gemma = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="auto",
            dtype=torch.float16,  # Mixed precision for GPU
            max_new_tokens=300,
        )

    def _extract_text_fields(self, obj: dict, path_prefix: str = "") -> list[tuple[str, str]]:
        """
        Recursively extract all text fields from a request object.
        Returns list of (text, field_path) tuples for relevant PII fields.
        """
        texts = []

        # Target fields (based on rules.jsonc)
        relevant_paths = {
            "system", "messages", "prompt", "tools", "description",
            "email", "name", "title", "content", "text", "input_schema"
        }

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path_prefix}.{key}" if path_prefix else key

                if key in relevant_paths or "description" in key.lower() or "text" in key.lower():
                    if isinstance(value, str) and len(value) > 20 and len(value) < 5000:
                        texts.append((value, current_path))
                    elif isinstance(value, (dict, list)):
                        texts.extend(self._extract_text_fields(value, current_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                texts.extend(self._extract_text_fields(item, f"{path_prefix}[{i}]"))

        return texts

    def _parse_gemma_response(self, response: str) -> dict:
        """
        Parse Gemma's JSON response. Gemma may output text before/after JSON,
        so extract the JSON block.
        """
        # Try to find JSON block
        start = response.find("{")
        end = response.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                log.warning(f"Failed to parse Gemma response: {response[:200]}")
                return {"correct": False, "missed": []}

        return {"correct": False, "missed": []}

    def _evaluate_entities(self, text: str, entities) -> dict:
        """
        Evaluate entities already extracted by proxy using Gemma.
        """
        # Build Gemma evaluation prompt with chain-of-thought
        extracted_list = [f'"{e.text}" as {e.type}' for e in entities]
        extracted_str = ", ".join(extracted_list) if extracted_list else "no entities"

        # Chain-of-thought prompt: ask model to reason FIRST, then output JSON
        prompt = f"""You are a PII validation expert for API requests and code contexts.
Your task is to evaluate whether an NER model correctly identified personal/sensitive information.

TEXT TO ANALYZE:
"{text}"

ENTITIES FOUND BY WikiNEural NER:
{extracted_str}

IMPORTANT CONTEXT:
- This text is from an API request or code context, not a document
- Generic terms like "User", "Admin", "Developer", "The Engineer" are NOT personal data
- Role descriptions like "role: 'Admin'" should not be flagged as PII
- Only flag ACTUAL names of specific people, organizations, or locations / addresses
- Text can be either French or English, or a mix of both

THINK STEP BY STEP:
1. For each detected entity, is it ACTUALLY PII in this context?
   - If it's a generic placeholder or role label, it's a false positive
   - If it's someone's real name or organization name, it's correct
2. Did the model miss any real PII?
   - Look for names of people, company names, locations, addresses
   - Ignore generic terms and placeholders, pronouns etc
3. Provide your confidence (high/medium/low) for your assessment

After thinking through the analysis, provide ONLY a JSON response (nothing else after the JSON):
{{
  "correct": true/false,
  "false_positives": ["entity1", "entity2"],
  "missed_entities": [{{"text": "...", "type": "PERSON/ORG/LOC/EMAIL/PHONE"}}],
  "confidence": "high/medium/low"
}}"""

        try:
            response = self.gemma(
                prompt,
                return_full_text=False,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
            )
            response_text = response[0]["generated_text"] if response else ""
            gemma_eval = self._parse_gemma_response(response_text)
        except Exception as e:
            log.warning(f"Gemma evaluation failed: {e}")
            gemma_eval = {
                "correct": True,
                "false_positives": [],
                "missed_entities": [],
                "confidence": "low"
            }

        # Compute disagreement score
        disagreement = 0
        if gemma_eval.get("false_positives"):
            disagreement += len(gemma_eval["false_positives"]) * 0.3
        if gemma_eval.get("missed_entities"):
            disagreement += len(gemma_eval["missed_entities"]) * 0.7

        return {
            "text": text,
            "wikineural": [(e.text, e.type) for e in entities],
            "gemma_eval": gemma_eval,
            "disagreement_score": disagreement,
        }

    def _evaluate_text(self, text: str, request_url: str = None, field_path: str = None) -> dict:
        """
        Run WikiNEural + Gemma evaluation on a single text.
        Uses cached entities if available (from proxy), otherwise runs NER.
        Uses chain-of-thought prompting: model thinks first, then outputs JSON.
        Returns disagreement info.
        """
        # Try to use cached entities from proxy first
        cached_ents = None
        if request_url and field_path:
            cached_ents = get_cached_entities(request_url, field_path, text)

        if cached_ents:
            # Convert cached entity dicts to Entity-like objects
            entities = [
                type('Entity', (), {'text': e['text'], 'type': e['type'], 'start': e['start'], 'end': e['end']})()
                for e in cached_ents
            ]
            log.debug(f"Using {len(entities)} cached entities for: {text[:50]}...")
        else:
            # Run WikiNEural extraction
            entities = list(self.ner.find_entities_batch([text], mappings=None))[0]
            log.debug(f"Extracted {len(entities)} entities via NER for: {text[:50]}...")

        # Filter out pronouns/articles that WikiNEural might incorrectly flag
        pronouns = {"i", "you", "he", "she", "it", "we", "they", "the", "a", "an"}
        entities = [e for e in entities if e.text.lower() not in pronouns and len(e.text) > 2]

        # Build Gemma evaluation prompt with chain-of-thought
        extracted_list = [f'"{e.text}" as {e.type}' for e in entities]
        extracted_str = ", ".join(extracted_list) if extracted_list else "no entities"

        # Chain-of-thought prompt: ask model to reason FIRST, then output JSON
        prompt = f"""You are a PII validation expert for API requests and code contexts.
Your task is to evaluate whether an NER model correctly identified personal/sensitive information.

TEXT TO ANALYZE:
"{text}"

ENTITIES FOUND BY WikiNEural NER:
{extracted_str}

IMPORTANT CONTEXT:
- This text is from an API request or code context, not a document
- Generic terms like "User", "Admin", "Developer", "The Engineer" are NOT personal data
- Role descriptions like "role: 'Admin'" should not be flagged as PII
- Only flag ACTUAL names of specific people, organizations, or locations / addresses
- Text can be either French or English, or a mix of both

THINK STEP BY STEP:
1. For each detected entity, is it ACTUALLY PII in this context?
   - If it's a generic placeholder or role label, it's a false positive
   - If it's someone's real name or organization name, it's correct
2. Did the model miss any real PII?
   - Look for names of people, company names, locations, addresses
   - Ignore generic terms and placeholders, pronouns etc
3. Provide your confidence (high/medium/low) for your assessment

After thinking through the analysis, provide ONLY a JSON response (nothing else after the JSON):
{{
  "correct": boolean,
  "false_positives": ["specific false positive entities", ...],
  "missed_entities": [
    {{"text": "actual PII text", "type": "PERSON|ORG|LOCATION", "reason": "why this is PII"}}
  ],
  "confidence": "high|medium|low"
}}"""

        # Run Gemma
        try:
            result = self.gemma(prompt, max_new_tokens=500)
            response_text = result[0]["generated_text"]

            # Extract the response (skip the prompt echo)
            # Look for where JSON starts
            if "{" in response_text:
                # Find the last opening brace and start from there
                json_start = response_text.rfind("{")
                response_text = response_text[json_start:]

            gemma_eval = self._parse_gemma_response(response_text)
        except Exception as e:
            log.warning(f"Gemma evaluation failed: {e}")
            gemma_eval = {
                "correct": True,
                "false_positives": [],
                "missed_entities": [],
                "confidence": "low"
            }

        # Compute disagreement score, adjusted by confidence
        disagreement = 0
        confidence_multiplier = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.3,
        }.get(gemma_eval.get("confidence", "medium"), 0.7)

        # False positives reduce confidence
        if gemma_eval.get("false_positives"):
            disagreement += len(gemma_eval["false_positives"]) * 0.3 * confidence_multiplier

        # Missed entities (high cost)
        if gemma_eval.get("missed_entities"):
            disagreement += len(gemma_eval["missed_entities"]) * 0.7 * confidence_multiplier

        return {
            "text": text,
            "wikineural": [(e.text, e.type) for e in entities],
            "gemma_eval": gemma_eval,
            "disagreement_score": disagreement,
        }

    def evaluate_batch(self, limit: int = None) -> Generator[dict, None, None]:
        """
        Process entities directly from requests-entities.jsonl.
        Evaluates already-extracted entities via Gemma.
        Yields disagreement samples for training dataset.
        """
        log.info(f"Processing entities from {self.entities_file}")

        count = 0
        processed = 0
        skipped = 0
        disagreements = 0

        with open(self.entities_file, "r") as f:
            for line in f:
                if limit and disagreements >= limit:
                    break

                try:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    processed += 1

                    if not text or len(text) < 10:
                        skipped += 1
                        if processed % 50 == 0:
                            log.info(f"[Progress] Processed: {processed}, Skipped: {skipped}, Disagreements: {disagreements}")
                        continue

                    # Get entities already extracted by proxy
                    entities_list = entry.get("entities", [])
                    if not entities_list:
                        skipped += 1
                        if processed % 50 == 0:
                            log.info(f"[Progress] Processed: {processed}, Skipped: {skipped}, Disagreements: {disagreements}")
                        continue

                    # Convert to Entity-like objects
                    entities = [
                        type('Entity', (), {
                            'text': e['text'],
                            'type': e['type'],
                            'start': e.get('start', 0),
                            'end': e.get('end', 0)
                        })()
                        for e in entities_list
                    ]

                    # Filter out pronouns/articles
                    pronouns = {"i", "you", "he", "she", "it", "we", "they", "the", "a", "an"}
                    entities = [e for e in entities if e.text.lower() not in pronouns and len(e.text) > 2]

                    if not entities:
                        skipped += 1
                        if processed % 50 == 0:
                            log.info(f"[Progress] Processed: {processed}, Skipped: {skipped}, Disagreements: {disagreements}")
                        continue

                    # Evaluate with Gemma
                    result = self._evaluate_entities(text, entities)
                    if result.get("disagreement_score", 0) > 0:
                        yield result
                        disagreements += 1
                        log.debug(f"[Disagreement {disagreements}] Score: {result['disagreement_score']:.2f}, Text: {text[:60]}...")

                    if processed % 50 == 0:
                        log.info(f"[Progress] Processed: {processed}, Skipped: {skipped}, Disagreements: {disagreements}")

                except Exception as e:
                    log.warning(f"Error processing entity: {e}")
                    continue

        log.info(f"[Complete] Total processed: {processed}, Skipped: {skipped}, Disagreements found: {disagreements}")
