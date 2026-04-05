"""
Tester: Evaluate baseline and fine-tuned models, select the better one.

Compares F1 scores on held-out test data and keeps the best model.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Tuple

from transformers import pipeline
from seqeval.metrics import classification_report, f1_score

log = logging.getLogger(__name__)

_BASELINE_MODEL = "Babelscape/wikineural-multilingual-ner"


class Tester:
    """Evaluate and compare NER models."""

    def __init__(self, models_dir: Path, test_data_dir: Path):
        self.models_dir = Path(models_dir)
        self.test_data_dir = Path(test_data_dir)

        self.test_file = self.test_data_dir / "test_data.jsonl"
        self.baseline_checkpoint = self.models_dir / "baseline"
        self.finetuned_checkpoint = self._find_latest_finetuned()

    def _find_latest_finetuned(self) -> Path:
        """Find the most recently fine-tuned model."""
        finetuned_models = sorted(
            self.models_dir.glob("wikineural_finetuned_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if finetuned_models:
            return finetuned_models[0]
        return None

    def _load_test_data(self) -> list:
        """Load test data from JSONL file."""
        data = []
        with open(self.test_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _evaluate_model(self, model_name_or_path: str) -> Tuple[float, dict]:
        """
        Evaluate a model on test data.

        Returns:
            (f1_score, detailed_report)
        """
        log.info(f"Loading model: {model_name_or_path}")
        import torch
        ner_pipe = pipeline(
            "ner",
            model=model_name_or_path,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,  # GPU if available, else CPU
        )

        test_data = self._load_test_data()

        # Accumulate predictions
        all_preds = []
        all_true = []

        for sample in test_data:
            text = sample["text"]
            entities = sample.get("entities", [])

            if not text or not entities:  # Skip empty samples
                continue

            # Get model predictions
            preds = ner_pipe(text)

            # Convert to BIO format for comparison
            pred_labels = self._convert_to_bio(text, preds)
            true_labels = self._convert_to_bio(text, entities)

            # Ensure equal length for seqeval
            min_len = min(len(pred_labels), len(true_labels))
            all_preds.append(pred_labels[:min_len])
            all_true.append(true_labels[:min_len])

        # Calculate F1 (seqeval expects list of lists)
        if all_preds and all_true:
            f1 = f1_score(all_true, all_preds)
            report = classification_report(all_true, all_preds)
        else:
            f1 = 0.0
            report = "No test data"

        return f1, report

    def _convert_to_bio(self, text: str, entities: list) -> list:
        """Simple BIO label conversion from entity list."""
        tokens = text.split()
        labels = ["O"] * len(tokens)

        # Map each token to entity type (simplified)
        for entity in entities:
            ent_text = entity.get("text") if isinstance(entity, dict) else entity
            for i, token in enumerate(tokens):
                if ent_text.lower() in token.lower():
                    ent_type = entity.get("type", "PER") if isinstance(entity, dict) else "PER"
                    labels[i] = f"B-{ent_type}"

        return labels

    def evaluate_both(self) -> Tuple[float, float]:
        """
        Evaluate both baseline and fine-tuned models.

        Returns:
            (baseline_f1, finetuned_f1)
        """
        log.info("Evaluating baseline model...")
        baseline_f1, baseline_report = self._evaluate_model(_BASELINE_MODEL)
        log.info(f"Baseline F1: {baseline_f1:.4f}\n{baseline_report}")

        if self.finetuned_checkpoint and self.finetuned_checkpoint.exists():
            log.info("Evaluating fine-tuned model...")
            finetuned_f1, finetuned_report = self._evaluate_model(str(self.finetuned_checkpoint))
            log.info(f"Fine-tuned F1: {finetuned_f1:.4f}\n{finetuned_report}")
        else:
            log.warning("No fine-tuned model found")
            finetuned_f1 = 0.0

        return baseline_f1, finetuned_f1

    def keep_finetuned(self):
        """
        Fine-tuned model is better — back it up as the new baseline.
        """
        if not self.finetuned_checkpoint or not self.finetuned_checkpoint.exists():
            log.warning("Fine-tuned model not found, cannot keep")
            return

        # Archive old baseline
        if self.baseline_checkpoint.exists():
            archive_dir = self.models_dir / "baseline_archive"
            archive_dir.mkdir(exist_ok=True)
            shutil.rmtree(archive_dir, ignore_errors=True)
            shutil.move(str(self.baseline_checkpoint), str(archive_dir))
            log.info(f"Archived old baseline to {archive_dir}")

        # Move fine-tuned to baseline
        shutil.move(str(self.finetuned_checkpoint), str(self.baseline_checkpoint))
        log.info(f"Promoted fine-tuned model to baseline at {self.baseline_checkpoint}")

    def delete_finetuned(self):
        """
        Fine-tuned model is not better — delete it.
        """
        if self.finetuned_checkpoint and self.finetuned_checkpoint.exists():
            shutil.rmtree(self.finetuned_checkpoint)
            log.info(f"Deleted fine-tuned model {self.finetuned_checkpoint}")
