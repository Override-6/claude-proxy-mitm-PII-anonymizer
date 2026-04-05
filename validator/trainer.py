"""
Trainer: Fine-tune WikiNEural on collected dataset.

Uses HuggingFace transformers to fine-tune the token classification model.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer as HFTrainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset

log = logging.getLogger(__name__)

_MODEL_NAME = "Babelscape/wikineural-multilingual-ner"
_LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
}
_ID2LABEL = {v: k for k, v in _LABEL2ID.items()}


class Trainer:
    """Fine-tune WikiNEural model on domain-specific NER data."""

    def __init__(self, dataset_file: Path, output_dir: Path):
        self.dataset_file = dataset_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_output_dir = self.output_dir / f"wikineural_finetuned_{timestamp}"

        log.info(f"Loading base model: {_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, model_max_length=512)
        self.model = AutoModelForTokenClassification.from_pretrained(_MODEL_NAME)

    def _load_dataset(self) -> Dataset:
        """Load training data from JSONL file."""
        data = []
        with open(self.dataset_file, "r") as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

        return Dataset.from_dict({
            "text": [d["text"] for d in data],
            "entities": [d["entities"] for d in data]
        })

    def _tokenize_and_align_labels(self, examples):
        """
        Tokenize text and align BIO labels with subword tokens.
        """
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            is_split_into_words=False,
            max_length=512,
        )

        labels = []
        for i, entities in enumerate(examples["entities"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                else:
                    # Find entity for this word position
                    text = examples["text"][i]
                    entity_label = "O"

                    # Simple approach: check if word matches any entity
                    for entity in entities:
                        if entity["text"] in text:
                            entity_label = f"B-{entity['type']}"
                            break

                    label_ids.append(_LABEL2ID.get(entity_label, 0))

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self) -> Path:
        """Fine-tune the model."""
        log.info(f"Loading dataset from {self.dataset_file}")
        dataset = self._load_dataset()

        log.info(f"Dataset size: {len(dataset)}")

        # Tokenize
        tokenized = dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
        )

        # Training arguments (GPU-optimized)
        import torch
        training_args = TrainingArguments(
            output_dir=str(self.model_output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16 if torch.cuda.is_available() else 8,  # Larger batch on GPU
            per_device_eval_batch_size=16 if torch.cuda.is_available() else 8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            learning_rate=2e-5,
            save_strategy="epoch",
            save_total_limit=1,
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,
            fp16=torch.cuda.is_available(),  # Mixed precision on GPU
            optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",  # 8bit optimizer on GPU
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Trainer
        trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        log.info("Starting fine-tuning...")
        trainer.train()

        log.info(f"Saving fine-tuned model to {self.model_output_dir}")
        trainer.save_model(str(self.model_output_dir))
        self.tokenizer.save_pretrained(str(self.model_output_dir))

        return self.model_output_dir
