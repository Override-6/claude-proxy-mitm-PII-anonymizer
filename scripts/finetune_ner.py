#!/usr/bin/env python3
"""
Fine-tune google-bert/bert-base-multilingual-uncased for NER (PER, ORG, LOC)
using the WikiANN/wikiann dataset from HuggingFace.

Trains on English, French, Italian, Spanish, and German by default.
"""

import argparse
import logging
import os
import sys
from collections import Counter

# Use cached models/datasets — avoids SSL issues from the local MITM proxy.
# Override by setting these env vars to "0" when a fresh download is needed.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# WikiANN uses the IOB2 scheme with three entity types.
# Label ids in the raw dataset:
#   0 -> O, 1 -> B-PER, 2 -> I-PER, 3 -> B-ORG, 4 -> I-ORG,
#   5 -> B-LOC, 6 -> I-LOC
# We keep these as-is since they already follow BIO for PER/ORG/LOC.
# ---------------------------------------------------------------------------

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

MODEL_NAME = "google-bert/bert-base-multilingual-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune mBERT for multilingual NER on WikiANN"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/bert-multilingual-uncased-ner",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,fr",
        help="Comma-separated language codes (default: en,fr,it,es,de)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data/model, train for 1 step, then exit (for testing)",
    )
    return parser.parse_args()


# ---- data helpers ---------------------------------------------------------


def load_and_combine(languages: list[str], split: str):
    """Load wikiann for each language and concatenate."""
    parts = []
    for lang in languages:
        ds = load_dataset("wikiann", lang, split=split)
        parts.append(ds)
    return concatenate_datasets(parts)


def print_label_distribution(dataset, tag: str):
    """Print label counts across all examples."""
    counts: Counter = Counter()
    for example in dataset:
        for label_id in example["ner_tags"]:
            counts[LABEL_LIST[label_id]] += 1
    total = sum(counts.values())
    print(f"\n--- Label distribution ({tag}, {len(dataset)} examples) ---")
    for label in LABEL_LIST:
        c = counts.get(label, 0)
        print(f"  {label:6s}: {c:>9,}  ({100 * c / total:.1f}%)")
    print(f"  {'TOTAL':6s}: {total:>9,}")


# ---- tokenization ---------------------------------------------------------


def tokenize_and_align(examples, tokenizer):
    """
    Tokenize inputs and realign labels to sub-word tokens.
    The first sub-word of each word gets the original label;
    subsequent sub-words get -100 (ignored in loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # special tokens
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # first sub-token of a word
                label_ids.append(labels[word_id])
            else:
                # subsequent sub-tokens
                label_ids.append(-100)
            previous_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ---- metrics --------------------------------------------------------------


def build_compute_metrics():
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(predictions, labels):
            t_labels = []
            t_preds = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                t_labels.append(LABEL_LIST[l])
                t_preds.append(LABEL_LIST[p])
            true_labels.append(t_labels)
            true_preds.append(t_preds)

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    return compute_metrics


# ---- sample predictions ---------------------------------------------------


def show_sample_predictions(model, tokenizer, dataset, n=5, device="cpu"):
    """Decode a few examples from the dataset and print predictions."""
    model.eval()
    model.to(device)
    print(f"\n--- Sample predictions (first {n} validation examples) ---")
    for idx in range(min(n, len(dataset))):
        tokens = dataset[idx]["tokens"]
        ner_tags = dataset[idx]["ner_tags"]

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        word_ids = encoded.word_ids(batch_index=0)

        pred_labels = []
        prev_word_id = None
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None or wid == prev_word_id:
                prev_word_id = wid
                continue
            pred_labels.append(LABEL_LIST[pid])
            prev_word_id = wid

        gold_labels = [LABEL_LIST[t] for t in ner_tags]

        print(f"\nExample {idx + 1}:")
        for tok, gold, pred in zip(tokens, gold_labels, pred_labels):
            marker = " " if gold == pred else "*"
            print(f"  {marker} {tok:20s}  gold={gold:6s}  pred={pred:6s}")


# ---- main -----------------------------------------------------------------


def main():
    args = parse_args()
    languages = [l.strip() for l in args.languages.split(",")]

    print(f"Model       : {MODEL_NAME}")
    print(f"Languages   : {languages}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"LR          : {args.learning_rate}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Dry-run     : {args.dry_run}")

    # ---- load data --------------------------------------------------------
    print("\nLoading datasets ...")
    train_raw = load_and_combine(languages, "train")
    val_raw = load_and_combine(languages, "validation")

    print(f"Train examples : {len(train_raw):,}")
    print(f"Val examples   : {len(val_raw):,}")

    print_label_distribution(train_raw, "train")

    # ---- tokenizer & model ------------------------------------------------
    print("\nLoading tokenizer and model ...")
    # transformers 5.x incorrectly flags BERT tokenizers with a Mistral regex
    # warning — it's a false positive; suppress it.
    logging.getLogger("transformers.tokenization_utils_tokenizers").addFilter(
        lambda r: "incorrect regex pattern" not in r.getMessage()
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ---- tokenize datasets ------------------------------------------------
    print("Tokenizing datasets ...")
    train_ds = train_raw.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    val_ds = val_raw.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # ---- training args ----------------------------------------------------
    effective_epochs = args.epochs
    if args.dry_run:
        effective_epochs = 1

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=effective_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="no" if args.dry_run else "epoch",
        save_strategy="no" if args.dry_run else "epoch",
        load_best_model_at_end=not args.dry_run,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        max_steps=1 if args.dry_run else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    # ---- train ------------------------------------------------------------
    print("\nStarting training ...")
    trainer.train()

    if args.dry_run:
        print("\n[dry-run] Training completed after 1 step. Exiting.")
        # Still show a few sample predictions
        device = "cuda" if torch.cuda.is_available() else "cpu"
        show_sample_predictions(model, tokenizer, val_raw, n=3, device=device)
        sys.exit(0)

    # ---- evaluate ---------------------------------------------------------
    print("\nRunning final evaluation ...")
    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")

    # ---- save -------------------------------------------------------------
    print(f"\nSaving model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- sample predictions -----------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    show_sample_predictions(model, tokenizer, val_raw, n=5, device=device)

    print("\nDone.")


if __name__ == "__main__":
    main()
