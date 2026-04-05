"""
Validator module for NER model fine-tuning and evaluation.

Reads from data/requests-sample.jsonl, uses Gemma 4 to validate WikiNEural predictions,
collects disagreements to build a fine-tuning dataset, trains an improved model,
and selects the best-performing version.
"""
