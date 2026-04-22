"""
Validator orchestration — collect disagreements, fine-tune, and test NER models.

Usage (CLI):
  poetry run python validator/main.py --collect           # Collect disagreements
  poetry run python validator/main.py --finetune          # Train new model
  poetry run python validator/main.py --test              # Evaluate & select best
  poetry run python validator/main.py --all               # Full pipeline
  poetry run python validator/main.py --daemon            # Run as scheduled service

Environment variables (for scheduled mode):
  VALIDATOR_SCHEDULE_ENABLED=true/false                  # Enable scheduler (default: true)
  VALIDATOR_SCHEDULE_DAY=sun                             # Cron day (sun, mon, ..., sat)
  VALIDATOR_SCHEDULE_HOUR=2                              # Hour (0-23)
  VALIDATOR_SCHEDULE_MINUTE=0                            # Minute (0-59)
  VALIDATOR_REQUEST_LIMIT=200                            # Max requests to process
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from validator.dataset_builder import DatasetBuilder
from validator.evaluator import Evaluator
from validator.trainer import Trainer
from validator.tester import Tester
from validator.scheduler import ValidatorScheduler

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s: %(levelname)s: %(message)s'
)

# Use relative paths from repo root
repo_root = Path(__file__).parent.parent
DATA_DIR = repo_root / "data"
MODELS_DIR = repo_root / "models"

VALIDATOR_DIR = DATA_DIR / "validator"
ENTITIES_FILE = DATA_DIR / "requests-entities.jsonl"

# Ensure directories exist
VALIDATOR_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def collect_disagreements(limit: int = None):
    """Collect disagreements between WikiNEural and Gemma 4."""
    log.info("Starting disagreement collection...")

    evaluator = Evaluator(entities_file=ENTITIES_FILE)
    dataset_builder = DatasetBuilder(output_dir=VALIDATOR_DIR)

    count = 0
    for sample in evaluator.evaluate_batch(limit=limit):
        dataset_builder.add_sample(sample)
        count += 1

    dataset_builder.finalize()
    log.info(f"✓ Collected {count} samples (Gemma-validated). Dataset saved to {VALIDATOR_DIR}")


def finetune_model():
    """Fine-tune WikiNEural on collected disagreements."""
    log.info("Starting fine-tuning...")

    dataset_file = VALIDATOR_DIR / "training_data.jsonl"
    if not dataset_file.exists():
        log.error(f"Training dataset not found: {dataset_file}")
        log.error("Run with --collect first to generate training data")
        return False

    # Check if dataset is empty
    sample_count = 0
    try:
        with open(dataset_file, "r") as f:
            sample_count = sum(1 for _ in f)
    except Exception as e:
        log.error(f"Could not read dataset: {e}")
        return False

    if sample_count == 0:
        log.error("Training dataset is empty. No disagreements were collected.")
        log.error("Run with --collect first to generate training data with disagreements")
        return False

    trainer = Trainer(dataset_file=dataset_file, output_dir=MODELS_DIR)
    model_path = trainer.train()

    log.info(f"Fine-tuning complete. Model saved to {model_path}")
    return True


def test_models():
    """Evaluate baseline and fine-tuned models, keep the better one."""
    log.info("Starting model evaluation...")

    tester = Tester(models_dir=MODELS_DIR, test_data_dir=VALIDATOR_DIR)
    baseline_score, finetuned_score = tester.evaluate_both()

    log.info(f"Baseline (WikiNEural) F1: {baseline_score:.4f}")
    log.info(f"Fine-tuned model F1: {finetuned_score:.4f}")

    if finetuned_score > baseline_score:
        log.info(f"✅ Fine-tuned model is better (+{finetuned_score - baseline_score:.4f}). Keeping it.")
        tester.keep_finetuned()
        return True
    else:
        log.info(f"❌ Fine-tuned model is not better. Deleting and keeping baseline.")
        tester.delete_finetuned()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validator: collect disagreements, fine-tune NER, and select best model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Collect disagreements:
    python validator/main.py --collect --limit 50

  Full pipeline (manual run):
    python validator/main.py --all --limit 200

  Run as scheduled service (daemon):
    python validator/main.py --daemon

  Manually trigger pipeline in daemon mode:
    echo "trigger" | nc 127.0.0.1 9998
        """
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect disagreements between WikiNEural and Gemma 4 evaluator"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune WikiNEural model on collected disagreements"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate both baseline and fine-tuned models, keep the better one"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: collect → finetune → test → select best"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to collect (unique texts evaluated by Gemma)"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a scheduled service (long-running, executes pipeline on schedule)"
    )

    args = parser.parse_args()

    # Daemon mode: run scheduled service
    if args.daemon:
        log.info("Starting validator in daemon mode (scheduled service)...")
        scheduler = ValidatorScheduler()

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            log.info("Received SIGTERM/SIGINT, shutting down gracefully...")
            scheduler.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        scheduler.start()

        # Keep the process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt, shutting down...")
            scheduler.stop()

    # CLI mode: run operations
    elif args.all:
        # Full pipeline
        collect_disagreements(limit=args.limit)
        if finetune_model():
            test_models()
    else:
        if args.collect:
            collect_disagreements(limit=args.limit)
        if args.finetune:
            finetune_model()
        if args.test:
            test_models()

        if not any([args.collect, args.finetune, args.test]):
            parser.print_help()


if __name__ == "__main__":
    main()
