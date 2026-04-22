"""
Scheduled validator runner using APScheduler.

Runs the validator pipeline on a recurring schedule inside the container.
No host cron needed — everything is self-contained.
"""

import logging
import os
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger(__name__)


class ValidatorScheduler:
    """Manages scheduled validator runs."""

    def __init__(self, schedule_enabled: bool = True):
        self.scheduler = BackgroundScheduler(daemon=True)
        self.schedule_enabled = schedule_enabled

        # Get schedule from environment variables
        self.schedule_day = os.getenv("VALIDATOR_SCHEDULE_DAY", "sun")  # sun, mon, tue, etc.
        self.schedule_hour = int(os.getenv("VALIDATOR_SCHEDULE_HOUR", "2"))  # 0-23
        self.schedule_minute = int(os.getenv("VALIDATOR_SCHEDULE_MINUTE", "0"))  # 0-59
        self.request_limit = int(os.getenv("VALIDATOR_REQUEST_LIMIT", "200"))

    def run_full_pipeline(self):
        """Execute the full validation pipeline: collect → finetune → test → select best."""
        # Import locally to avoid circular imports
        from validator.main import collect_disagreements, finetune_model, test_models

        timestamp = datetime.now().isoformat()
        log.info(f"[{timestamp}] Starting scheduled validator pipeline...")

        try:
            log.info(f"Step 1/3: Collecting disagreements from {self.request_limit} requests...")
            collect_disagreements(limit=self.request_limit)

            log.info("Step 2/3: Fine-tuning model on collected data...")
            if not finetune_model():
                log.error("Fine-tuning failed, skipping test phase")
                return

            log.info("Step 3/3: Testing and selecting best model...")
            test_models()

            log.info("✅ Scheduled validator pipeline completed successfully")

        except Exception as e:
            log.error(f"❌ Scheduled validator pipeline failed: {e}", exc_info=True)

    def start(self):
        """Start the scheduler."""
        if not self.schedule_enabled:
            log.info("Validator scheduler disabled (VALIDATOR_SCHEDULE_ENABLED=false)")
            return

        log.info(
            f"Starting validator scheduler: "
            f"{self.schedule_day.upper()} {self.schedule_hour:02d}:{self.schedule_minute:02d}"
        )

        # Schedule using cron trigger
        self.scheduler.add_job(
            self.run_full_pipeline,
            CronTrigger(
                day_of_week=self.schedule_day,
                hour=self.schedule_hour,
                minute=self.schedule_minute,
            ),
            id="validator_pipeline",
            name="Full validator pipeline (collect → finetune → test)",
            replace_existing=True,
        )

        self.scheduler.start()
        log.info("Validator scheduler started")

    def stop(self):
        """Stop the scheduler gracefully."""
        if self.scheduler.running:
            log.info("Stopping validator scheduler...")
            self.scheduler.shutdown()
            log.info("Validator scheduler stopped")

    def run_once_blocking(self):
        """Run the pipeline once, blocking until complete."""
        log.info("Running validator pipeline (blocking)...")
        self.run_full_pipeline()
        log.info("Pipeline completed")

    def trigger_now(self):
        """Manually trigger the pipeline immediately."""
        log.info("Manually triggering validator pipeline...")
        self.run_full_pipeline()
