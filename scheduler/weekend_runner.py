"""
scheduler/weekend_runner.py
Weekend prediction scheduler for ScoreBorga 2.5.

Usage:
    python scheduler/weekend_runner.py           # Start scheduler (runs every Friday)
    python scheduler/weekend_runner.py --run-now  # Run pipeline immediately
"""

import argparse
import logging
import sys
import time

import schedule

from output.dispatcher import run_pipeline
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def job() -> None:
    """Scheduled job: run the full prediction pipeline."""
    logger.info("Scheduled job triggered — running prediction pipeline...")
    try:
        run_pipeline()
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)


def start_scheduler() -> None:
    """Set up and start the schedule loop (runs every Friday at configured time)."""
    run_time = settings.PREDICTION_RUN_TIME
    schedule.every().friday.at(run_time).do(job)
    logger.info("Scheduler started — predictions will run every Friday at %s.", run_time)

    while True:
        schedule.run_pending()
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ScoreBorga 2.5 Weekend Prediction Runner")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the prediction pipeline immediately without scheduling.",
    )
    args = parser.parse_args()

    if args.run_now:
        logger.info("--run-now flag detected — running pipeline immediately.")
        try:
            message = run_pipeline()
            print(message)
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            sys.exit(1)
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
