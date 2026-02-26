"""
scheduler/weekend_runner.py
Weekend prediction scheduler for ScoreBorga 2.5.

Usage:
    python scheduler/weekend_runner.py                     # Start weekly scheduler
    python scheduler/weekend_runner.py --run-now           # Run pipeline immediately (no scheduling)
    python scheduler/weekend_runner.py --run-once-then-schedule  # Run now, then schedule weekly
"""

import argparse
import logging
import sys
import time

from output.dispatcher import run_pipeline
from config.settings import settings
from scheduler.schedule_utils import next_friday_run, seconds_until

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


def start_scheduler(run_now: bool = False) -> None:
    """Start the timezone-aware schedule loop (runs every Friday at configured time).

    Args:
        run_now: When *True*, run the pipeline immediately before entering the
                 schedule loop (``--run-once-then-schedule`` behaviour).
    """
    tz_name = settings.TIMEZONE
    run_time = settings.PREDICTION_RUN_TIME

    logger.info(
        "Scheduler started — predictions will run every Friday at %s %s.",
        run_time,
        tz_name,
    )

    if run_now:
        logger.info("--run-once-then-schedule flag detected — running pipeline immediately.")
        job()

    while True:
        next_run = next_friday_run(tz_name, run_time)
        secs = seconds_until(next_run)
        logger.info(
            "Next scheduled run: %s (%s) — sleeping for %.0f seconds.",
            next_run.strftime("%Y-%m-%d %H:%M"),
            tz_name,
            secs,
        )
        time.sleep(secs)
        job()
        # Brief pause to avoid re-triggering immediately if job runs fast
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ScoreBorga 2.5 Weekend Prediction Runner")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the prediction pipeline immediately without scheduling.",
    )
    parser.add_argument(
        "--run-once-then-schedule",
        action="store_true",
        help="Run the pipeline immediately and then enter the weekly schedule loop.",
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
        start_scheduler(run_now=args.run_once_then_schedule)


if __name__ == "__main__":
    main()
