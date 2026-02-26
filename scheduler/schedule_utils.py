"""
scheduler/schedule_utils.py
Timezone-aware scheduling utilities for ScoreBorga 2.5.
"""

from datetime import datetime, timedelta

import pytz


def next_friday_run(timezone: str, run_time: str) -> datetime:
    """Return the next Friday at ``run_time`` (``HH:MM``) in *timezone*.

    If today is Friday and *run_time* has not yet passed, returns a datetime
    for today at *run_time*.  Otherwise returns the coming Friday at *run_time*.
    The returned datetime is timezone-aware.
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    hour, minute = (int(p) for p in run_time.split(":"))

    # Friday == weekday 4
    days_until_friday = (4 - now.weekday()) % 7

    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if days_until_friday == 0 and now >= candidate:
        # Already past run time on a Friday — wait for next Friday
        days_until_friday = 7

    return candidate + timedelta(days=days_until_friday)


def seconds_until(dt: datetime) -> float:
    """Return the number of seconds until *dt* (never negative)."""
    now = datetime.now(dt.tzinfo)
    return max(0.0, (dt - now).total_seconds())
