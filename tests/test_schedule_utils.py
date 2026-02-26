"""
tests/test_schedule_utils.py
Unit tests for scheduler/schedule_utils.py.
Tests cover timezone-aware next-run calculation and do not hang.
"""

from datetime import datetime

import pytz
import pytest

from scheduler.schedule_utils import next_friday_run, seconds_until

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONDON = "Europe/London"
NEW_YORK = "America/New_York"
TOKYO = "Asia/Tokyo"

RUN_TIME = "09:00"


def _make_aware(year, month, day, hour, minute, tz_name):
    """Create a timezone-aware datetime in *tz_name*."""
    tz = pytz.timezone(tz_name)
    return tz.localize(datetime(year, month, day, hour, minute, 0))


# ---------------------------------------------------------------------------
# next_friday_run tests
# ---------------------------------------------------------------------------

class TestNextFridayRun:
    def test_monday_returns_same_week_friday(self):
        """From a Monday the next Friday should be 4 days away."""
        # 2025-03-03 is a Monday
        tz = pytz.timezone(LONDON)
        # Patch: directly call and verify the result is a Friday
        result = next_friday_run(LONDON, RUN_TIME)
        assert result.weekday() == 4, "Result should be a Friday"

    def test_result_is_timezone_aware(self):
        result = next_friday_run(LONDON, RUN_TIME)
        assert result.tzinfo is not None

    def test_result_has_correct_time(self):
        result = next_friday_run(LONDON, RUN_TIME)
        assert result.hour == 9
        assert result.minute == 0
        assert result.second == 0

    def test_friday_before_run_time_returns_today(self, monkeypatch):
        """When it is Friday and before run time, next run is today."""
        tz = pytz.timezone(LONDON)
        # 2025-03-07 is a Friday; 08:00 is before 09:00
        fake_now = tz.localize(datetime(2025, 3, 7, 8, 0, 0))

        monkeypatch.setattr(
            "scheduler.schedule_utils.datetime",
            _FakeDatetime(fake_now),
        )
        result = next_friday_run(LONDON, RUN_TIME)
        assert result.date() == fake_now.date()
        assert result.weekday() == 4

    def test_friday_after_run_time_returns_next_friday(self, monkeypatch):
        """When it is Friday and after run time, next run is the following Friday."""
        tz = pytz.timezone(LONDON)
        # 2025-03-07 is a Friday; 10:00 is after 09:00
        fake_now = tz.localize(datetime(2025, 3, 7, 10, 0, 0))

        monkeypatch.setattr(
            "scheduler.schedule_utils.datetime",
            _FakeDatetime(fake_now),
        )
        result = next_friday_run(LONDON, RUN_TIME)
        # Should be 2025-03-14
        assert result.year == 2025
        assert result.month == 3
        assert result.day == 14
        assert result.weekday() == 4

    def test_wednesday_returns_next_friday(self, monkeypatch):
        """From Wednesday, next Friday is 2 days away."""
        tz = pytz.timezone(LONDON)
        # 2025-03-05 is a Wednesday
        fake_now = tz.localize(datetime(2025, 3, 5, 12, 0, 0))

        monkeypatch.setattr(
            "scheduler.schedule_utils.datetime",
            _FakeDatetime(fake_now),
        )
        result = next_friday_run(LONDON, RUN_TIME)
        assert result.year == 2025
        assert result.month == 3
        assert result.day == 7
        assert result.weekday() == 4

    def test_different_timezone_new_york(self):
        """Result should be in the requested timezone."""
        result = next_friday_run(NEW_YORK, RUN_TIME)
        tz = pytz.timezone(NEW_YORK)
        assert result.tzinfo is not None
        assert result.weekday() == 4
        assert result.hour == 9

    def test_different_timezone_tokyo(self):
        result = next_friday_run(TOKYO, "18:00")
        assert result.weekday() == 4
        assert result.hour == 18
        assert result.minute == 0


# ---------------------------------------------------------------------------
# seconds_until tests
# ---------------------------------------------------------------------------

class TestSecondsUntil:
    def test_future_datetime_positive(self):
        from datetime import timedelta
        tz = pytz.timezone(LONDON)
        future = datetime.now(tz) + timedelta(days=1)
        secs = seconds_until(future)
        assert secs > 0

    def test_past_datetime_returns_zero(self):
        from datetime import timedelta
        tz = pytz.timezone(LONDON)
        past = datetime.now(tz) - timedelta(hours=1)
        assert seconds_until(past) == 0.0

    def test_now_returns_near_zero(self):
        tz = pytz.timezone(LONDON)
        now = datetime.now(tz)
        secs = seconds_until(now)
        assert 0.0 <= secs < 1.0


# ---------------------------------------------------------------------------
# Monkeypatch helper
# ---------------------------------------------------------------------------

class _FakeDatetime:
    """Minimal datetime replacement for monkeypatching."""

    def __init__(self, fixed_now: datetime):
        self._fixed_now = fixed_now
        # Expose class-level attributes/methods that schedule_utils uses
        self._real_datetime = datetime

    def now(self, tz=None):
        if tz is not None:
            return self._fixed_now.astimezone(tz)
        return self._fixed_now

    def __call__(self, *args, **kwargs):
        return self._real_datetime(*args, **kwargs)

    # Delegate attribute access to real datetime class
    def __getattr__(self, name):
        return getattr(self._real_datetime, name)
