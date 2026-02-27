"""
tests/test_sportmonks.py
Unit tests for data/sportmonks.py.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytz
import requests

from data.sportmonks import SportmonksClient
from config.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fixture(fixture_id: int, starting_at: str) -> dict:
    return {"id": fixture_id, "starting_at": starting_at, "scores": [{"description": "FT"}]}


def _client() -> SportmonksClient:
    """Return a client with a dummy API key (no real HTTP calls)."""
    return SportmonksClient(api_key="test_key")


# ---------------------------------------------------------------------------
# get_recent_fixtures tests
# ---------------------------------------------------------------------------

class TestGetRecentFixtures:
    def test_uses_between_endpoint(self):
        """Verify that the correct fixtures/between/… endpoint is called."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_recent_fixtures(team_id=42)
            endpoint = mock_paginate.call_args[0][0]
            assert endpoint.startswith("fixtures/between/")

    def test_endpoint_contains_date_range(self):
        """Endpoint path should embed today and the lookback date plus the team id."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_recent_fixtures(team_id=42)
            endpoint = mock_paginate.call_args[0][0]
            # fixtures/between/YYYY-MM-DD/YYYY-MM-DD/{team_id}
            parts = endpoint.split("/")
            assert len(parts) == 5
            assert parts[0] == "fixtures"
            assert parts[1] == "between"
            # Rough date-format check
            for date_part in parts[2:4]:
                datetime.strptime(date_part, "%Y-%m-%d")

    def test_team_id_in_endpoint_path(self):
        """The team ID must appear in the endpoint path; no invalid query filters."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_recent_fixtures(team_id=99)
            endpoint = mock_paginate.call_args[0][0]
            assert endpoint.endswith("/99")
            params = mock_paginate.call_args[1].get("params", {})
            filters = params.get("filters", "")
            assert "fixtureTeams" not in filters
            assert "fixtureStatus" not in filters

    def test_returns_empty_list_on_empty_response(self):
        """When API returns no fixtures the method should return []."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]):
            result = client.get_recent_fixtures(team_id=1)
        assert result == []

    def test_returns_empty_list_on_http_error(self):
        """HTTP errors should be caught and an empty list returned."""
        client = _client()
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch.object(
            client,
            "_paginate",
            side_effect=requests.exceptions.HTTPError(response=mock_response),
        ):
            result = client.get_recent_fixtures(team_id=1)
        assert result == []

    def test_returns_empty_list_on_request_exception(self):
        """Network errors should be caught and an empty list returned."""
        client = _client()
        with patch.object(
            client,
            "_paginate",
            side_effect=requests.exceptions.ConnectionError("timeout"),
        ):
            result = client.get_recent_fixtures(team_id=1)
        assert result == []

    def test_truncates_to_count(self):
        """Only the top `count` fixtures (by date desc) should be returned."""
        fixtures = [
            _make_fixture(i, f"2025-0{i}-01T15:00:00+00:00") for i in range(1, 9)
        ]
        client = _client()
        with patch.object(client, "_paginate", return_value=fixtures):
            result = client.get_recent_fixtures(team_id=1, count=3)
        assert len(result) == 3

    def test_sorted_descending_by_starting_at(self):
        """Results must be ordered newest-first."""
        fixtures = [
            _make_fixture(1, "2025-01-01T15:00:00+00:00"),
            _make_fixture(3, "2025-03-01T15:00:00+00:00"),
            _make_fixture(2, "2025-02-01T15:00:00+00:00"),
        ]
        client = _client()
        with patch.object(client, "_paginate", return_value=fixtures):
            result = client.get_recent_fixtures(team_id=1, count=5)
        dates = [f["starting_at"] for f in result]
        assert dates == sorted(dates, reverse=True)

    def test_most_recent_returned_when_truncated(self):
        """When truncating, the most recent fixtures must be kept."""
        fixtures = [
            _make_fixture(1, "2025-01-01T15:00:00+00:00"),
            _make_fixture(3, "2025-03-01T15:00:00+00:00"),
            _make_fixture(2, "2025-02-01T15:00:00+00:00"),
        ]
        client = _client()
        with patch.object(client, "_paginate", return_value=fixtures):
            result = client.get_recent_fixtures(team_id=1, count=2)
        ids = [f["id"] for f in result]
        assert ids == [3, 2]

    def test_include_params_preserved(self):
        """Downstream code needs participants;scores to be included."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_recent_fixtures(team_id=7)
            params = mock_paginate.call_args[1]["params"]
            assert "participants" in params["include"]
            assert "scores" in params["include"]

    def test_filters_to_completed_fixtures(self):
        """Only fixtures with FT/AET/AP score descriptions should be returned."""
        fixtures = [
            {"id": 1, "starting_at": "2025-01-01T15:00:00+00:00", "scores": [{"description": "FT"}]},
            {"id": 2, "starting_at": "2025-01-02T15:00:00+00:00", "scores": []},
            {"id": 3, "starting_at": "2025-01-03T15:00:00+00:00", "scores": [{"description": "CURRENT"}]},
            {"id": 4, "starting_at": "2025-01-04T15:00:00+00:00", "scores": [{"description": "AET"}]},
        ]
        client = _client()
        with patch.object(client, "_paginate", return_value=fixtures):
            result = client.get_recent_fixtures(team_id=1, count=10)
        ids = [f["id"] for f in result]
        assert 1 in ids   # FT
        assert 4 in ids   # AET
        assert 2 not in ids  # no scores
        assert 3 not in ids  # live (CURRENT)


# ---------------------------------------------------------------------------
# get_league tests
# ---------------------------------------------------------------------------

class TestGetLeague:
    def test_calls_leagues_endpoint(self):
        """Verify the correct leagues/{id} endpoint is called."""
        client = _client()
        with patch.object(client, "_get", return_value={"data": {}}) as mock_get:
            client.get_league(8)
            endpoint = mock_get.call_args[0][0]
            assert endpoint == "leagues/8"

    def test_returns_data_field(self):
        """The 'data' key from the API response should be returned."""
        client = _client()
        league_data = {"id": 8, "name": "Premier League", "country_id": 462}
        with patch.object(client, "_get", return_value={"data": league_data}):
            result = client.get_league(8)
        assert result == league_data

    def test_include_passed_as_param(self):
        """When include is given it must appear in the params dict."""
        client = _client()
        with patch.object(client, "_get", return_value={"data": {}}) as mock_get:
            client.get_league(8, include="seasons")
            params = mock_get.call_args[1]["params"]
            assert params.get("include") == "seasons"

    def test_no_include_omits_param(self):
        """When no include is given the params dict must not contain 'include'."""
        client = _client()
        with patch.object(client, "_get", return_value={"data": {}}) as mock_get:
            client.get_league(8)
            params = mock_get.call_args[1]["params"]
            assert "include" not in params

    def test_returns_empty_dict_on_missing_data(self):
        """Missing 'data' key in the response should yield an empty dict."""
        client = _client()
        with patch.object(client, "_get", return_value={}):
            result = client.get_league(8)
        assert result == {}


# ---------------------------------------------------------------------------
# get_fixtures_by_date_range tests
# ---------------------------------------------------------------------------

class TestGetFixturesByDateRange:
    def test_uses_between_endpoint(self):
        """Each per-league call must use fixtures/between/{date_from}/{date_to}."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31", league_ids=[8])
            endpoint = mock_paginate.call_args[0][0]
            assert endpoint == "fixtures/between/2025-01-01/2025-01-31"

    def test_filters_by_provided_league_ids(self):
        """Each league ID must be fetched in a separate _paginate call."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31", league_ids=[8, 82])
            assert mock_paginate.call_count == 2
            filters = [
                call[1]["params"]["filters"]
                for call in mock_paginate.call_args_list
            ]
            assert "fixtureLeagues:8" in filters
            assert "fixtureLeagues:82" in filters

    def test_uses_default_league_ids_when_none_given(self):
        """Default settings.LEAGUE_IDS must each get their own _paginate call."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31")
            assert mock_paginate.call_count == len(settings.LEAGUE_IDS)
            filters = [
                call[1]["params"]["filters"]
                for call in mock_paginate.call_args_list
            ]
            for lid in settings.LEAGUE_IDS:
                assert f"fixtureLeagues:{lid}" in filters

    def test_includes_participants_scores_league(self):
        """participants, scores, and league must be present in every include param."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31", league_ids=[8])
            params = mock_paginate.call_args[1]["params"]
            assert "participants" in params["include"]
            assert "scores" in params["include"]
            assert "league" in params["include"]

    def test_returns_fixtures_list(self):
        """Fixtures from all leagues must be merged into a single list."""
        fixtures_8 = [{"id": 1}]
        fixtures_82 = [{"id": 2}]
        client = _client()
        with patch.object(
            client, "_paginate", side_effect=[fixtures_8, fixtures_82]
        ):
            result = client.get_fixtures_by_date_range(
                "2025-01-01", "2025-01-31", league_ids=[8, 82]
            )
        assert result == [{"id": 1}, {"id": 2}]

    def test_returns_empty_list_when_no_fixtures(self):
        """Empty _paginate response should propagate as empty list."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]):
            result = client.get_fixtures_by_date_range(
                "2025-01-01", "2025-01-31", league_ids=[8]
            )
        assert result == []

    def test_skips_failed_league_and_returns_others(self):
        """A failing league call should be skipped; other leagues' fixtures returned."""
        client = _client()
        with patch.object(
            client,
            "_paginate",
            side_effect=[Exception("API error"), [{"id": 10}]],
        ):
            result = client.get_fixtures_by_date_range(
                "2025-01-01", "2025-01-31", league_ids=[8, 82]
            )
        assert result == [{"id": 10}]


# ---------------------------------------------------------------------------
# get_standings tests
# ---------------------------------------------------------------------------

class TestGetStandings:
    def test_calls_standings_endpoint(self):
        """Verify the correct 'standings' endpoint is called."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings()
            endpoint = mock_paginate.call_args[0][0]
            assert endpoint == "standings"

    def test_filters_by_provided_league_ids(self):
        """Explicit league IDs must appear in the standingLeagues filter."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings(league_ids=[8, 82])
            params = mock_paginate.call_args[1]["params"]
            assert "standingLeagues:8;82" in params["filters"]

    def test_uses_default_league_ids_when_none_given(self):
        """Default settings.LEAGUE_IDS must be applied when league_ids is omitted."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings()
            params = mock_paginate.call_args[1]["params"]
            for lid in settings.LEAGUE_IDS:
                assert str(lid) in params["filters"]

    def test_season_filter_added_when_provided(self):
        """When season_id is given it must appear as standingSeasons filter."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings(season_id=12345)
            params = mock_paginate.call_args[1]["params"]
            assert "standingSeasons:12345" in params["filters"]

    def test_season_filter_absent_when_not_provided(self):
        """When season_id is omitted, standingSeasons must not appear in filters."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings()
            params = mock_paginate.call_args[1]["params"]
            assert "standingSeasons" not in params["filters"]

    def test_include_passed_as_param(self):
        """When include is given it must appear in the params dict."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings(include="participant;rule")
            params = mock_paginate.call_args[1]["params"]
            assert params.get("include") == "participant;rule"

    def test_no_include_omits_param(self):
        """When no include is given params must not contain 'include'."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_standings()
            params = mock_paginate.call_args[1]["params"]
            assert "include" not in params

    def test_returns_standings_list(self):
        """Return value should be the list returned by _paginate."""
        standings = [{"id": 1, "position": 1}, {"id": 2, "position": 2}]
        client = _client()
        with patch.object(client, "_paginate", return_value=standings):
            result = client.get_standings()
        assert result == standings

    def test_returns_empty_list_when_no_data(self):
        """Empty _paginate response should propagate as empty list."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]):
            result = client.get_standings()
        assert result == []


# ---------------------------------------------------------------------------
# get_weekend_fixtures tests
# ---------------------------------------------------------------------------

def _make_aware(year, month, day, hour=12, tz_str="UTC"):
    """Create a timezone-aware datetime for the given date."""
    tz = pytz.timezone(tz_str)
    return tz.localize(datetime(year, month, day, hour, 0, 0))


class TestGetWeekendFixtures:
    """
    Tests for the Friday-anchor logic in get_weekend_fixtures.
    We freeze 'now' to a known weekday and verify the date range passed to
    get_fixtures_by_date_range is always the Friday–Sunday of that weekend.
    """

    def _run(self, fake_now: datetime):
        """Call get_weekend_fixtures with a frozen clock and return (date_from, date_to)."""
        client = _client()
        with patch("data.sportmonks.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            with patch.object(client, "get_fixtures_by_date_range", return_value=[]) as mock_range:
                client.get_weekend_fixtures(league_ids=[8])
                return mock_range.call_args[0][0], mock_range.call_args[0][1]

    def test_called_on_friday_uses_same_friday(self):
        """When called on a Friday, date_from must be that same Friday."""
        # 2025-01-03 is a Friday
        tz = pytz.timezone(settings.TIMEZONE)
        fake_now = tz.localize(datetime(2025, 1, 3, 12, 0, 0))
        date_from, date_to = self._run(fake_now)
        assert date_from == "2025-01-03"
        assert date_to == "2025-01-05"

    def test_called_on_saturday_uses_preceding_friday(self):
        """When called on a Saturday, date_from must be the preceding Friday."""
        # 2025-01-04 is a Saturday
        tz = pytz.timezone(settings.TIMEZONE)
        fake_now = tz.localize(datetime(2025, 1, 4, 12, 0, 0))
        date_from, date_to = self._run(fake_now)
        assert date_from == "2025-01-03"
        assert date_to == "2025-01-05"

    def test_called_on_sunday_uses_preceding_friday(self):
        """When called on a Sunday, date_from must be the Friday two days prior."""
        # 2025-01-05 is a Sunday
        tz = pytz.timezone(settings.TIMEZONE)
        fake_now = tz.localize(datetime(2025, 1, 5, 12, 0, 0))
        date_from, date_to = self._run(fake_now)
        assert date_from == "2025-01-03"
        assert date_to == "2025-01-05"

    def test_called_on_monday_uses_upcoming_friday(self):
        """When called on a Monday, date_from must be the upcoming Friday."""
        # 2025-01-06 is a Monday
        tz = pytz.timezone(settings.TIMEZONE)
        fake_now = tz.localize(datetime(2025, 1, 6, 12, 0, 0))
        date_from, date_to = self._run(fake_now)
        assert date_from == "2025-01-10"
        assert date_to == "2025-01-12"

    def test_called_on_thursday_uses_upcoming_friday(self):
        """When called on a Thursday, date_from must be the next day (Friday)."""
        # 2025-01-09 is a Thursday
        tz = pytz.timezone(settings.TIMEZONE)
        fake_now = tz.localize(datetime(2025, 1, 9, 12, 0, 0))
        date_from, date_to = self._run(fake_now)
        assert date_from == "2025-01-10"
        assert date_to == "2025-01-12"

    def test_date_to_is_always_sunday(self):
        """date_to must always be exactly 2 days after date_from (Sunday)."""
        tz = pytz.timezone(settings.TIMEZONE)
        fake_now = tz.localize(datetime(2025, 1, 7, 12, 0, 0))  # Tuesday
        date_from, date_to = self._run(fake_now)
        from_dt = datetime.strptime(date_from, "%Y-%m-%d")
        to_dt = datetime.strptime(date_to, "%Y-%m-%d")
        assert (to_dt - from_dt).days == 2
