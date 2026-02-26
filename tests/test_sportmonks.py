"""
tests/test_sportmonks.py
Unit tests for data/sportmonks.py.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

from data.sportmonks import SportmonksClient
from config.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fixture(fixture_id: int, starting_at: str) -> dict:
    return {"id": fixture_id, "starting_at": starting_at}


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
        """Endpoint path should embed today and the lookback date."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_recent_fixtures(team_id=42)
            endpoint = mock_paginate.call_args[0][0]
            # fixtures/between/YYYY-MM-DD/YYYY-MM-DD
            parts = endpoint.split("/")
            assert len(parts) == 4
            assert parts[0] == "fixtures"
            assert parts[1] == "between"
            # Rough date-format check
            for date_part in parts[2:]:
                datetime.strptime(date_part, "%Y-%m-%d")

    def test_filter_includes_team_and_ft_status(self):
        """The API filters param must include the team id and fixtureStatus:FT."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_recent_fixtures(team_id=99)
            params = mock_paginate.call_args[1]["params"]
            assert "fixtureTeams:99" in params["filters"]
            assert "fixtureStatus:FT" in params["filters"]

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
        """Endpoint must be fixtures/between/{date_from}/{date_to}."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31")
            endpoint = mock_paginate.call_args[0][0]
            assert endpoint == "fixtures/between/2025-01-01/2025-01-31"

    def test_filters_by_provided_league_ids(self):
        """Explicit league IDs must appear in the fixtureLeagues filter."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31", league_ids=[8, 82])
            params = mock_paginate.call_args[1]["params"]
            assert "fixtureLeagues:8;82" in params["filters"]

    def test_uses_default_league_ids_when_none_given(self):
        """Default settings.LEAGUE_IDS must be applied when league_ids is omitted."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31")
            params = mock_paginate.call_args[1]["params"]
            for lid in settings.LEAGUE_IDS:
                assert str(lid) in params["filters"]

    def test_includes_participants_scores_league(self):
        """participants, scores, and league must be present in the include param."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]) as mock_paginate:
            client.get_fixtures_by_date_range("2025-01-01", "2025-01-31")
            params = mock_paginate.call_args[1]["params"]
            assert "participants" in params["include"]
            assert "scores" in params["include"]
            assert "league" in params["include"]

    def test_returns_fixtures_list(self):
        """Return value should be the list returned by _paginate."""
        fixtures = [{"id": 1}, {"id": 2}]
        client = _client()
        with patch.object(client, "_paginate", return_value=fixtures):
            result = client.get_fixtures_by_date_range("2025-01-01", "2025-01-31")
        assert result == fixtures

    def test_returns_empty_list_when_no_fixtures(self):
        """Empty _paginate response should propagate as empty list."""
        client = _client()
        with patch.object(client, "_paginate", return_value=[]):
            result = client.get_fixtures_by_date_range("2025-01-01", "2025-01-31")
        assert result == []


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
