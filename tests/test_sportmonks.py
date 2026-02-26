"""
tests/test_sportmonks.py
Unit tests for data/sportmonks.py – focusing on get_recent_fixtures.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

from data.sportmonks import SportmonksClient


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
