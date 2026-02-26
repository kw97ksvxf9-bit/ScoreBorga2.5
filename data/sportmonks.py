"""
data/sportmonks.py
Sportmonks API v3 client for ScoreBorga 2.5.
Fetches fixtures, team statistics, and head-to-head records.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import pytz

from config.settings import settings

logger = logging.getLogger(__name__)


class SportmonksClient:
    """Client for the Sportmonks Football API v3."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SPORTMONKS_API_KEY
        self.base_url = settings.SPORTMONKS_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Perform a GET request and return the parsed JSON response."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.error("Sportmonks HTTP error %s – %s", exc.response.status_code, url)
            raise
        except requests.exceptions.RequestException as exc:
            logger.error("Sportmonks request failed: %s", exc)
            raise

    def _paginate(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch all pages for a paginated endpoint and return combined data."""
        params = params or {}
        results: List[Dict] = []
        page = 1
        while True:
            params["page"] = page
            data = self._get(endpoint, params)
            items = data.get("data", [])
            results.extend(items)
            pagination = data.get("pagination", {})
            if not pagination.get("has_more", False):
                break
            page += 1
        return results

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_weekend_fixtures(self, league_ids: Optional[List[int]] = None) -> List[Dict]:
        """
        Fetch upcoming fixtures for the weekend (Friday–Sunday) across the
        specified leagues (defaults to all Top 7 leagues).
        """
        league_ids = league_ids or settings.LEAGUE_IDS
        tz = pytz.timezone(settings.TIMEZONE)
        now = datetime.now(tz)

        # Find the upcoming Friday
        days_until_friday = (4 - now.weekday()) % 7
        friday = now + timedelta(days=days_until_friday)
        sunday = friday + timedelta(days=2)

        date_from = friday.strftime("%Y-%m-%d")
        date_to = sunday.strftime("%Y-%m-%d")

        logger.info("Fetching fixtures from %s to %s", date_from, date_to)

        league_ids_str = ";".join(str(lid) for lid in league_ids)
        fixtures = self._paginate(
            "fixtures/between/{}/{}".format(date_from, date_to),
            params={
                "filters": f"fixtureLeagues:{league_ids_str}",
                "include": "participants;scores;league",
            },
        )
        return fixtures

    def get_team_statistics(self, team_id: int, season_id: int) -> Dict:
        """
        Fetch statistics for a team in a specific season.
        Returns aggregated stats like goals scored/conceded and recent form.
        """
        data = self._get(
            f"teams/{team_id}",
            params={"include": "statistics.season;latestFixtures"},
        )
        return data.get("data", {})

    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """
        Fetch head-to-head records between two teams (last 10 encounters).
        """
        fixtures = self._paginate(
            f"fixtures/head-to-head/{team1_id}/{team2_id}",
            params={"include": "participants;scores", "per_page": 10},
        )
        return fixtures

    def get_recent_fixtures(self, team_id: int, count: int = 5) -> List[Dict]:
        """
        Fetch the most recent completed fixtures for a team.
        Used for calculating current form.
        """
        fixtures = self._paginate(
            f"teams/{team_id}/fixtures",
            params={
                "filters": "fixtureStatus:FT",
                "include": "participants;scores",
                "per_page": count,
                "sort": "-starting_at",
            },
        )
        return fixtures[:count]
