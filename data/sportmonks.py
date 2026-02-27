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

# Score descriptions that indicate a fixture has finished.
_COMPLETED_SCORE_DESCRIPTIONS = {"FT", "AET", "AP"}


def _is_completed(fixture: dict) -> bool:
    """Return True if the fixture appears to have finished (best-effort check)."""
    for score in fixture.get("scores", []):
        desc = (score.get("description") or "").upper()
        if desc in _COMPLETED_SCORE_DESCRIPTIONS:
            return True
    return False


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

    def get_league(self, league_id: int, include: Optional[str] = None) -> Dict:
        """
        Fetch a single league by its Sportmonks ID.
        Returns the league data dict (the 'data' key from the API response).

        Args:
            league_id: Sportmonks league ID.
            include: Optional comma/semicolon-separated include string
                     (e.g. "seasons", "currentSeason;stages").
        """
        params: Dict = {}
        if include:
            params["include"] = include
        data = self._get(f"leagues/{league_id}", params=params)
        return data.get("data", {})

    def get_fixtures_by_date_range(
        self,
        date_from: str,
        date_to: str,
        league_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Fetch fixtures within a date range for the specified leagues.
        Fetches each league individually to avoid the multi-ID fixtureLeagues
        filter being silently ignored on lower-tier Sportmonks plans.

        Args:
            date_from: Start date in YYYY-MM-DD format.
            date_to: End date in YYYY-MM-DD format.
            league_ids: Sportmonks league IDs to filter by (defaults to settings.LEAGUE_IDS).
        """
        league_ids = league_ids or settings.LEAGUE_IDS
        all_fixtures: List[Dict] = []

        for league_id in league_ids:
            logger.info("Fetching fixtures for league_id=%d (%s to %s)", league_id, date_from, date_to)
            try:
                fixtures = self._paginate(
                    f"fixtures/between/{date_from}/{date_to}",
                    params={
                        "filters": f"fixtureLeagues:{league_id}",
                        "include": "participants;scores;league",
                    },
                )
                logger.info("  → %d fixtures found for league_id=%d", len(fixtures), league_id)
                all_fixtures.extend(fixtures)
            except Exception as exc:
                logger.warning("Failed to fetch fixtures for league_id=%d: %s", league_id, exc)

        return all_fixtures

    def get_standings(
        self,
        league_ids: Optional[List[int]] = None,
        season_id: Optional[int] = None,
        include: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch standings for the specified leagues, optionally filtered by season.
        Uses the GET All Standings endpoint with standingLeagues/standingSeasons filters.

        Args:
            league_ids: Sportmonks league IDs to filter by (defaults to settings.LEAGUE_IDS).
            season_id: Optional Sportmonks season ID to restrict standings to one season.
            include: Optional include string (e.g. "participant;rule;details").
        """
        league_ids = league_ids or settings.LEAGUE_IDS
        league_ids_str = ";".join(str(lid) for lid in league_ids)
        filters = f"standingLeagues:{league_ids_str}"
        if season_id is not None:
            filters += f";standingSeasons:{season_id}"
        params: Dict = {"filters": filters}
        if include:
            params["include"] = include
        return self._paginate("standings", params=params)

    def get_weekend_fixtures(self, league_ids: Optional[List[int]] = None) -> List[Dict]:
        """
        Fetch fixtures for the current/upcoming weekend (Friday–Sunday).
        Works correctly regardless of what day of the week it is called.
        """
        tz = pytz.timezone(settings.TIMEZONE)
        now = datetime.now(tz)
        weekday = now.weekday()  # Mon=0, Tue=1, ..., Fri=4, Sat=5, Sun=6

        # If already in a weekend (Fri/Sat/Sun), anchor to this Friday.
        # Otherwise, jump forward to the next Friday.
        if weekday == 4:    # Friday
            days_to_friday = 0
        elif weekday == 5:  # Saturday
            days_to_friday = -1
        elif weekday == 6:  # Sunday
            days_to_friday = -2
        else:               # Mon–Thu: upcoming Friday
            days_to_friday = 4 - weekday

        friday = now + timedelta(days=days_to_friday)
        sunday = friday + timedelta(days=2)

        date_from = friday.strftime("%Y-%m-%d")
        date_to = sunday.strftime("%Y-%m-%d")

        logger.info("Fetching weekend fixtures from %s to %s", date_from, date_to)
        return self.get_fixtures_by_date_range(date_from, date_to, league_ids)

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
        tz = pytz.timezone(settings.TIMEZONE)
        now = datetime.now(tz)
        lookback = getattr(settings, "RECENT_FIXTURES_LOOKBACK_DAYS", 180)
        date_from = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
        date_to = now.strftime("%Y-%m-%d")

        try:
            fixtures = self._paginate(
                f"fixtures/between/date/{date_from}/{date_to}/{team_id}",
                params={"include": "participants;scores"},
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Could not fetch recent fixtures for team %s: %s", team_id, exc)
            return []

        completed = [f for f in fixtures if _is_completed(f)]
        if not completed:
            logger.warning("No completed fixtures found for team %s", team_id)
        else:
            logger.debug("Found %d completed fixtures for team %s", len(completed), team_id)

        completed.sort(key=lambda f: f.get("starting_at", ""), reverse=True)
        return completed[:count]
