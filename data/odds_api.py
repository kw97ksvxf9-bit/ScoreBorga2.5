"""
data/odds_api.py
Odds API client for ScoreBorga 2.5.
Fetches current 1X2 match odds for upcoming fixtures.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OddsApiClient:
    """Client for the-odds-api.com v4."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ODDS_API_KEY
        self.base_url = settings.ODDS_API_BASE_URL
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Perform a GET request and return parsed JSON."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.error("Odds API HTTP error %s – %s", exc.response.status_code, url)
            raise
        except requests.exceptions.RequestException as exc:
            logger.error("Odds API request failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_odds_for_sport(self, sport_key: str) -> List[Dict]:
        """
        Fetch h2h (1X2) odds for all upcoming events in the given sport.

        Args:
            sport_key: Odds API sport key (e.g. "soccer_epl").

        Returns:
            List of event dicts each containing bookmaker odds.
        """
        data = self._get(
            f"sports/{sport_key}/odds",
            params={"regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
        )
        return data if isinstance(data, list) else []

    def get_odds_for_all_top7(self, leagues: Optional[List[Dict]] = None) -> Dict[str, List[Dict]]:
        """
        Fetch 1X2 odds for all Top 7 European leagues.

        Args:
            leagues: List of league dicts (from leagues/top7.py).
                     Defaults to importing TOP7_LEAGUES automatically.

        Returns:
            Dict mapping odds_key → list of event odds dicts.
        """
        if leagues is None:
            from leagues.top7 import TOP7_LEAGUES
            leagues = TOP7_LEAGUES

        all_odds: Dict[str, List[Dict]] = {}
        for league in leagues:
            sport_key = league["odds_key"]
            try:
                all_odds[sport_key] = self.get_odds_for_sport(sport_key)
                logger.info("Fetched odds for %s (%d events)", league["name"], len(all_odds[sport_key]))
            except Exception as exc:
                logger.warning("Could not fetch odds for %s: %s", league["name"], exc)
                all_odds[sport_key] = []
        return all_odds

    def map_odds_to_fixture(
        self,
        odds_events: List[Dict],
        home_team: str,
        away_team: str,
    ) -> Optional[Dict[str, float]]:
        """
        Find and return 1X2 decimal odds for a specific fixture by matching
        team names against the Odds API event list.

        Returns:
            Dict with keys 'home', 'draw', 'away' containing best available
            decimal odds, or None if not found.
        """
        home_lower = home_team.lower()
        away_lower = away_team.lower()

        for event in odds_events:
            event_home = event.get("home_team", "").lower()
            event_away = event.get("away_team", "").lower()
            if home_lower in event_home or event_home in home_lower:
                if away_lower in event_away or event_away in away_lower:
                    return self._extract_best_h2h_odds(event)
        return None

    @staticmethod
    def _extract_best_h2h_odds(event: Dict) -> Dict[str, float]:
        """Extract the best available 1X2 decimal odds from an event dict."""
        best = {"home": 1.0, "draw": 1.0, "away": 1.0}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = float(outcome.get("price", 1.0))
                    if name == event.get("home_team"):
                        best["home"] = max(best["home"], price)
                    elif name == event.get("away_team"):
                        best["away"] = max(best["away"], price)
                    elif name.lower() == "draw":
                        best["draw"] = max(best["draw"], price)
        return best
