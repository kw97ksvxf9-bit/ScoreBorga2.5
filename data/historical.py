"""
data/historical.py
Historical data fetching for ML model training in ScoreBorga 2.5.

Fetches past seasons' fixture data from Sportmonks API for training
machine learning models on historical match outcomes.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


def _extract_participants(fixture: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (home_participant, away_participant) from a fixture dict."""
    participants = fixture.get("participants", [])
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), None)
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), None)
    return home, away


def _extract_score(fixture: Dict, team_location: str) -> int:
    """Extract the final score for home or away team from a fixture.
    
    Checks multiple score descriptions in priority order:
    1. CURRENT - the live/final cumulative score
    2. FT - full-time score for completed fixtures
    3. AET - after extra time score
    4. AP - after penalties score
    """
    scores = fixture.get("scores", [])
    # Priority order: prefer CURRENT, then fall back to FT/AET/AP
    valid_descriptions = ("CURRENT", "FT", "AET", "AP")
    for desc in valid_descriptions:
        for score_entry in scores:
            entry_desc = (score_entry.get("description") or "").upper()
            if (
                entry_desc == desc
                and score_entry.get("score", {}).get("participant") == team_location
            ):
                return score_entry["score"].get("goals", 0)
    return 0


def _determine_outcome(home_goals: int, away_goals: int) -> int:
    """
    Determine match outcome.
    Returns: 0 = Home Win, 1 = Draw, 2 = Away Win
    """
    if home_goals > away_goals:
        return 0  # Home Win
    elif home_goals == away_goals:
        return 1  # Draw
    else:
        return 2  # Away Win


class HistoricalDataFetcher:
    """Fetches and processes historical match data for ML training."""

    def __init__(self, sportmonks_client: Any):
        """
        Initialize with a Sportmonks client instance.

        Args:
            sportmonks_client: Instance of SportmonksClient from data/sportmonks.py
        """
        self.client = sportmonks_client

    def get_seasons_for_league(self, league_id: int, num_seasons: int = 3) -> List[Dict]:
        """
        Fetch the most recent completed seasons for a league.

        Args:
            league_id: Sportmonks league ID
            num_seasons: Number of past seasons to fetch

        Returns:
            List of season dicts sorted by year descending
        """
        try:
            league = self.client.get_league(league_id, include="seasons")
            seasons = league.get("seasons", [])
            # Filter for completed seasons and sort by end date
            completed_seasons = [
                s for s in seasons
                if s.get("is_current") is False or s.get("finished") is True
            ]
            # Sort by starting date descending (most recent first)
            completed_seasons.sort(
                key=lambda s: s.get("starting_at", ""),
                reverse=True
            )
            return completed_seasons[:num_seasons]
        except Exception as exc:
            logger.error("Failed to fetch seasons for league %d: %s", league_id, exc)
            return []

    def get_season_fixtures(self, season_id: int) -> List[Dict]:
        """
        Fetch all completed fixtures for a season.

        Args:
            season_id: Sportmonks season ID

        Returns:
            List of completed fixture dicts
        """
        try:
            fixtures = self.client._paginate(
                f"fixtures",
                params={
                    "filters": f"fixtureSeasons:{season_id};fixtureStatus:FT",
                    "include": "participants;scores;statistics",
                    "per_page": 100,
                }
            )
            return fixtures
        except Exception as exc:
            logger.error("Failed to fetch fixtures for season %d: %s", season_id, exc)
            return []

    def get_team_season_stats(self, team_id: int, season_id: int) -> Dict:
        """
        Fetch aggregated statistics for a team in a specific season.

        Args:
            team_id: Sportmonks team ID
            season_id: Sportmonks season ID

        Returns:
            Dict with aggregated team statistics
        """
        try:
            data = self.client._get(
                f"teams/{team_id}",
                params={
                    "include": f"statistics.season:{season_id}",
                }
            )
            team_data = data.get("data", {})
            stats = team_data.get("statistics", [])
            # Find statistics for the requested season
            for stat_entry in stats:
                if stat_entry.get("season_id") == season_id:
                    return stat_entry.get("details", {})
            return {}
        except Exception as exc:
            logger.warning("Failed to fetch stats for team %d season %d: %s", team_id, season_id, exc)
            return {}

    def build_training_sample(self, fixture: Dict, home_form: Dict, away_form: Dict) -> Optional[Dict]:
        """
        Build a single training sample from a fixture and team forms.

        Args:
            fixture: Completed fixture dict from Sportmonks
            home_form: Home team's form statistics
            away_form: Away team's form statistics

        Returns:
            Dict with features and target outcome, or None if invalid
        """
        home_p, away_p = _extract_participants(fixture)
        if not home_p or not away_p:
            return None

        home_goals = _extract_score(fixture, "home")
        away_goals = _extract_score(fixture, "away")
        outcome = _determine_outcome(home_goals, away_goals)

        home_matches = max(
            home_form.get("wins", 0) + home_form.get("draws", 0) + home_form.get("losses", 0), 1
        )
        away_matches = max(
            away_form.get("wins", 0) + away_form.get("draws", 0) + away_form.get("losses", 0), 1
        )

        # Extract features from form data
        features = {
            # Home team features
            "home_wins": home_form.get("wins", 0),
            "home_draws": home_form.get("draws", 0),
            "home_losses": home_form.get("losses", 0),
            "home_goals_scored": home_form.get("goals_scored", 0),
            "home_goals_conceded": home_form.get("goals_conceded", 0),
            "home_points": home_form.get("points", 0),
            "home_avg_goals_scored": home_form.get("avg_goals_scored", 0.0),
            "home_avg_goals_conceded": home_form.get("avg_goals_conceded", 0.0),
            # Away team features
            "away_wins": away_form.get("wins", 0),
            "away_draws": away_form.get("draws", 0),
            "away_losses": away_form.get("losses", 0),
            "away_goals_scored": away_form.get("goals_scored", 0),
            "away_goals_conceded": away_form.get("goals_conceded", 0),
            "away_points": away_form.get("points", 0),
            "away_avg_goals_scored": away_form.get("avg_goals_scored", 0.0),
            "away_avg_goals_conceded": away_form.get("avg_goals_conceded", 0.0),
            # Derived features
            "home_form_score": home_form.get("points", 0) / max(home_matches * 3, 1),
            "away_form_score": away_form.get("points", 0) / max(away_matches * 3, 1),
            "goal_diff_home": home_form.get("goals_scored", 0) - home_form.get("goals_conceded", 0),
            "goal_diff_away": away_form.get("goals_scored", 0) - away_form.get("goals_conceded", 0),
            # Rate-based features
            "home_btts_rate": round(home_form.get("btts_count", 0) / home_matches, 4),
            "away_btts_rate": round(away_form.get("btts_count", 0) / away_matches, 4),
            "home_over_2_5_rate": round(home_form.get("over_2_5_count", 0) / home_matches, 4),
            "away_over_2_5_rate": round(away_form.get("over_2_5_count", 0) / away_matches, 4),
            "h2h_btts_rate": 0.0,
            "h2h_over_2_5_rate": 0.0,
            "home_clean_sheet_rate": round(home_form.get("clean_sheets", 0) / home_matches, 4),
            "away_clean_sheet_rate": round(away_form.get("clean_sheets", 0) / away_matches, 4),
        }

        return {
            "fixture_id": fixture.get("id"),
            "features": features,
            "outcome": outcome,
            "home_team": home_p.get("name"),
            "away_team": away_p.get("name"),
        }

    def fetch_historical_training_data(
        self,
        league_ids: Optional[List[int]] = None,
        num_seasons: int = 3,
    ) -> List[Dict]:
        """
        Fetch all historical training data for specified leagues.

        Args:
            league_ids: List of Sportmonks league IDs (defaults to LEAGUE_IDS from settings)
            num_seasons: Number of past seasons to fetch per league

        Returns:
            List of training samples with features and outcomes
        """
        league_ids = league_ids or settings.LEAGUE_IDS
        all_samples: List[Dict] = []

        for league_id in league_ids:
            logger.info("Fetching historical data for league %d...", league_id)
            seasons = self.get_seasons_for_league(league_id, num_seasons)

            for season in seasons:
                season_id = season.get("id")
                if not season_id:
                    continue

                logger.info("  Processing season %s (%s)", season.get("name"), season_id)
                fixtures = self.get_season_fixtures(season_id)

                # Build a cache of team forms from recent fixtures in this season
                team_forms: Dict[int, Dict] = {}

                for fixture in fixtures:
                    home_p, away_p = _extract_participants(fixture)
                    if not home_p or not away_p:
                        continue

                    home_id = home_p.get("id")
                    away_id = away_p.get("id")

                    # Use cached forms or default values
                    home_form = team_forms.get(home_id, {
                        "wins": 0, "draws": 0, "losses": 0,
                        "goals_scored": 0, "goals_conceded": 0, "points": 0,
                        "avg_goals_scored": 0.0, "avg_goals_conceded": 0.0,
                    })
                    away_form = team_forms.get(away_id, {
                        "wins": 0, "draws": 0, "losses": 0,
                        "goals_scored": 0, "goals_conceded": 0, "points": 0,
                        "avg_goals_scored": 0.0, "avg_goals_conceded": 0.0,
                    })

                    sample = self.build_training_sample(fixture, home_form, away_form)
                    if sample:
                        sample["league_id"] = league_id
                        sample["season_id"] = season_id
                        all_samples.append(sample)

                    # Update team forms after processing this fixture
                    self._update_team_form(team_forms, fixture, home_id, away_id)

        logger.info("Fetched %d historical training samples", len(all_samples))
        return all_samples

    def _update_team_form(
        self,
        team_forms: Dict[int, Dict],
        fixture: Dict,
        home_id: int,
        away_id: int,
    ) -> None:
        """Update team form dictionaries after a fixture result."""
        home_goals = _extract_score(fixture, "home")
        away_goals = _extract_score(fixture, "away")

        # Update home team form
        if home_id not in team_forms:
            team_forms[home_id] = {
                "wins": 0, "draws": 0, "losses": 0,
                "goals_scored": 0, "goals_conceded": 0, "points": 0,
                "matches": 0,
                "btts_count": 0, "over_2_5_count": 0, "clean_sheets": 0,
            }
        hf = team_forms[home_id]
        hf["goals_scored"] += home_goals
        hf["goals_conceded"] += away_goals
        hf["matches"] += 1
        if home_goals > away_goals:
            hf["wins"] += 1
            hf["points"] += 3
        elif home_goals == away_goals:
            hf["draws"] += 1
            hf["points"] += 1
        else:
            hf["losses"] += 1
        hf["avg_goals_scored"] = round(hf["goals_scored"] / hf["matches"], 2)
        hf["avg_goals_conceded"] = round(hf["goals_conceded"] / hf["matches"], 2)
        if home_goals >= 1 and away_goals >= 1:
            hf["btts_count"] += 1
        if home_goals + away_goals > 2:
            hf["over_2_5_count"] += 1
        if away_goals == 0:
            hf["clean_sheets"] += 1

        # Update away team form
        if away_id not in team_forms:
            team_forms[away_id] = {
                "wins": 0, "draws": 0, "losses": 0,
                "goals_scored": 0, "goals_conceded": 0, "points": 0,
                "matches": 0,
                "btts_count": 0, "over_2_5_count": 0, "clean_sheets": 0,
            }
        af = team_forms[away_id]
        af["goals_scored"] += away_goals
        af["goals_conceded"] += home_goals
        af["matches"] += 1
        if away_goals > home_goals:
            af["wins"] += 1
            af["points"] += 3
        elif away_goals == home_goals:
            af["draws"] += 1
            af["points"] += 1
        else:
            af["losses"] += 1
        af["avg_goals_scored"] = round(af["goals_scored"] / af["matches"], 2)
        af["avg_goals_conceded"] = round(af["goals_conceded"] / af["matches"], 2)
        if home_goals >= 1 and away_goals >= 1:
            af["btts_count"] += 1
        if home_goals + away_goals > 2:
            af["over_2_5_count"] += 1
        if home_goals == 0:
            af["clean_sheets"] += 1
