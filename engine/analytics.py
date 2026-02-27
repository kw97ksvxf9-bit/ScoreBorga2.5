"""
engine/analytics.py
Processes raw fixture/team data from Sportmonks and Odds API into structured
analytics per fixture that can be fed into the predictor.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Result codes used internally
WIN = "W"
DRAW = "D"
LOSS = "L"


def _extract_participants(fixture: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (home_participant, away_participant) from a fixture dict."""
    participants = fixture.get("participants", [])
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), None)
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), None)
    return home, away


def _extract_score(fixture: Dict, team_location: str) -> int:
    """Extract the final score for home or away team from a fixture."""
    scores = fixture.get("scores", [])
    for score_entry in scores:
        if (
            score_entry.get("description") == "CURRENT"
            and score_entry.get("score", {}).get("participant") == team_location
        ):
            return score_entry["score"].get("goals", 0)
    return 0


def _compute_btts_rate(fixtures: List[Dict], team_id: int) -> float:
    """Fraction of team's recent fixtures where both teams scored."""
    if not fixtures:
        return 0.0
    count = 0
    valid = 0
    for fixture in fixtures:
        home_p, away_p = _extract_participants(fixture)
        if home_p is None or away_p is None:
            continue
        is_home = home_p.get("id") == team_id
        my_location = "home" if is_home else "away"
        opp_location = "away" if is_home else "home"
        my_goals = _extract_score(fixture, my_location)
        opp_goals = _extract_score(fixture, opp_location)
        valid += 1
        if my_goals >= 1 and opp_goals >= 1:
            count += 1
    return round(count / valid, 4) if valid else 0.0


def _compute_over_2_5_rate(fixtures: List[Dict]) -> float:
    """Fraction of fixtures with total goals > 2."""
    if not fixtures:
        return 0.0
    count = 0
    valid = 0
    for fixture in fixtures:
        home_p, away_p = _extract_participants(fixture)
        if home_p is None or away_p is None:
            continue
        home_goals = _extract_score(fixture, "home")
        away_goals = _extract_score(fixture, "away")
        valid += 1
        if home_goals + away_goals > 2:
            count += 1
    return round(count / valid, 4) if valid else 0.0


def _compute_clean_sheet_rate(fixtures: List[Dict], team_id: int) -> float:
    """Fraction of fixtures where the team kept a clean sheet."""
    if not fixtures:
        return 0.0
    count = 0
    valid = 0
    for fixture in fixtures:
        home_p, away_p = _extract_participants(fixture)
        if home_p is None or away_p is None:
            continue
        is_home = home_p.get("id") == team_id
        opp_location = "away" if is_home else "home"
        opp_goals = _extract_score(fixture, opp_location)
        valid += 1
        if opp_goals == 0:
            count += 1
    return round(count / valid, 4) if valid else 0.0


def calculate_form(recent_fixtures: List[Dict], team_id: int) -> Dict:
    """
    Calculate form for a team from its last N completed fixtures.

    Args:
        recent_fixtures: List of fixture dicts (most recent first).
        team_id: The Sportmonks team ID to calculate form for.

    Returns:
        Dict with keys: wins, draws, losses, form_string (e.g. "WDLWW"),
        goals_scored, goals_conceded, points.
    """
    wins = draws = losses = goals_scored = goals_conceded = 0
    form_chars: List[str] = []

    for fixture in recent_fixtures:
        home_p, away_p = _extract_participants(fixture)
        if home_p is None or away_p is None:
            continue

        is_home = home_p.get("id") == team_id
        my_location = "home" if is_home else "away"
        opp_location = "away" if is_home else "home"

        my_goals = _extract_score(fixture, my_location)
        opp_goals = _extract_score(fixture, opp_location)
        goals_scored += my_goals
        goals_conceded += opp_goals

        if my_goals > opp_goals:
            wins += 1
            form_chars.append(WIN)
        elif my_goals == opp_goals:
            draws += 1
            form_chars.append(DRAW)
        else:
            losses += 1
            form_chars.append(LOSS)

    n = len(recent_fixtures) or 1
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "form_string": "".join(form_chars),
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "avg_goals_scored": round(goals_scored / n, 2),
        "avg_goals_conceded": round(goals_conceded / n, 2),
        "points": wins * 3 + draws,
    }


def calculate_h2h_stats(h2h_fixtures: List[Dict], team1_id: int, team2_id: int) -> Dict:
    """
    Compute head-to-head win/draw/loss rates between two teams.

    Returns:
        Dict with team1_wins, team2_wins, draws, total, team1_win_rate.
    """
    team1_wins = team2_wins = draws = 0

    for fixture in h2h_fixtures:
        home_p, away_p = _extract_participants(fixture)
        if home_p is None or away_p is None:
            continue

        home_id = home_p.get("id")
        home_goals = _extract_score(fixture, "home")
        away_goals = _extract_score(fixture, "away")

        if home_goals > away_goals:
            winner_id = home_id
        elif away_goals > home_goals:
            winner_id = away_p.get("id")
        else:
            winner_id = None

        if winner_id == team1_id:
            team1_wins += 1
        elif winner_id == team2_id:
            team2_wins += 1
        else:
            draws += 1

    total = team1_wins + team2_wins + draws or 1
    return {
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "draws": draws,
        "total": total,
        "team1_win_rate": round(team1_wins / total, 3),
        "team2_win_rate": round(team2_wins / total, 3),
        "draw_rate": round(draws / total, 3),
    }


def build_fixture_analytics(
    fixture: Dict,
    home_recent: List[Dict],
    away_recent: List[Dict],
    h2h_fixtures: List[Dict],
    odds: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Aggregate all analytics for a single fixture.

    Args:
        fixture: Raw fixture dict from Sportmonks.
        home_recent: Recent fixtures for the home team.
        away_recent: Recent fixtures for the away team.
        h2h_fixtures: Head-to-head fixtures between the two teams.
        odds: Optional dict with 'home', 'draw', 'away' decimal odds.

    Returns:
        Structured analytics dict ready for the predictor.
    """
    home_p, away_p = _extract_participants(fixture)
    home_id = home_p.get("id") if home_p else None
    away_id = away_p.get("id") if away_p else None
    home_name = home_p.get("name", "Home") if home_p else "Home"
    away_name = away_p.get("name", "Away") if away_p else "Away"

    home_form = calculate_form(home_recent, home_id) if home_id else {}
    away_form = calculate_form(away_recent, away_id) if away_id else {}
    h2h_stats = calculate_h2h_stats(h2h_fixtures, home_id, away_id) if (home_id and away_id) else {}

    # Compute rate-based features from recent fixtures
    home_btts_rate = _compute_btts_rate(home_recent, home_id) if home_id else 0.0
    away_btts_rate = _compute_btts_rate(away_recent, away_id) if away_id else 0.0
    home_over_2_5_rate = _compute_over_2_5_rate(home_recent)
    away_over_2_5_rate = _compute_over_2_5_rate(away_recent)
    home_clean_sheet_rate = _compute_clean_sheet_rate(home_recent, home_id) if home_id else 0.0
    away_clean_sheet_rate = _compute_clean_sheet_rate(away_recent, away_id) if away_id else 0.0
    h2h_over_2_5_rate = _compute_over_2_5_rate(h2h_fixtures)

    # Compute proper h2h BTTS rate using both team ids
    if home_id and away_id and h2h_fixtures:
        h2h_btts_count = sum(
            1 for f in h2h_fixtures
            if _extract_score(f, "home") >= 1 and _extract_score(f, "away") >= 1
        )
        h2h_btts_rate = round(h2h_btts_count / len(h2h_fixtures), 4)
    else:
        h2h_btts_rate = 0.0

    return {
        "fixture_id": fixture.get("id"),
        "home_team": home_name,
        "away_team": away_name,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "kickoff": fixture.get("starting_at"),
        "league_id": fixture.get("league_id"),
        "home_form": home_form,
        "away_form": away_form,
        "h2h": h2h_stats,
        "odds": odds or {},
        "home_adv_factor": 1.1,
        "home_btts_rate": home_btts_rate,
        "away_btts_rate": away_btts_rate,
        "home_over_2_5_rate": home_over_2_5_rate,
        "away_over_2_5_rate": away_over_2_5_rate,
        "home_clean_sheet_rate": home_clean_sheet_rate,
        "away_clean_sheet_rate": away_clean_sheet_rate,
        "h2h_btts_rate": h2h_btts_rate,
        "h2h_over_2_5_rate": h2h_over_2_5_rate,
    }

