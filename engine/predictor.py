"""
engine/predictor.py
Core prediction logic for ScoreBorga 2.5.

Uses a weighted scoring model combining:
  - Recent form (last 5 matches)
  - Head-to-head record
  - Home advantage
  - Implied probability from odds
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Outcome labels
HOME_WIN = "Home Win"
DRAW = "Draw"
AWAY_WIN = "Away Win"

# Model weights (must sum to 1.0)
WEIGHT_FORM = 0.35
WEIGHT_H2H = 0.20
WEIGHT_HOME_ADV = 0.15
WEIGHT_ODDS = 0.30


def _form_score(form: Dict) -> float:
    """
    Convert form stats into a score in [0, 1].
    Based on points per game out of a maximum of 3 points per game.
    """
    if not form:
        return 0.5  # neutral
    n = form.get("wins", 0) + form.get("draws", 0) + form.get("losses", 0) or 1
    ppg = form.get("points", 0) / n
    return round(ppg / 3.0, 4)


def _implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        return 0.0
    return round(1.0 / odds, 4)


def predict_fixture(analytics: Dict) -> Dict:
    """
    Generate a prediction for a single fixture.

    Args:
        analytics: Structured analytics dict from engine/analytics.py.

    Returns:
        Dict with keys:
            fixture_id, home_team, away_team, kickoff,
            prediction (str), confidence (float 0–100),
            reasoning (str), scores (dict with home/draw/away raw scores).
    """
    home_form = analytics.get("home_form", {})
    away_form = analytics.get("away_form", {})
    h2h = analytics.get("h2h", {})
    odds = analytics.get("odds", {})

    # --- Form component ---
    home_form_score = _form_score(home_form)
    away_form_score = _form_score(away_form)

    # --- H2H component ---
    home_h2h = h2h.get("team1_win_rate", 0.33)  # team1 = home
    away_h2h = h2h.get("team2_win_rate", 0.33)
    draw_h2h = h2h.get("draw_rate", 0.34)

    # --- Home advantage component (static bonus) ---
    home_adv = 0.6   # home team gets 60% of home-advantage score
    away_adv = 0.4

    # --- Odds component (implied probability, normalised) ---
    home_odds_prob = _implied_prob(odds.get("home", 2.5))
    draw_odds_prob = _implied_prob(odds.get("draw", 3.2))
    away_odds_prob = _implied_prob(odds.get("away", 3.0))
    total_odds_prob = home_odds_prob + draw_odds_prob + away_odds_prob or 1.0
    home_odds_norm = home_odds_prob / total_odds_prob
    draw_odds_norm = draw_odds_prob / total_odds_prob
    away_odds_norm = away_odds_prob / total_odds_prob

    # --- Weighted composite scores ---
    home_score = (
        WEIGHT_FORM * home_form_score
        + WEIGHT_H2H * home_h2h
        + WEIGHT_HOME_ADV * home_adv
        + WEIGHT_ODDS * home_odds_norm
    )
    draw_score = (
        WEIGHT_FORM * 0.5 * (1 - abs(home_form_score - away_form_score))
        + WEIGHT_H2H * draw_h2h
        + WEIGHT_HOME_ADV * 0.0
        + WEIGHT_ODDS * draw_odds_norm
    )
    away_score = (
        WEIGHT_FORM * away_form_score
        + WEIGHT_H2H * away_h2h
        + WEIGHT_HOME_ADV * away_adv
        + WEIGHT_ODDS * away_odds_norm
    )

    scores = {"home": round(home_score, 4), "draw": round(draw_score, 4), "away": round(away_score, 4)}

    # --- Determine prediction and confidence ---
    max_score = max(scores.values())
    total_score = sum(scores.values()) or 1.0

    if max_score == scores["home"]:
        prediction = HOME_WIN
        confidence = round((scores["home"] / total_score) * 100, 1)
    elif max_score == scores["draw"]:
        prediction = DRAW
        confidence = round((scores["draw"] / total_score) * 100, 1)
    else:
        prediction = AWAY_WIN
        confidence = round((scores["away"] / total_score) * 100, 1)

    # --- Build reasoning string ---
    home_team = analytics.get("home_team", "Home")
    away_team = analytics.get("away_team", "Away")
    home_form_str = home_form.get("form_string", "N/A")
    away_form_str = away_form.get("form_string", "N/A")
    h2h_total = h2h.get("total", 0)

    reasoning_parts = [
        f"{home_team} form: {home_form_str}",
        f"{away_team} form: {away_form_str}",
    ]
    if h2h_total > 0:
        reasoning_parts.append(
            f"H2H ({h2h_total} games): {home_team} {h2h.get('team1_wins', 0)}W "
            f"– {h2h.get('draws', 0)}D – {h2h.get('team2_wins', 0)}W {away_team}"
        )
    if odds:
        reasoning_parts.append(
            f"Odds: H {odds.get('home', '-')} / D {odds.get('draw', '-')} / A {odds.get('away', '-')}"
        )

    return {
        "fixture_id": analytics.get("fixture_id"),
        "home_team": home_team,
        "away_team": away_team,
        "kickoff": analytics.get("kickoff"),
        "league_id": analytics.get("league_id"),
        "prediction": prediction,
        "confidence": confidence,
        "reasoning": " | ".join(reasoning_parts),
        "scores": scores,
        "odds": odds,
    }


def predict_all(analytics_list: list) -> list:
    """
    Run predictions for a list of fixture analytics dicts.

    Args:
        analytics_list: List of dicts from engine/analytics.py.

    Returns:
        List of prediction dicts.
    """
    predictions = []
    for analytics in analytics_list:
        try:
            pred = predict_fixture(analytics)
            predictions.append(pred)
        except Exception as exc:
            logger.error(
                "Failed to predict fixture %s: %s",
                analytics.get("fixture_id"),
                exc,
            )
    return predictions
