"""
engine/predictor.py
Core prediction logic for ScoreBorga 2.5.

Supports three prediction modes:
  - "stat": Uses a weighted scoring model combining:
      - Recent form (last 5 matches)
      - Head-to-head record
      - Home advantage
      - Implied probability from odds
  - "ml": Uses machine learning model trained on historical data
  - "hybrid": Combines both stat-based and ML predictions for sharper results
"""

import logging
import math
from typing import Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Outcome labels
HOME_WIN = "Home Win"
DRAW = "Draw"
AWAY_WIN = "Away Win"

# Model weights for stat-based prediction (must sum to 1.0)
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


def _poisson_over_prob(lam: float, line: float) -> float:
    """
    Compute P(goals > line) using a Poisson approximation.

    Args:
        lam: Expected total goals (lambda for Poisson distribution).
        line: The over/under line (e.g. 1.5, 2.5, 3.5).

    Returns:
        Probability as a float in [0, 1].
    """
    k_max = int(math.floor(line))  # highest integer <= line, so P(X > line) = P(X >= k_max+1)
    prob_under = 0.0
    for k in range(0, k_max + 1):
        prob_under += math.exp(-lam) * (lam ** k) / math.factorial(k)
    return max(0.0, min(1.0, 1.0 - prob_under))


def _totals_prediction(analytics: Dict) -> Dict:
    """
    Compute heuristic totals probabilities for Over 1.5 / 2.5 / 3.5.

    Uses a simple expected-goals approach: combine home attack vs away defence
    and away attack vs home defence to estimate expected total goals, then
    apply a Poisson-based approximation.

    Args:
        analytics: Structured analytics dict from engine/analytics.py.

    Returns:
        Dict with keys:
            totals: { over_1_5, over_2_5, over_3_5 } as percentages (float 0-100)
            best_total: { line: str, probability: float }
    """
    home_form = analytics.get("home_form", {})
    away_form = analytics.get("away_form", {})

    home_avg_scored = home_form.get("avg_goals_scored", 1.2)
    home_avg_conceded = home_form.get("avg_goals_conceded", 1.0)
    away_avg_scored = away_form.get("avg_goals_scored", 1.0)
    away_avg_conceded = away_form.get("avg_goals_conceded", 1.2)

    # Expected goals: average of team attack and opponent defence
    home_xg = (home_avg_scored + away_avg_conceded) / 2.0
    away_xg = (away_avg_scored + home_avg_conceded) / 2.0
    expected_total = max(0.1, home_xg + away_xg)

    over_1_5 = round(_poisson_over_prob(expected_total, 1.5) * 100, 1)
    over_2_5 = round(_poisson_over_prob(expected_total, 2.5) * 100, 1)
    over_3_5 = round(_poisson_over_prob(expected_total, 3.5) * 100, 1)

    # Select the best total: the line with the highest probability
    candidates = [
        ("Over 1.5", over_1_5),
        ("Over 2.5", over_2_5),
        ("Over 3.5", over_3_5),
    ]
    # Pick the highest-probability line
    best_line, best_prob = max(candidates, key=lambda x: x[1])

    return {
        "totals": {
            "over_1_5": over_1_5,
            "over_2_5": over_2_5,
            "over_3_5": over_3_5,
        },
        "best_total": {
            "line": best_line,
            "probability": best_prob,
        },
    }


def _stat_based_prediction(analytics: Dict) -> Dict:
    """
    Generate a prediction using the stat-based weighted scoring model.

    Args:
        analytics: Structured analytics dict from engine/analytics.py.

    Returns:
        Dict with prediction, confidence, and scores.
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

    return {
        "prediction": prediction,
        "confidence": confidence,
        "scores": scores,
    }


def _hybrid_prediction(analytics: Dict) -> Dict:
    """
    Generate a prediction combining stat-based and ML models.

    Args:
        analytics: Structured analytics dict from engine/analytics.py.

    Returns:
        Dict with prediction, confidence, scores, and hybrid details.
    """
    # Import here to avoid circular imports
    from engine.ml_model import get_ml_predictor

    # Get stat-based prediction
    stat_pred = _stat_based_prediction(analytics)
    stat_scores = stat_pred["scores"]

    # Get ML prediction
    ml_predictor = get_ml_predictor()
    ml_pred = ml_predictor.predict(analytics)
    ml_probs = ml_pred["probabilities"]

    # Combine predictions using configured weight
    ml_weight = settings.ML_WEIGHT
    stat_weight = 1.0 - ml_weight

    # Normalize stat scores to probabilities
    stat_total = sum(stat_scores.values()) or 1.0
    stat_probs = {
        "home": stat_scores["home"] / stat_total,
        "draw": stat_scores["draw"] / stat_total,
        "away": stat_scores["away"] / stat_total,
    }

    # Combined probabilities (hybrid)
    hybrid_probs = {
        "home": stat_weight * stat_probs["home"] + ml_weight * ml_probs["home"],
        "draw": stat_weight * stat_probs["draw"] + ml_weight * ml_probs["draw"],
        "away": stat_weight * stat_probs["away"] + ml_weight * ml_probs["away"],
    }

    # Determine final prediction
    max_prob = max(hybrid_probs.values())
    if max_prob == hybrid_probs["home"]:
        prediction = HOME_WIN
    elif max_prob == hybrid_probs["draw"]:
        prediction = DRAW
    else:
        prediction = AWAY_WIN

    confidence = round(max_prob * 100, 1)

    # Convert hybrid probs back to scores format
    hybrid_scores = {
        "home": round(hybrid_probs["home"], 4),
        "draw": round(hybrid_probs["draw"], 4),
        "away": round(hybrid_probs["away"], 4),
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "scores": hybrid_scores,
        "stat_prediction": stat_pred["prediction"],
        "ml_prediction": ml_pred["prediction"],
        "ml_confidence": ml_pred["confidence"],
    }


def predict_fixture(analytics: Dict, mode: Optional[str] = None) -> Dict:
    """
    Generate a prediction for a single fixture.

    Args:
        analytics: Structured analytics dict from engine/analytics.py.
        mode: Prediction mode ("stat", "ml", or "hybrid"). Defaults to settings.PREDICTION_MODE.

    Returns:
        Dict with keys:
            fixture_id, home_team, away_team, kickoff,
            prediction (str), confidence (float 0–100),
            reasoning (str), scores (dict with home/draw/away raw scores).
    """
    mode = mode or settings.PREDICTION_MODE

    if mode == "ml":
        # Import here to avoid circular imports
        from engine.ml_model import predict_with_ml
        ml_result = predict_with_ml(analytics)
        result = {
            "prediction": ml_result["prediction"],
            "confidence": ml_result["confidence"],
            "scores": {
                "home": round(ml_result["probabilities"]["home"], 4),
                "draw": round(ml_result["probabilities"]["draw"], 4),
                "away": round(ml_result["probabilities"]["away"], 4),
            },
        }
    elif mode == "hybrid":
        result = _hybrid_prediction(analytics)
    else:  # "stat" or default
        result = _stat_based_prediction(analytics)

    # Build reasoning string
    home_team = analytics.get("home_team", "Home")
    away_team = analytics.get("away_team", "Away")
    home_form = analytics.get("home_form", {})
    away_form = analytics.get("away_form", {})
    h2h = analytics.get("h2h", {})
    odds = analytics.get("odds", {})

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

    # Add mode indicator to reasoning
    if mode == "hybrid":
        reasoning_parts.append(f"Mode: Hybrid (ML+Stats)")
        if "ml_prediction" in result:
            reasoning_parts.append(f"ML: {result['ml_prediction']} ({result['ml_confidence']}%)")
    elif mode == "ml":
        reasoning_parts.append("Mode: ML Model")
    else:
        reasoning_parts.append("Mode: Statistics")

    return {
        "fixture_id": analytics.get("fixture_id"),
        "home_team": home_team,
        "away_team": away_team,
        "kickoff": analytics.get("kickoff"),
        "league_id": analytics.get("league_id"),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "reasoning": " | ".join(reasoning_parts),
        "scores": result["scores"],
        "odds": odds,
        **_totals_prediction(analytics),
    }


def predict_all(analytics_list: list, mode: Optional[str] = None) -> list:
    """
    Run predictions for a list of fixture analytics dicts.

    Args:
        analytics_list: List of dicts from engine/analytics.py.
        mode: Prediction mode ("stat", "ml", or "hybrid"). Defaults to settings.PREDICTION_MODE.

    Returns:
        List of prediction dicts.
    """
    predictions = []
    for analytics in analytics_list:
        try:
            pred = predict_fixture(analytics, mode=mode)
            predictions.append(pred)
        except Exception as exc:
            logger.error(
                "Failed to predict fixture %s: %s",
                analytics.get("fixture_id"),
                exc,
            )
    return predictions
