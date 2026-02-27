"""
engine/markets.py
Side-market predictions: BTTS, Over 1.5, Over 2.5, Over 3.5.
Uses Poisson distribution to compute probabilities from expected goals.
"""
import math
from typing import Dict

from config.settings import settings


def _poisson_prob_at_least(lam: float, k: int) -> float:
    """P(X >= k) for Poisson(lam)."""
    prob_less = sum(
        (lam ** i) * math.exp(-lam) / math.factorial(i)
        for i in range(k)
    )
    return max(0.0, min(1.0, 1.0 - prob_less))


def _poisson_prob_score(home_lam: float, away_lam: float, home_g: int, away_g: int) -> float:
    """P(home scores exactly home_g AND away scores exactly away_g)."""
    def p(lam, k):
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    return p(home_lam, home_g) * p(away_lam, away_g)


def compute_market_predictions(analytics: Dict) -> Dict:
    """
    Compute side-market probabilities for a fixture.

    Uses avg_goals_scored and avg_goals_conceded from home_form and away_form
    to derive expected goals (lambda) for each team via Dixon-Coles-style
    attack/defence estimation.

    Returns a dict with keys:
        btts_prob       float [0,1]
        over_1_5_prob   float [0,1]
        over_2_5_prob   float [0,1]
        over_3_5_prob   float [0,1]
        btts            bool  (True if btts_prob >= settings.BTTS_THRESHOLD)
        over_1_5        bool
        over_2_5        bool
        over_3_5        bool
        home_xg         float  expected goals for home team
        away_xg         float  expected goals for away team
    """
    home_form = analytics.get("home_form", {})
    away_form = analytics.get("away_form", {})

    # Expected goals: home team attack vs away team defence, and vice-versa
    home_attack  = home_form.get("avg_goals_scored", 1.3)
    home_defence = home_form.get("avg_goals_conceded", 1.1)
    away_attack  = away_form.get("avg_goals_scored", 1.1)
    away_defence = away_form.get("avg_goals_conceded", 1.3)

    # Apply home advantage multiplier
    home_adv_factor = analytics.get("home_adv_factor", 1.1)

    # Blend attack vs opposing defence to get expected goals
    home_xg = ((home_attack + away_defence) / 2.0) * home_adv_factor
    away_xg = (away_attack + home_defence) / 2.0

    # Clamp to [0.3, 4.0]
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.3, min(4.0, away_xg))

    # Over N.5 = P(total goals >= N+1) — compute via complement of Poisson sum
    # We approximate total goals as Poisson(home_xg + away_xg)
    total_lam = home_xg + away_xg

    over_1_5_prob = _poisson_prob_at_least(total_lam, 2)
    over_2_5_prob = _poisson_prob_at_least(total_lam, 3)
    over_3_5_prob = _poisson_prob_at_least(total_lam, 4)

    # BTTS = P(home >= 1) * P(away >= 1)
    btts_prob = _poisson_prob_at_least(home_xg, 1) * _poisson_prob_at_least(away_xg, 1)

    return {
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "btts_prob": round(btts_prob, 4),
        "over_1_5_prob": round(over_1_5_prob, 4),
        "over_2_5_prob": round(over_2_5_prob, 4),
        "over_3_5_prob": round(over_3_5_prob, 4),
        "btts": btts_prob >= settings.BTTS_THRESHOLD,
        "over_1_5": over_1_5_prob >= settings.OVER_1_5_THRESHOLD,
        "over_2_5": over_2_5_prob >= settings.OVER_2_5_THRESHOLD,
        "over_3_5": over_3_5_prob >= settings.OVER_3_5_THRESHOLD,
    }
