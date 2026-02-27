"""
engine/polisher.py
Transforms raw prediction dicts into polished, human-readable Telegram messages
using Markdown formatting.
"""

import logging
from typing import Dict, List, Optional

import pytz
from datetime import datetime

from config.settings import settings
from leagues.top7 import LEAGUE_BY_ID

logger = logging.getLogger(__name__)

# Emoji mappings
OUTCOME_EMOJI = {
    "Home Win": "🏠",
    "Draw": "🤝",
    "Away Win": "✈️",
}

CONFIDENCE_EMOJI = {
    (80, 101): "🔥",
    (60, 80): "✅",
    (40, 60): "⚠️",
    (0, 40): "❓",
}


def _confidence_emoji(confidence: float) -> str:
    for (low, high), emoji in CONFIDENCE_EMOJI.items():
        if low <= confidence < high:
            return emoji
    return "❓"


def _format_kickoff(kickoff_str: Optional[str]) -> str:
    """Format a Sportmonks ISO kickoff string to a readable local time."""
    if not kickoff_str:
        return "TBD"
    try:
        utc_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        tz = pytz.timezone(settings.TIMEZONE)
        local_dt = utc_dt.astimezone(tz)
        return local_dt.strftime("%a %d %b %Y %H:%M %Z")
    except Exception:
        return kickoff_str


def polish_prediction(prediction: Dict) -> str:
    """
    Convert a single prediction dict into a Telegram Markdown-formatted string.

    Args:
        prediction: Prediction dict from engine/predictor.py.

    Returns:
        Formatted string ready to be sent via Telegram.
    """
    home = prediction.get("home_team", "Home")
    away = prediction.get("away_team", "Away")
    kickoff = _format_kickoff(prediction.get("kickoff"))
    outcome = prediction.get("prediction", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    reasoning = prediction.get("reasoning", "")
    odds = prediction.get("odds", {})
    league_id = prediction.get("league_id")

    league_info = LEAGUE_BY_ID.get(league_id, {})
    league_name = league_info.get("name", "Unknown League")
    country = league_info.get("country", "")

    outcome_emoji = OUTCOME_EMOJI.get(outcome, "⚽")
    conf_emoji = _confidence_emoji(confidence)

    odds_line = ""
    if odds:
        odds_line = (
            f"📊 *Odds:* H `{odds.get('home', '-')}` "
            f"/ D `{odds.get('draw', '-')}` "
            f"/ A `{odds.get('away', '-')}`\n"
        )

    best_total = prediction.get("best_total", {})
    totals = prediction.get("totals", {})
    totals_line = ""
    if best_total:
        totals_line = (
            f"⚽ *Best Total:* {best_total.get('line', '')} "
            f"({best_total.get('probability', 0):.1f}%)\n"
        )
        if totals:
            totals_line += (
                f"📈 *Totals:* O1.5 {totals.get('over_1_5', 0):.1f}% "
                f"| O2.5 {totals.get('over_2_5', 0):.1f}% "
                f"| O3.5 {totals.get('over_3_5', 0):.1f}%\n"
            )

    message = (
        f"⚽ *{home}* vs *{away}*\n"
        f"🏆 {league_name} ({country})\n"
        f"🕒 {kickoff}\n"
        f"\n"
        f"{outcome_emoji} *Prediction:* {outcome}\n"
        f"{conf_emoji} *Confidence:* {confidence}%\n"
        f"{odds_line}"
        f"{totals_line}"
        f"\n"
        f"📝 _{reasoning}_\n"
        f"{'─' * 30}"
    )
    return message


def polish_all(predictions: List[Dict]) -> str:
    """
    Polish all predictions and combine them into a single Telegram message.

    Args:
        predictions: List of prediction dicts.

    Returns:
        Full formatted message string.
    """
    if not predictions:
        return "No predictions available for this weekend. 😔"

    header = (
        "🔮 *ScoreBorga 2.5 — Weekend Predictions*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    footer = (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "_Predictions are for entertainment purposes only. "
        "Please gamble responsibly._ 🙏"
    )

    polished_parts = [polish_prediction(p) for p in predictions]
    return header + "\n".join(polished_parts) + footer
