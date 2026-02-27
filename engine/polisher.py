"""
engine/polisher.py
Formats ScoreBorga predictions into the official Telegram post layout.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pytz

from config.settings import settings
from leagues.top7 import LEAGUE_BY_ID

logger = logging.getLogger(__name__)

# Number emojis for positions 1-10
_NUM_EMOJI = {
    1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣",
    6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟",
}

OUTCOME_EMOJI = {
    "Home Win": "🏠",
    "Draw": "🤝",
    "Away Win": "✈️",
}


def _confidence_rocket(confidence: float) -> str:
    if confidence >= 70:
        return "🚀"
    elif confidence >= 50:
        return "💪"
    return "📊"


def _format_date(kickoff_str: Optional[str]) -> str:
    """Format kickoff ISO string to 'Saturday, 08 November 2025'."""
    if not kickoff_str:
        return "TBD"
    try:
        utc_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        tz = pytz.timezone(settings.TIMEZONE)
        local_dt = utc_dt.astimezone(tz)
        return local_dt.strftime("%A, %d %B %Y")
    except Exception:
        return kickoff_str


def _rank_label(position: int) -> str:
    return _NUM_EMOJI.get(position, f"{position}.")


def _format_extra_picks(prediction: Dict) -> str:
    """Build the 💡 Extra Picks block."""
    lines = ["💡 Extra Picks:"]

    # BTTS
    btts = prediction.get("btts", False)
    lines.append(f" BTTS {'✅' if btts else '❌'}")

    # Over 1.5
    over_1_5 = prediction.get("over_1_5", False)
    lines.append(f" Over 1.5 ⚽️{'✅' if over_1_5 else '❌'}")

    # Over 2.5 — only show if predicted yes
    over_2_5 = prediction.get("over_2_5", False)
    if over_2_5:
        lines.append(" Over 2.5 ⚽️✅")

    # Over 3.5 — only show if predicted yes (no tick/cross, just the line)
    over_3_5 = prediction.get("over_3_5", False)
    if over_3_5:
        lines.append(" Over 3.5 ⚽️")

    return "\n".join(lines)


def polish_prediction(prediction: Dict, position: int) -> str:
    """Format a single prediction dict into its Telegram block."""
    home = prediction.get("home_team", "Home")
    away = prediction.get("away_team", "Away")
    outcome = prediction.get("prediction", "Unknown")
    confidence = prediction.get("confidence", 0.0)

    outcome_emoji = OUTCOME_EMOJI.get(outcome, "⚽")
    rocket = _confidence_rocket(confidence)
    rank = _rank_label(position)
    extra = _format_extra_picks(prediction)

    return (
        f"{rank} {home} 🆚 {away}\n"
        f"👉 Prediction: {outcome} {outcome_emoji}\n"
        f"{extra}\n"
        f"📈 Confidence: {confidence}% {rocket}"
    )


def polish_all(predictions: List[Dict], top_n: int = 15) -> str:
    """
    Build the complete Telegram message from a list of prediction dicts.

    - Sorts predictions by confidence descending
    - Limits to top_n (default 15)
    - Derives the date header from the earliest kickoff
    """
    if not predictions:
        return "No predictions available for this weekend. 😔"

    # Sort by confidence descending
    sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
    top_preds = sorted_preds[:top_n]

    # Derive date from earliest kickoff
    kickoffs = [p.get("kickoff") for p in top_preds if p.get("kickoff")]
    if kickoffs:
        earliest = min(kickoffs)
        date_str = _format_date(earliest)
    else:
        date_str = "TBD"

    header = (
        f"📢 Football Predictions ⚽️🔥\n"
        f"📅 Date: {date_str}\n"
        f"📊 Top {len(top_preds)} Picks Ranked by Confidence\n"
    )

    body_parts = []
    for i, pred in enumerate(top_preds, start=1):
        body_parts.append(polish_prediction(pred, i))

    return header + "\n" + "\n\n".join(body_parts)

