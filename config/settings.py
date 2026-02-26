"""
config/settings.py
Loads environment variables and defines global configuration for ScoreBorga 2.5.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration object loaded from environment variables."""

    # API Keys
    SPORTMONKS_API_KEY: str = os.getenv("SPORTMONKS_API_KEY", "")
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # API Base URLs
    SPORTMONKS_BASE_URL: str = "https://api.sportmonks.com/v3/football"
    ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"

    # Timezone
    TIMEZONE: str = os.getenv("TIMEZONE", "Europe/London")

    # Scheduler
    PREDICTION_RUN_TIME: str = os.getenv("PREDICTION_RUN_TIME", "09:00")

    # Prediction Mode: "stat" (original), "ml" (machine learning only), "hybrid" (combined)
    PREDICTION_MODE: str = os.getenv("PREDICTION_MODE", "hybrid")

    # Number of historical seasons to use for ML training (default: 3)
    HISTORICAL_SEASONS: int = int(os.getenv("HISTORICAL_SEASONS", "3"))

    # Weight of ML prediction in hybrid mode (0.0-1.0, remainder goes to stat-based)
    ML_WEIGHT: float = float(os.getenv("ML_WEIGHT", "0.5"))

    # Lookback window (days) for fetching recent team fixtures used in form calculation
    RECENT_FIXTURES_LOOKBACK_DAYS: int = int(os.getenv("RECENT_FIXTURES_LOOKBACK_DAYS", "180"))

    # European League IDs (Sportmonks v3) – top 5 European leagues
    LEAGUE_IDS: list = [
        8,     # Premier League (England)
        301,   # Ligue 1 (France)
        82,    # Bundesliga (Germany)
        384,   # Serie A (Italy)
        564,   # La Liga (Spain)
    ]


settings = Settings()
