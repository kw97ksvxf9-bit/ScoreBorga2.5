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

    # Top 7 European League IDs (Sportmonks v3)
    LEAGUE_IDS: list = [
        271,   # Premier League (England)
        564,   # La Liga (Spain)
        82,    # Bundesliga (Germany)
        384,   # Serie A (Italy)
        301,   # Ligue 1 (France)
        72,    # Eredivisie (Netherlands)
        462,   # Primeira Liga (Portugal)
    ]


settings = Settings()
