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

    # European League IDs (Sportmonks v3) – 27 leagues across top European competitions
    LEAGUE_IDS: list = [
        181,   # Admiral Bundesliga (Austria)
        208,   # Pro League (Belgium)
        244,   # 1. HNL (Croatia)
        271,   # Superliga (Denmark)
        8,     # Premier League (England)
        9,     # Championship (England)
        24,    # FA Cup (England)
        27,    # Carabao Cup (England)
        1371,  # UEFA Europa League Play-offs
        301,   # Ligue 1 (France)
        82,    # Bundesliga (Germany)
        384,   # Serie A (Italy)
        387,   # Serie B (Italy)
        390,   # Coppa Italia (Italy)
        72,    # Eredivisie (Netherlands)
        444,   # Eliteserien (Norway)
        453,   # Ekstraklasa (Poland)
        462,   # Liga Portugal (Portugal)
        486,   # Premier League (Russia)
        501,   # Premiership (Scotland)
        564,   # La Liga (Spain)
        567,   # La Liga 2 (Spain)
        570,   # Copa Del Rey (Spain)
        573,   # Allsvenskan (Sweden)
        591,   # Super League (Switzerland)
        600,   # Super Lig (Turkey)
        609,   # Premier League (Ukraine)
    ]


settings = Settings()
