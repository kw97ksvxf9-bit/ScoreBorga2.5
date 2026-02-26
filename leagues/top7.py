"""
leagues/top7.py
Definitions for the top 5 European football leagues supported by ScoreBorga 2.5.
Note: `TOP7_LEAGUES` is kept for backward compatibility; `SUPPORTED_LEAGUES` is the canonical alias.
"""

from typing import List, Dict

# League definitions: name, country, Sportmonks v3 league ID, Odds API sport key
TOP7_LEAGUES: List[Dict] = [
    # England
    {
        "id": 8,
        "name": "Premier League",
        "country": "England",
        "odds_key": "soccer_epl",
    },
    # France
    {
        "id": 301,
        "name": "Ligue 1",
        "country": "France",
        "odds_key": "soccer_france_ligue_one",
    },
    # Germany
    {
        "id": 82,
        "name": "Bundesliga",
        "country": "Germany",
        "odds_key": "soccer_germany_bundesliga",
    },
    # Italy
    {
        "id": 384,
        "name": "Serie A",
        "country": "Italy",
        "odds_key": "soccer_italy_serie_a",
    },
    # Spain
    {
        "id": 564,
        "name": "La Liga",
        "country": "Spain",
        "odds_key": "soccer_spain_la_liga",
    },
]

# Canonical alias for the top 5 European leagues
SUPPORTED_LEAGUES = TOP7_LEAGUES

# Convenience lookup by Sportmonks league ID
LEAGUE_BY_ID: Dict[int, Dict] = {league["id"]: league for league in TOP7_LEAGUES}

# Convenience lookup by Odds API sport key
LEAGUE_BY_ODDS_KEY: Dict[str, Dict] = {
    league["odds_key"]: league for league in TOP7_LEAGUES
}
