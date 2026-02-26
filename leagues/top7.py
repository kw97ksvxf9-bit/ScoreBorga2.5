"""
leagues/top7.py
Definitions for the Top 7 European football leagues supported by ScoreBorga 2.5.
"""

from typing import List, Dict

# League definitions: name, country, Sportmonks v3 league ID, Odds API sport key
TOP7_LEAGUES: List[Dict] = [
    {
        "id": 271,
        "name": "Premier League",
        "country": "England",
        "odds_key": "soccer_epl",
    },
    {
        "id": 564,
        "name": "La Liga",
        "country": "Spain",
        "odds_key": "soccer_spain_la_liga",
    },
    {
        "id": 82,
        "name": "Bundesliga",
        "country": "Germany",
        "odds_key": "soccer_germany_bundesliga",
    },
    {
        "id": 384,
        "name": "Serie A",
        "country": "Italy",
        "odds_key": "soccer_italy_serie_a",
    },
    {
        "id": 301,
        "name": "Ligue 1",
        "country": "France",
        "odds_key": "soccer_france_ligue_one",
    },
    {
        "id": 72,
        "name": "Eredivisie",
        "country": "Netherlands",
        "odds_key": "soccer_netherlands_eredivisie",
    },
    {
        "id": 462,
        "name": "Primeira Liga",
        "country": "Portugal",
        "odds_key": "soccer_portugal_primeira_liga",
    },
]

# Convenience lookup by Sportmonks league ID
LEAGUE_BY_ID: Dict[int, Dict] = {league["id"]: league for league in TOP7_LEAGUES}

# Convenience lookup by Odds API sport key
LEAGUE_BY_ODDS_KEY: Dict[str, Dict] = {
    league["odds_key"]: league for league in TOP7_LEAGUES
}
