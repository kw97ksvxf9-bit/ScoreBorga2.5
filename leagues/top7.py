"""
leagues/top7.py
Definitions for the European football leagues supported by ScoreBorga 2.5.
Note: `TOP7_LEAGUES` is kept for backward compatibility; `SUPPORTED_LEAGUES` is the canonical alias.
"""

from typing import List, Dict

# League definitions: name, country, Sportmonks v3 league ID, Odds API sport key
TOP7_LEAGUES: List[Dict] = [
    # Austria
    {
        "id": 181,
        "name": "Admiral Bundesliga",
        "country": "Austria",
        "odds_key": "soccer_austria_bundesliga",
    },
    # Belgium
    {
        "id": 208,
        "name": "Pro League",
        "country": "Belgium",
        "odds_key": "soccer_belgium_first_div",
    },
    # Croatia
    {
        "id": 244,
        "name": "1. HNL",
        "country": "Croatia",
        "odds_key": "soccer_croatia_hnl",
    },
    # Denmark
    {
        "id": 271,
        "name": "Superliga",
        "country": "Denmark",
        "odds_key": "soccer_denmark_superliga",
    },
    # England
    {
        "id": 8,
        "name": "Premier League",
        "country": "England",
        "odds_key": "soccer_epl",
    },
    {
        "id": 9,
        "name": "Championship",
        "country": "England",
        "odds_key": "soccer_england_championship",
    },
    {
        "id": 24,
        "name": "FA Cup",
        "country": "England",
        "odds_key": "soccer_fa_cup",
    },
    {
        "id": 27,
        "name": "Carabao Cup",
        "country": "England",
        "odds_key": "soccer_league_cup",
    },
    # Europe
    {
        "id": 1371,
        "name": "UEFA Europa League Play-offs",
        "country": "Europe",
        "odds_key": "soccer_uefa_europa_league",
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
    {
        "id": 387,
        "name": "Serie B",
        "country": "Italy",
        "odds_key": "soccer_italy_serie_b",
    },
    {
        "id": 390,
        "name": "Coppa Italia",
        "country": "Italy",
        "odds_key": "soccer_italy_coppa_italia",
    },
    # Netherlands
    {
        "id": 72,
        "name": "Eredivisie",
        "country": "Netherlands",
        "odds_key": "soccer_netherlands_eredivisie",
    },
    # Norway
    {
        "id": 444,
        "name": "Eliteserien",
        "country": "Norway",
        "odds_key": "soccer_norway_eliteserien",
    },
    # Poland
    {
        "id": 453,
        "name": "Ekstraklasa",
        "country": "Poland",
        "odds_key": "soccer_poland_ekstraklasa",
    },
    # Portugal
    {
        "id": 462,
        "name": "Liga Portugal",
        "country": "Portugal",
        "odds_key": "soccer_portugal_primeira_liga",
    },
    # Russia
    {
        "id": 486,
        "name": "Premier League",
        "country": "Russia",
        "odds_key": "soccer_russia_premier_league",
    },
    # Scotland
    {
        "id": 501,
        "name": "Premiership",
        "country": "Scotland",
        "odds_key": "soccer_scotland_premiership",
    },
    # Spain
    {
        "id": 564,
        "name": "La Liga",
        "country": "Spain",
        "odds_key": "soccer_spain_la_liga",
    },
    {
        "id": 567,
        "name": "La Liga 2",
        "country": "Spain",
        "odds_key": "soccer_spain_segunda_division",
    },
    {
        "id": 570,
        "name": "Copa Del Rey",
        "country": "Spain",
        "odds_key": "soccer_copa_del_rey",
    },
    # Sweden
    {
        "id": 573,
        "name": "Allsvenskan",
        "country": "Sweden",
        "odds_key": "soccer_sweden_allsvenskan",
    },
    # Switzerland
    {
        "id": 591,
        "name": "Super League",
        "country": "Switzerland",
        "odds_key": "soccer_switzerland_superleague",
    },
    # Turkey
    {
        "id": 600,
        "name": "Super Lig",
        "country": "Turkey",
        "odds_key": "soccer_turkey_super_league",
    },
    # Ukraine
    {
        "id": 609,
        "name": "Premier League",
        "country": "Ukraine",
        "odds_key": "soccer_ukraine_premier_league",
    },
]

# Canonical alias reflecting the full 27-league coverage
SUPPORTED_LEAGUES = TOP7_LEAGUES

# Convenience lookup by Sportmonks league ID
LEAGUE_BY_ID: Dict[int, Dict] = {league["id"]: league for league in TOP7_LEAGUES}

# Convenience lookup by Odds API sport key
LEAGUE_BY_ODDS_KEY: Dict[str, Dict] = {
    league["odds_key"]: league for league in TOP7_LEAGUES
}
