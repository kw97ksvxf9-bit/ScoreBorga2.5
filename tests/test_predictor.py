"""
tests/test_predictor.py
Unit tests for engine/predictor.py and engine/analytics.py.
"""

import pytest
from leagues.top7 import TOP7_LEAGUES, LEAGUE_BY_ID
from config.settings import settings
from engine.predictor import (
    predict_fixture,
    predict_all,
    _form_score,
    _implied_prob,
    HOME_WIN,
    DRAW,
    AWAY_WIN,
)
from engine.analytics import (
    calculate_form,
    calculate_h2h_stats,
    build_fixture_analytics,
)


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

def _make_participant(team_id: int, name: str, location: str) -> dict:
    return {"id": team_id, "name": name, "meta": {"location": location}}


def _make_fixture(fixture_id: int, home_id: int, away_id: int, home_goals: int, away_goals: int) -> dict:
    return {
        "id": fixture_id,
        "league_id": 8,
        "starting_at": "2025-03-01T15:00:00+00:00",
        "participants": [
            _make_participant(home_id, "Home FC", "home"),
            _make_participant(away_id, "Away FC", "away"),
        ],
        "scores": [
            {"description": "CURRENT", "score": {"participant": "home", "goals": home_goals}},
            {"description": "CURRENT", "score": {"participant": "away", "goals": away_goals}},
        ],
    }


SAMPLE_HOME_RECENT = [
    _make_fixture(101, 1, 2, 3, 0),  # W
    _make_fixture(102, 1, 3, 1, 1),  # D
    _make_fixture(103, 4, 1, 0, 2),  # W (team 1 away)
    _make_fixture(104, 1, 5, 2, 3),  # L
    _make_fixture(105, 1, 6, 1, 0),  # W
]

SAMPLE_AWAY_RECENT = [
    _make_fixture(201, 7, 10, 0, 1),  # W
    _make_fixture(202, 10, 8, 2, 0),  # W (team 10 home)
    _make_fixture(203, 10, 9, 1, 1),  # D
    _make_fixture(204, 10, 11, 0, 2), # L
    _make_fixture(205, 12, 10, 3, 1), # L (team 10 away)
]

SAMPLE_H2H = [
    _make_fixture(301, 1, 10, 2, 1),   # home win
    _make_fixture(302, 10, 1, 1, 1),   # draw
    _make_fixture(303, 1, 10, 0, 1),   # away (team 10) win
    _make_fixture(304, 1, 10, 3, 0),   # home win
]

SAMPLE_ODDS = {"home": 1.8, "draw": 3.5, "away": 4.2}


# ---------------------------------------------------------------------------
# analytics tests
# ---------------------------------------------------------------------------

class TestCalculateForm:
    def test_all_wins(self):
        fixtures = [_make_fixture(i, 1, i + 1, 2, 0) for i in range(1, 6)]
        form = calculate_form(fixtures, team_id=1)
        assert form["wins"] == 5
        assert form["draws"] == 0
        assert form["losses"] == 0
        assert form["form_string"] == "WWWWW"
        assert form["points"] == 15

    def test_all_losses(self):
        fixtures = [_make_fixture(i, 1, i + 1, 0, 2) for i in range(1, 6)]
        form = calculate_form(fixtures, team_id=1)
        assert form["wins"] == 0
        assert form["losses"] == 5
        assert form["form_string"] == "LLLLL"
        assert form["points"] == 0

    def test_mixed_form(self):
        form = calculate_form(SAMPLE_HOME_RECENT, team_id=1)
        # fixtures: W, D, W(away), L, W
        assert form["wins"] == 3
        assert form["draws"] == 1
        assert form["losses"] == 1

    def test_empty_fixtures(self):
        form = calculate_form([], team_id=1)
        assert form["form_string"] == ""
        assert form["points"] == 0

    def test_avg_goals(self):
        fixtures = [_make_fixture(i, 1, i + 1, 2, 1) for i in range(1, 4)]
        form = calculate_form(fixtures, team_id=1)
        assert form["avg_goals_scored"] == 2.0
        assert form["avg_goals_conceded"] == 1.0


class TestCalculateH2H:
    def test_basic_h2h(self):
        stats = calculate_h2h_stats(SAMPLE_H2H, team1_id=1, team2_id=10)
        assert stats["team1_wins"] == 2
        assert stats["team2_wins"] == 1
        assert stats["draws"] == 1
        assert stats["total"] == 4
        assert stats["team1_win_rate"] == pytest.approx(0.5)

    def test_empty_h2h(self):
        stats = calculate_h2h_stats([], team1_id=1, team2_id=10)
        assert stats["team1_wins"] == 0
        assert stats["team2_wins"] == 0
        assert stats["total"] == 1  # clamped to 1


class TestBuildFixtureAnalytics:
    def test_builds_correctly(self):
        fixture = _make_fixture(999, 1, 10, 0, 0)
        analytics = build_fixture_analytics(
            fixture, SAMPLE_HOME_RECENT, SAMPLE_AWAY_RECENT, SAMPLE_H2H, SAMPLE_ODDS
        )
        assert analytics["fixture_id"] == 999
        assert analytics["home_team"] == "Home FC"
        assert analytics["away_team"] == "Away FC"
        assert analytics["league_id"] == 8
        assert analytics["odds"] == SAMPLE_ODDS
        assert "home_form" in analytics
        assert "away_form" in analytics
        assert "h2h" in analytics


# ---------------------------------------------------------------------------
# predictor tests
# ---------------------------------------------------------------------------

class TestFormScore:
    def test_perfect_form(self):
        form = {"wins": 5, "draws": 0, "losses": 0, "points": 15}
        assert _form_score(form) == pytest.approx(1.0)

    def test_zero_form(self):
        form = {"wins": 0, "draws": 0, "losses": 5, "points": 0}
        assert _form_score(form) == pytest.approx(0.0)

    def test_empty_form(self):
        assert _form_score({}) == 0.5  # neutral default


class TestImpliedProb:
    def test_even_odds(self):
        assert _implied_prob(2.0) == pytest.approx(0.5)

    def test_short_odds(self):
        assert _implied_prob(1.25) == pytest.approx(0.8)

    def test_invalid_odds(self):
        assert _implied_prob(0.5) == 0.0
        assert _implied_prob(1.0) == 0.0


class TestPredictFixture:
    def _build_analytics(self, home_wins=3, away_wins=1, odds=None):
        form = calculate_form(SAMPLE_HOME_RECENT, team_id=1)
        away_form = calculate_form(SAMPLE_AWAY_RECENT, team_id=10)
        h2h = calculate_h2h_stats(SAMPLE_H2H, 1, 10)
        return {
            "fixture_id": 1,
            "home_team": "Home FC",
            "away_team": "Away FC",
            "kickoff": "2025-03-01T15:00:00+00:00",
            "league_id": 8,
            "home_form": form,
            "away_form": away_form,
            "h2h": h2h,
            "odds": odds or SAMPLE_ODDS,
        }

    def test_returns_valid_keys(self):
        analytics = self._build_analytics()
        pred = predict_fixture(analytics)
        for key in ("fixture_id", "home_team", "away_team", "prediction", "confidence", "reasoning", "scores"):
            assert key in pred

    def test_prediction_is_valid_outcome(self):
        analytics = self._build_analytics()
        pred = predict_fixture(analytics)
        assert pred["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)

    def test_confidence_in_range(self):
        analytics = self._build_analytics()
        pred = predict_fixture(analytics)
        assert 0.0 <= pred["confidence"] <= 100.0

    def test_strong_home_form_favours_home_win(self):
        """When home team has perfect form and away team has no wins, expect Home Win."""
        perfect_home_form = {"wins": 5, "draws": 0, "losses": 0, "points": 15, "form_string": "WWWWW",
                             "goals_scored": 10, "goals_conceded": 0, "avg_goals_scored": 2.0, "avg_goals_conceded": 0.0}
        poor_away_form = {"wins": 0, "draws": 0, "losses": 5, "points": 0, "form_string": "LLLLL",
                          "goals_scored": 0, "goals_conceded": 10, "avg_goals_scored": 0.0, "avg_goals_conceded": 2.0}
        analytics = {
            "fixture_id": 2,
            "home_team": "Strong FC",
            "away_team": "Weak FC",
            "kickoff": "2025-03-01T15:00:00+00:00",
            "league_id": 8,
            "home_form": perfect_home_form,
            "away_form": poor_away_form,
            "h2h": {"team1_wins": 4, "team2_wins": 0, "draws": 1, "total": 5,
                    "team1_win_rate": 0.8, "team2_win_rate": 0.0, "draw_rate": 0.2},
            "odds": {"home": 1.4, "draw": 4.5, "away": 7.0},
        }
        pred = predict_fixture(analytics)
        assert pred["prediction"] == HOME_WIN

    def test_empty_analytics(self):
        """Predictor should not crash on missing / empty analytics."""
        pred = predict_fixture({})
        assert pred["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)


class TestPredictAll:
    def test_handles_empty_list(self):
        assert predict_all([]) == []

    def test_handles_multiple_fixtures(self):
        analytics_list = [
            {
                "fixture_id": i,
                "home_team": f"Home{i}",
                "away_team": f"Away{i}",
                "kickoff": "2025-03-01T15:00:00+00:00",
                "league_id": 8,
                "home_form": {},
                "away_form": {},
                "h2h": {},
                "odds": {},
            }
            for i in range(5)
        ]
        results = predict_all(analytics_list)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# League ID regression tests
# ---------------------------------------------------------------------------

class TestLeagueIDs:
    def test_epl_id_is_8(self):
        """England Premier League must map to Sportmonks league_id 8."""
        epl = next(lg for lg in TOP7_LEAGUES if lg["country"] == "England" and lg["name"] == "Premier League")
        assert epl["id"] == 8

    def test_all_5_leagues_present(self):
        """TOP7_LEAGUES must contain exactly 5 leagues."""
        assert len(TOP7_LEAGUES) == 5

    def test_settings_league_ids_match_top7(self):
        """settings.LEAGUE_IDS must contain exactly the same 5 IDs as SUPPORTED_LEAGUES."""
        expected = {lg["id"] for lg in TOP7_LEAGUES}
        assert set(settings.LEAGUE_IDS) == expected

    def test_league_by_id_contains_epl(self):
        """LEAGUE_BY_ID lookup must resolve league_id 8 to England Premier League."""
        assert 8 in LEAGUE_BY_ID
        assert LEAGUE_BY_ID[8]["name"] == "Premier League"
        assert LEAGUE_BY_ID[8]["country"] == "England"
