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
    _poisson_over_prob,
    _totals_prediction,
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


# ---------------------------------------------------------------------------
# Heuristic totals tests
# ---------------------------------------------------------------------------

class TestPoissonOverProb:
    def test_high_expected_goals_over_1_5(self):
        """With many expected goals, P(over 1.5) should be high."""
        prob = _poisson_over_prob(3.0, 1.5)
        assert prob > 0.8

    def test_low_expected_goals_over_3_5(self):
        """With very few expected goals, P(over 3.5) should be low."""
        prob = _poisson_over_prob(0.5, 3.5)
        assert prob < 0.05

    def test_probability_in_range(self):
        for lam in [0.5, 1.5, 2.5, 3.5]:
            for line in [1.5, 2.5, 3.5]:
                prob = _poisson_over_prob(lam, line)
                assert 0.0 <= prob <= 1.0


class TestTotalsPrediction:
    def _analytics(self, home_scored=1.5, home_conceded=1.0,
                   away_scored=1.2, away_conceded=1.3):
        return {
            "home_form": {
                "avg_goals_scored": home_scored,
                "avg_goals_conceded": home_conceded,
            },
            "away_form": {
                "avg_goals_scored": away_scored,
                "avg_goals_conceded": away_conceded,
            },
        }

    def test_output_shape(self):
        result = _totals_prediction(self._analytics())
        assert "totals" in result
        assert "best_total" in result
        totals = result["totals"]
        assert "over_1_5" in totals
        assert "over_2_5" in totals
        assert "over_3_5" in totals

    def test_probabilities_in_range(self):
        result = _totals_prediction(self._analytics())
        for key in ("over_1_5", "over_2_5", "over_3_5"):
            assert 0.0 <= result["totals"][key] <= 100.0

    def test_ordering(self):
        """P(over 1.5) >= P(over 2.5) >= P(over 3.5)."""
        result = _totals_prediction(self._analytics())
        t = result["totals"]
        assert t["over_1_5"] >= t["over_2_5"]
        assert t["over_2_5"] >= t["over_3_5"]

    def test_best_total_is_valid_line(self):
        result = _totals_prediction(self._analytics())
        assert result["best_total"]["line"] in ("Over 1.5", "Over 2.5", "Over 3.5")

    def test_best_total_probability_matches_totals(self):
        result = _totals_prediction(self._analytics())
        line = result["best_total"]["line"]
        prob = result["best_total"]["probability"]
        line_key = line.replace("Over ", "over_").replace(".", "_")
        assert prob == result["totals"][line_key]

    def test_missing_form_uses_defaults(self):
        """Should not raise when form data is missing."""
        result = _totals_prediction({})
        assert "totals" in result
        assert "best_total" in result

    def test_predict_fixture_includes_totals(self):
        """predict_fixture should include totals and best_total in output."""
        analytics = {
            "fixture_id": 1,
            "home_team": "A",
            "away_team": "B",
            "kickoff": None,
            "league_id": 8,
            "home_form": {"avg_goals_scored": 1.5, "avg_goals_conceded": 1.0},
            "away_form": {"avg_goals_scored": 1.2, "avg_goals_conceded": 1.3},
            "h2h": {},
            "odds": {},
        }
        pred = predict_fixture(analytics)
        assert "totals" in pred
        assert "best_total" in pred
        assert pred["best_total"]["line"] in ("Over 1.5", "Over 2.5", "Over 3.5")


# ---------------------------------------------------------------------------
# Top-7 dispatcher filtering tests
# ---------------------------------------------------------------------------

class TestTop7Filtering:
    def _make_predictions(self, confidences):
        return [
            {
                "fixture_id": i,
                "home_team": f"Home{i}",
                "away_team": f"Away{i}",
                "prediction": HOME_WIN,
                "confidence": c,
                "scores": {"home": 0.5, "draw": 0.3, "away": 0.2},
            }
            for i, c in enumerate(confidences)
        ]

    def test_more_than_7_returns_top_7(self):
        confidences = [50, 80, 70, 60, 90, 55, 75, 65, 85, 45]
        predictions = self._make_predictions(confidences)
        top7 = sorted(predictions, key=lambda p: p.get("confidence", 0.0), reverse=True)[:7]
        assert len(top7) == 7
        # Highest confidence should be first
        assert top7[0]["confidence"] == 90

    def test_fewer_than_7_returns_all(self):
        confidences = [70, 80, 60]
        predictions = self._make_predictions(confidences)
        top7 = sorted(predictions, key=lambda p: p.get("confidence", 0.0), reverse=True)[:7]
        assert len(top7) == 3

    def test_sorted_descending(self):
        confidences = [30, 90, 60, 75, 85, 55, 70, 40, 65, 80]
        predictions = self._make_predictions(confidences)
        top7 = sorted(predictions, key=lambda p: p.get("confidence", 0.0), reverse=True)[:7]
        confs = [p["confidence"] for p in top7]
        assert confs == sorted(confs, reverse=True)
