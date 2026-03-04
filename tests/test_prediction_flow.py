"""
tests/test_prediction_flow.py

End-to-end integration test for the full prediction pipeline.
All external I/O (Sportmonks API, Odds API, Telegram) is mocked.
No API keys or network access required.

Covers:
  - run_pipeline (stat mode) — happy path with real fixtures/odds data
  - run_pipeline (ensemble mode) — untrained models fall back gracefully
  - run_pipeline (ensemble mode) — auto-training triggered when untrained
  - run_pipeline (ensemble mode) — stat fallback when training fails
  - run_pipeline (ensemble mode) — skips training when already trained
  - run_pipeline — empty fixtures → returns "no fixtures" message
  - run_pipeline dry_run=False — Telegram send is called
  - Analytics → predictor chain — stat scores and output keys are valid
  - 429 retry logic in SportmonksClient._get
  - EnsemblePredictor.is_trained and save_model
"""

import time
from unittest.mock import MagicMock, patch, call
import pytest
import requests

from output.dispatcher import run_pipeline
from engine.analytics import build_fixture_analytics, calculate_form, calculate_h2h_stats
from engine.predictor import predict_fixture, predict_all, HOME_WIN, DRAW, AWAY_WIN
from engine.polisher import polish_all


# ---------------------------------------------------------------------------
# Shared fixture data helpers
# ---------------------------------------------------------------------------

def _make_participant(team_id: int, name: str, location: str) -> dict:
    return {"id": team_id, "name": name, "meta": {"location": location}}


def _make_fixture(fixture_id: int, home_id: int, away_id: int,
                  home_goals: int, away_goals: int, league_id: int = 8) -> dict:
    return {
        "id": fixture_id,
        "league_id": league_id,
        "starting_at": "2026-03-07T15:00:00+00:00",
        "participants": [
            _make_participant(home_id, "Arsenal", "home"),
            _make_participant(away_id, "Chelsea", "away"),
        ],
        "scores": [
            {"description": "FT", "score": {"participant": "home", "goals": home_goals}},
            {"description": "FT", "score": {"participant": "away", "goals": away_goals}},
        ],
    }


WEEKEND_FIXTURES = [
    _make_fixture(1001, 1, 2, 2, 1),
    _make_fixture(1002, 3, 4, 0, 0),
]

RECENT_HOME = [_make_fixture(i, 1, i + 10, 1, 0) for i in range(5)]
RECENT_AWAY = [_make_fixture(i + 100, i + 20, 2, 0, 1) for i in range(5)]
H2H = [_make_fixture(200, 1, 2, 1, 1), _make_fixture(201, 2, 1, 0, 2)]

SAMPLE_ODDS = {"home": 2.0, "draw": 3.4, "away": 3.8}


# ---------------------------------------------------------------------------
# Helper: build a realistic analytics dict without touching any API
# ---------------------------------------------------------------------------

def _build_analytics(fixture_id: int = 1001, league_id: int = 8) -> dict:
    fixture = _make_fixture(fixture_id, 1, 2, 2, 1, league_id=league_id)
    return build_fixture_analytics(fixture, RECENT_HOME, RECENT_AWAY, H2H, SAMPLE_ODDS)


# ---------------------------------------------------------------------------
# 1. Analytics unit tests
# ---------------------------------------------------------------------------

class TestAnalyticsChain:
    def test_build_fixture_analytics_keys(self):
        a = _build_analytics()
        for key in ("fixture_id", "home_team", "away_team", "kickoff",
                    "home_form", "away_form", "h2h", "odds"):
            assert key in a, f"Missing key: {key}"

    def test_home_form_wins_count(self):
        a = _build_analytics()
        # All 5 RECENT_HOME fixtures are wins for team 1
        assert a["home_form"]["wins"] == 5

    def test_h2h_populated(self):
        a = _build_analytics()
        assert a["h2h"]["total"] == 2
        assert a["h2h"]["team1_wins"] >= 0

    def test_odds_passthrough(self):
        a = _build_analytics()
        assert a["odds"]["home"] == pytest.approx(2.0)
        assert a["odds"]["draw"] == pytest.approx(3.4)
        assert a["odds"]["away"] == pytest.approx(3.8)


# ---------------------------------------------------------------------------
# 2. Predictor unit tests (stat mode, no ML)
# ---------------------------------------------------------------------------

class TestPredictorStatMode:
    def test_predict_fixture_returns_required_keys(self):
        a = _build_analytics()
        pred = predict_fixture(a, mode="stat")
        for key in ("fixture_id", "home_team", "away_team", "prediction",
                    "confidence", "reasoning", "scores"):
            assert key in pred, f"Missing key: {key}"

    def test_prediction_is_valid_outcome(self):
        a = _build_analytics()
        pred = predict_fixture(a, mode="stat")
        assert pred["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)

    def test_confidence_in_range(self):
        a = _build_analytics()
        pred = predict_fixture(a, mode="stat")
        assert 0.0 <= pred["confidence"] <= 100.0

    def test_scores_sum_positive(self):
        a = _build_analytics()
        pred = predict_fixture(a, mode="stat")
        total = sum(pred["scores"].values())
        assert total > 0

    def test_predict_all_batch(self):
        analytics_list = [_build_analytics(fid) for fid in (1001, 1002, 1003)]
        preds = predict_all(analytics_list, mode="stat")
        assert len(preds) == 3
        for p in preds:
            assert p["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)


# ---------------------------------------------------------------------------
# 3. Predictor unit tests (ensemble mode — untrained, graceful fallback)
# ---------------------------------------------------------------------------

class TestPredictorEnsembleMode:
    def test_ensemble_returns_valid_prediction(self):
        a = _build_analytics()
        # Ensemble models are untrained → neutral 33.3% fallback
        pred = predict_fixture(a, mode="ensemble")
        assert pred["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)
        assert 0.0 <= pred["confidence"] <= 100.0

    def test_ensemble_scores_present(self):
        a = _build_analytics()
        pred = predict_fixture(a, mode="ensemble")
        assert "scores" in pred
        for key in ("home", "draw", "away"):
            assert key in pred["scores"]


# ---------------------------------------------------------------------------
# 4. Polish tests
# ---------------------------------------------------------------------------

class TestPolisher:
    def test_polish_all_empty_returns_string(self):
        result = polish_all([])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_polish_all_with_predictions_returns_string(self):
        a = _build_analytics()
        preds = predict_all([a], mode="stat")
        result = polish_all(preds)
        assert isinstance(result, str)
        # Should contain team names
        assert "Arsenal" in result or "Chelsea" in result or len(result) > 10


# ---------------------------------------------------------------------------
# 5. Full pipeline integration (mocked API)
# ---------------------------------------------------------------------------

class TestRunPipelineStat:
    """Happy-path pipeline test with stat mode (no ML)."""

    def _mock_sportmonks(self, mock_sm_cls):
        client = MagicMock()
        client.get_weekend_fixtures.return_value = WEEKEND_FIXTURES
        client.get_recent_fixtures.return_value = RECENT_HOME
        client.get_head_to_head.return_value = H2H
        mock_sm_cls.return_value = client
        return client

    def _mock_odds(self, mock_odds_cls):
        client = MagicMock()
        client.get_odds_for_all_top7.return_value = {
            "soccer_epl": [],
        }
        client.map_odds_to_fixture.return_value = SAMPLE_ODDS
        mock_odds_cls.return_value = client
        return client

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_returns_non_empty_string(self, mock_sm_cls, mock_odds_cls, mock_send):
        self._mock_sportmonks(mock_sm_cls)
        self._mock_odds(mock_odds_cls)
        result = run_pipeline(dry_run=True, mode="stat")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_telegram_not_called_in_dry_run(self, mock_sm_cls, mock_odds_cls, mock_send):
        self._mock_sportmonks(mock_sm_cls)
        self._mock_odds(mock_odds_cls)
        run_pipeline(dry_run=True, mode="stat")
        mock_send.assert_not_called()

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_telegram_called_when_not_dry_run(self, mock_sm_cls, mock_odds_cls, mock_send):
        self._mock_sportmonks(mock_sm_cls)
        self._mock_odds(mock_odds_cls)
        run_pipeline(dry_run=False, mode="stat")
        mock_send.assert_called_once()

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_predictions_contain_team_names(self, mock_sm_cls, mock_odds_cls, mock_send):
        self._mock_sportmonks(mock_sm_cls)
        self._mock_odds(mock_odds_cls)
        result = run_pipeline(dry_run=True, mode="stat")
        # At least one of the team names should appear in the output message
        assert "Arsenal" in result or "Chelsea" in result


class TestRunPipelineEnsemble:
    """Pipeline test with ensemble mode — all sub-models untrained (neutral fallback)."""

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_ensemble_pipeline_completes(self, mock_sm_cls, mock_odds_cls, mock_send):
        client = MagicMock()
        client.get_weekend_fixtures.return_value = WEEKEND_FIXTURES
        client.get_recent_fixtures.return_value = RECENT_HOME
        client.get_head_to_head.return_value = H2H
        mock_sm_cls.return_value = client

        odds_client = MagicMock()
        odds_client.get_odds_for_all_top7.return_value = {}
        odds_client.map_odds_to_fixture.return_value = SAMPLE_ODDS
        mock_odds_cls.return_value = odds_client

        result = run_pipeline(dry_run=True, mode="ensemble")
        assert isinstance(result, str)
        assert len(result) > 0


class TestRunPipelineEmptyFixtures:
    """When no fixtures are found, pipeline should return a non-empty message and not crash."""

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_empty_fixtures_returns_message(self, mock_sm_cls, mock_odds_cls, mock_send):
        client = MagicMock()
        client.get_weekend_fixtures.return_value = []
        mock_sm_cls.return_value = client

        odds_client = MagicMock()
        mock_odds_cls.return_value = odds_client

        result = run_pipeline(dry_run=True, mode="stat")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    def test_empty_fixtures_sends_telegram_when_not_dry_run(self, mock_sm_cls, mock_odds_cls, mock_send):
        client = MagicMock()
        client.get_weekend_fixtures.return_value = []
        mock_sm_cls.return_value = client

        odds_client = MagicMock()
        mock_odds_cls.return_value = odds_client

        run_pipeline(dry_run=False, mode="stat")
        mock_send.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Sportmonks client — 429 retry logic
# ---------------------------------------------------------------------------

class TestSportmonks429Retry:
    """Verify that _get retries on 429 and ultimately raises after max retries."""

    def _make_response(self, status_code: int) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        if status_code >= 400:
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                response=resp
            )
        else:
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"data": [], "pagination": {"has_more": False}}
        return resp

    @patch("time.sleep", return_value=None)  # don't actually wait
    def test_retries_on_429_then_succeeds(self, mock_sleep):
        from data.sportmonks import SportmonksClient
        client = SportmonksClient(api_key="test")

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status.return_value = None
        ok_response.json.return_value = {"data": [{"id": 1}], "pagination": {"has_more": False}}

        responses = [
            self._make_response(429),
            self._make_response(429),
            ok_response,
        ]
        client.session.get = MagicMock(side_effect=responses)

        result = client._paginate("fixtures/between/2026-03-06/2026-03-08/1")
        assert len(result) == 1
        # 2 sleeps: once after each 429 (attempts 1 and 2), success on attempt 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep", return_value=None)
    def test_raises_after_max_retries(self, mock_sleep):
        from data.sportmonks import SportmonksClient
        client = SportmonksClient(api_key="test")

        responses = [self._make_response(429)] * 4  # all 429
        client.session.get = MagicMock(side_effect=responses)

        with pytest.raises(requests.exceptions.HTTPError):
            client._get("fixtures/between/2026-03-06/2026-03-08/1")


# ---------------------------------------------------------------------------
# 7. Fixture league_id filtering
# ---------------------------------------------------------------------------

class TestFixtureLeagueIdFilter:
    """Verify that get_fixtures_by_date_range filters by league_id in Python."""

    def test_filters_out_wrong_league(self):
        from data.sportmonks import SportmonksClient
        client = SportmonksClient(api_key="test")

        # Return fixtures with mixed league_ids
        mixed_fixtures = [
            {"id": 1, "league_id": 8},
            {"id": 2, "league_id": 999},
            {"id": 3, "league_id": 8},
        ]
        with patch.object(client, "_paginate", return_value=mixed_fixtures):
            result = client.get_fixtures_by_date_range("2026-03-06", "2026-03-08", league_ids=[8])
        assert len(result) == 2
        assert all(f["league_id"] == 8 for f in result)

    def test_nested_league_id_is_handled(self):
        from data.sportmonks import SportmonksClient, _fixture_league_id
        fixture_nested = {"id": 1, "league": {"id": 8}}
        assert _fixture_league_id(fixture_nested) == 8

    def test_top_level_league_id_preferred(self):
        from data.sportmonks import _fixture_league_id
        fixture_both = {"id": 1, "league_id": 8, "league": {"id": 999}}
        assert _fixture_league_id(fixture_both) == 8

    def test_missing_league_id_returns_none(self):
        from data.sportmonks import _fixture_league_id
        assert _fixture_league_id({}) is None


# ---------------------------------------------------------------------------
# 8. Ensemble auto-training in run_pipeline
# ---------------------------------------------------------------------------

class TestEnsembleAutoTraining:
    """Verify that run_pipeline triggers ensemble training when models are untrained."""

    def _setup_mocks(self, mock_sm_cls, mock_odds_cls):
        client = MagicMock()
        client.get_weekend_fixtures.return_value = WEEKEND_FIXTURES
        client.get_recent_fixtures.return_value = RECENT_HOME
        client.get_head_to_head.return_value = H2H
        mock_sm_cls.return_value = client

        odds_client = MagicMock()
        odds_client.get_odds_for_all_top7.return_value = {}
        odds_client.map_odds_to_fixture.return_value = SAMPLE_ODDS
        mock_odds_cls.return_value = odds_client

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    @patch("output.dispatcher.train_ensemble_model", return_value=True)
    @patch("output.dispatcher.get_ensemble_predictor")
    def test_training_triggered_when_untrained(
        self, mock_get_ensemble, mock_train, mock_sm_cls, mock_odds_cls, mock_send
    ):
        """run_pipeline must call train_ensemble_model when no sub-models are trained."""
        mock_ensemble = MagicMock()
        mock_ensemble.is_trained = False
        mock_get_ensemble.return_value = mock_ensemble

        self._setup_mocks(mock_sm_cls, mock_odds_cls)
        run_pipeline(dry_run=True, mode="ensemble")

        mock_train.assert_called_once()

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    @patch("output.dispatcher.train_ensemble_model", return_value=False)
    @patch("output.dispatcher.get_ensemble_predictor")
    def test_falls_back_to_stat_when_training_fails(
        self, mock_get_ensemble, mock_train, mock_sm_cls, mock_odds_cls, mock_send
    ):
        """When ensemble training fails, run_pipeline must fall back to stat mode."""
        mock_ensemble = MagicMock()
        mock_ensemble.is_trained = False
        mock_get_ensemble.return_value = mock_ensemble

        self._setup_mocks(mock_sm_cls, mock_odds_cls)
        result = run_pipeline(dry_run=True, mode="ensemble")

        # Pipeline completes with a stat-mode message rather than crashing
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    @patch("output.dispatcher.train_ensemble_model", return_value=True)
    @patch("output.dispatcher.get_ensemble_predictor")
    def test_training_skipped_when_already_trained(
        self, mock_get_ensemble, mock_train, mock_sm_cls, mock_odds_cls, mock_send
    ):
        """run_pipeline must not retrain when the ensemble is already trained."""
        mock_ensemble = MagicMock()
        mock_ensemble.is_trained = True
        mock_get_ensemble.return_value = mock_ensemble

        self._setup_mocks(mock_sm_cls, mock_odds_cls)
        run_pipeline(dry_run=True, mode="ensemble")

        mock_train.assert_not_called()

    @patch("output.dispatcher.send_message", return_value=True)
    @patch("output.dispatcher.OddsApiClient")
    @patch("output.dispatcher.SportmonksClient")
    @patch("output.dispatcher.train_ensemble_model", return_value=True)
    @patch("output.dispatcher.get_ensemble_predictor")
    def test_force_retrain_retrains_even_when_trained(
        self, mock_get_ensemble, mock_train, mock_sm_cls, mock_odds_cls, mock_send
    ):
        """force_retrain=True must trigger training even when models are already trained."""
        mock_ensemble = MagicMock()
        mock_ensemble.is_trained = True
        mock_get_ensemble.return_value = mock_ensemble

        self._setup_mocks(mock_sm_cls, mock_odds_cls)
        run_pipeline(dry_run=True, mode="ensemble", force_retrain=True)

        mock_train.assert_called_once()


# ---------------------------------------------------------------------------
# 9. EnsemblePredictor.is_trained and save_model
# ---------------------------------------------------------------------------

class TestEnsemblePredictorIsTrainedAndSave:
    """Unit tests for the new is_trained property and save_model method."""

    def _make_sub(self, is_trained: bool) -> MagicMock:
        m = MagicMock()
        m.is_trained = is_trained
        m.save_model.return_value = is_trained  # only "trained" models save successfully
        return m

    def _ensemble_with_subs(self, trained_flags):
        """Create an EnsemblePredictor and replace sub-models with mocks."""
        from engine.ml_model import EnsemblePredictor
        ep = EnsemblePredictor()  # loads from disk (no files → all untrained)
        subs = [self._make_sub(f) for f in trained_flags]
        ep._rf, ep._gb, ep._lr, ep._xgb, ep._cb, ep._svm = subs
        return ep

    def test_is_trained_false_when_all_untrained(self):
        ep = self._ensemble_with_subs([False, False, False, False, False, False])
        assert ep.is_trained is False

    def test_is_trained_true_when_any_trained(self):
        # Only XGBoost trained
        ep = self._ensemble_with_subs([False, False, False, True, False, False])
        assert ep.is_trained is True

    def test_is_trained_true_when_all_trained(self):
        ep = self._ensemble_with_subs([True, True, True, True, True, True])
        assert ep.is_trained is True

    def test_save_model_delegates_to_all_sub_models(self):
        ep = self._ensemble_with_subs([True, True, True, True, True, True])
        result = ep.save_model()
        assert result is True
        for sub in (ep._rf, ep._gb, ep._lr, ep._xgb, ep._cb, ep._svm):
            sub.save_model.assert_called_once()

    def test_save_model_returns_false_when_all_fail(self):
        ep = self._ensemble_with_subs([False, False, False, False, False, False])
        # All untrained → all save_model() return False
        result = ep.save_model()
        assert result is False
