"""
tests/test_ml_model.py
Unit tests for engine/ml_model.py and hybrid prediction functionality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from engine.ml_model import (
    MLPredictor,
    FEATURE_NAMES,
    OUTCOME_LABELS,
    predict_with_ml,
)
from engine.predictor import (
    predict_fixture,
    predict_all,
    _stat_based_prediction,
    _hybrid_prediction,
    HOME_WIN,
    DRAW,
    AWAY_WIN,
)


# ---------------------------------------------------------------------------
# Test fixtures and sample data
# ---------------------------------------------------------------------------

SAMPLE_ANALYTICS = {
    "fixture_id": 1,
    "home_team": "Home FC",
    "away_team": "Away FC",
    "kickoff": "2025-03-01T15:00:00+00:00",
    "league_id": 271,
    "home_form": {
        "wins": 3, "draws": 1, "losses": 1,
        "goals_scored": 8, "goals_conceded": 4, "points": 10,
        "form_string": "WDWLW",
        "avg_goals_scored": 1.6, "avg_goals_conceded": 0.8,
    },
    "away_form": {
        "wins": 2, "draws": 1, "losses": 2,
        "goals_scored": 6, "goals_conceded": 6, "points": 7,
        "form_string": "WDLWL",
        "avg_goals_scored": 1.2, "avg_goals_conceded": 1.2,
    },
    "h2h": {
        "team1_wins": 2, "team2_wins": 1, "draws": 1, "total": 4,
        "team1_win_rate": 0.5, "team2_win_rate": 0.25, "draw_rate": 0.25,
    },
    "odds": {"home": 1.8, "draw": 3.5, "away": 4.2},
}

SAMPLE_TRAINING_DATA = [
    {
        "fixture_id": i,
        "features": {
            "home_wins": np.random.randint(0, 5),
            "home_draws": np.random.randint(0, 3),
            "home_losses": np.random.randint(0, 5),
            "home_goals_scored": np.random.randint(0, 15),
            "home_goals_conceded": np.random.randint(0, 10),
            "home_points": np.random.randint(0, 15),
            "home_avg_goals_scored": np.random.random() * 2,
            "home_avg_goals_conceded": np.random.random() * 2,
            "away_wins": np.random.randint(0, 5),
            "away_draws": np.random.randint(0, 3),
            "away_losses": np.random.randint(0, 5),
            "away_goals_scored": np.random.randint(0, 15),
            "away_goals_conceded": np.random.randint(0, 10),
            "away_points": np.random.randint(0, 15),
            "away_avg_goals_scored": np.random.random() * 2,
            "away_avg_goals_conceded": np.random.random() * 2,
            "home_form_score": np.random.random(),
            "away_form_score": np.random.random(),
            "goal_diff_home": np.random.randint(-5, 10),
            "goal_diff_away": np.random.randint(-5, 10),
        },
        "outcome": np.random.randint(0, 3),  # 0=Home, 1=Draw, 2=Away
    }
    for i in range(100)
]


# ---------------------------------------------------------------------------
# MLPredictor tests
# ---------------------------------------------------------------------------

class TestMLPredictor:
    def test_init_creates_untrained_model(self):
        predictor = MLPredictor(model_path="/tmp/test_model.pkl")
        assert predictor.model is None
        assert predictor.scaler is None
        assert predictor.is_trained is False

    def test_train_with_valid_data(self):
        predictor = MLPredictor(model_path="/tmp/test_model.pkl")
        success = predictor.train(SAMPLE_TRAINING_DATA)
        assert success is True
        assert predictor.is_trained is True
        assert predictor.model is not None
        assert predictor.scaler is not None

    def test_train_with_empty_data(self):
        predictor = MLPredictor(model_path="/tmp/test_model.pkl")
        success = predictor.train([])
        assert success is False
        assert predictor.is_trained is False

    def test_predict_returns_valid_structure(self):
        predictor = MLPredictor(model_path="/tmp/test_model.pkl")
        predictor.train(SAMPLE_TRAINING_DATA)

        result = predictor.predict(SAMPLE_ANALYTICS)
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)
        assert 0 <= result["confidence"] <= 100
        assert "home" in result["probabilities"]
        assert "draw" in result["probabilities"]
        assert "away" in result["probabilities"]

    def test_predict_untrained_returns_neutral(self):
        predictor = MLPredictor(model_path="/tmp/test_model.pkl")
        result = predictor.predict(SAMPLE_ANALYTICS)
        assert result["prediction"] == "Draw"
        assert result["confidence"] == pytest.approx(33.3, rel=0.1)

    def test_save_and_load_model(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")

            # Train and save
            predictor1 = MLPredictor(model_path=model_path)
            predictor1.train(SAMPLE_TRAINING_DATA)
            save_success = predictor1.save_model()
            assert save_success is True

            # Load in new instance
            predictor2 = MLPredictor(model_path=model_path)
            load_success = predictor2.load_model()
            assert load_success is True
            assert predictor2.is_trained is True

            # Verify predictions match
            result1 = predictor1.predict(SAMPLE_ANALYTICS)
            result2 = predictor2.predict(SAMPLE_ANALYTICS)
            assert result1["prediction"] == result2["prediction"]

    def test_load_nonexistent_model_fails_gracefully(self):
        predictor = MLPredictor(model_path="/tmp/nonexistent_model.pkl")
        success = predictor.load_model()
        assert success is False
        assert predictor.is_trained is False


class TestFeatureExtraction:
    def test_prepare_features_returns_correct_shape(self):
        predictor = MLPredictor()
        features = predictor._prepare_features(SAMPLE_ANALYTICS)
        assert features.shape == (1, len(FEATURE_NAMES))

    def test_prepare_features_handles_empty_form(self):
        predictor = MLPredictor()
        empty_analytics = {
            "home_form": {},
            "away_form": {},
        }
        features = predictor._prepare_features(empty_analytics)
        assert features.shape == (1, len(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Prediction mode tests
# ---------------------------------------------------------------------------

class TestStatBasedPrediction:
    def test_returns_valid_structure(self):
        result = _stat_based_prediction(SAMPLE_ANALYTICS)
        assert "prediction" in result
        assert "confidence" in result
        assert "scores" in result

    def test_prediction_is_valid_outcome(self):
        result = _stat_based_prediction(SAMPLE_ANALYTICS)
        assert result["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)

    def test_confidence_in_range(self):
        result = _stat_based_prediction(SAMPLE_ANALYTICS)
        assert 0 <= result["confidence"] <= 100


class TestHybridPrediction:
    @patch("engine.ml_model.get_ml_predictor")
    def test_combines_stat_and_ml(self, mock_get_predictor):
        # Mock ML predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "prediction": HOME_WIN,
            "confidence": 60.0,
            "probabilities": {"home": 0.6, "draw": 0.2, "away": 0.2},
        }
        mock_get_predictor.return_value = mock_predictor

        result = _hybrid_prediction(SAMPLE_ANALYTICS)

        assert "prediction" in result
        assert "confidence" in result
        assert "scores" in result
        assert "stat_prediction" in result
        assert "ml_prediction" in result

    @patch("engine.ml_model.get_ml_predictor")
    def test_hybrid_uses_configured_weight(self, mock_get_predictor):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "prediction": HOME_WIN,
            "confidence": 100.0,
            "probabilities": {"home": 1.0, "draw": 0.0, "away": 0.0},
        }
        mock_get_predictor.return_value = mock_predictor

        with patch("engine.predictor.settings") as mock_settings:
            mock_settings.ML_WEIGHT = 0.5
            mock_settings.PREDICTION_MODE = "hybrid"
            result = _hybrid_prediction(SAMPLE_ANALYTICS)

        # Hybrid should blend the predictions
        assert result["scores"]["home"] > 0
        assert 0 <= result["confidence"] <= 100


class TestPredictFixtureWithMode:
    def test_stat_mode(self):
        result = predict_fixture(SAMPLE_ANALYTICS, mode="stat")
        assert "Mode: Statistics" in result["reasoning"]

    @patch("engine.ml_model.get_ml_predictor")
    def test_hybrid_mode(self, mock_get_predictor):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "prediction": HOME_WIN,
            "confidence": 60.0,
            "probabilities": {"home": 0.6, "draw": 0.2, "away": 0.2},
        }
        mock_get_predictor.return_value = mock_predictor

        result = predict_fixture(SAMPLE_ANALYTICS, mode="hybrid")
        assert "Mode: Hybrid (ML+Stats)" in result["reasoning"]

    @patch("engine.ml_model.predict_with_ml")
    def test_ml_mode(self, mock_predict_ml):
        mock_predict_ml.return_value = {
            "prediction": AWAY_WIN,
            "confidence": 55.0,
            "probabilities": {"home": 0.25, "draw": 0.2, "away": 0.55},
        }

        result = predict_fixture(SAMPLE_ANALYTICS, mode="ml")
        assert "Mode: ML Model" in result["reasoning"]
        assert result["prediction"] == AWAY_WIN


class TestPredictAll:
    def test_passes_mode_to_individual_predictions(self):
        analytics_list = [SAMPLE_ANALYTICS.copy(), SAMPLE_ANALYTICS.copy()]
        results = predict_all(analytics_list, mode="stat")
        assert len(results) == 2
        for result in results:
            assert "Mode: Statistics" in result["reasoning"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestMLIntegration:
    def test_full_training_and_prediction_flow(self):
        """Test the complete flow from training to prediction."""
        predictor = MLPredictor(model_path="/tmp/integration_test_model.pkl")

        # Train
        success = predictor.train(SAMPLE_TRAINING_DATA)
        assert success is True

        # Predict
        result = predictor.predict(SAMPLE_ANALYTICS)
        assert result["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)
        assert 0 <= result["confidence"] <= 100

        # Probabilities should sum to ~1
        probs = result["probabilities"]
        total_prob = probs["home"] + probs["draw"] + probs["away"]
        assert 0.99 <= total_prob <= 1.01


class TestOutcomeLabels:
    def test_all_outcomes_mapped(self):
        assert 0 in OUTCOME_LABELS
        assert 1 in OUTCOME_LABELS
        assert 2 in OUTCOME_LABELS
        assert OUTCOME_LABELS[0] == "Home Win"
        assert OUTCOME_LABELS[1] == "Draw"
        assert OUTCOME_LABELS[2] == "Away Win"


# ---------------------------------------------------------------------------
# Ensemble predictor tests
# ---------------------------------------------------------------------------

class TestEnsemblePredictor:
    def test_ensemble_has_multiple_members(self):
        predictor = MLPredictor(model_path="/tmp/ensemble_test.pkl")
        predictor.train(SAMPLE_TRAINING_DATA)
        # Should have at least LR + RF (XGBoost optional)
        assert len(predictor.models) >= 2

    def test_ensemble_probabilities_sum_to_one(self):
        predictor = MLPredictor(model_path="/tmp/ensemble_test.pkl")
        predictor.train(SAMPLE_TRAINING_DATA)
        result = predictor.predict(SAMPLE_ANALYTICS)
        probs = result["probabilities"]
        total = probs["home"] + probs["draw"] + probs["away"]
        assert 0.99 <= total <= 1.01

    def test_ensemble_prediction_valid_outcome(self):
        predictor = MLPredictor(model_path="/tmp/ensemble_test.pkl")
        predictor.train(SAMPLE_TRAINING_DATA)
        result = predictor.predict(SAMPLE_ANALYTICS)
        assert result["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)

    def test_ensemble_confidence_in_range(self):
        predictor = MLPredictor(model_path="/tmp/ensemble_test.pkl")
        predictor.train(SAMPLE_TRAINING_DATA)
        result = predictor.predict(SAMPLE_ANALYTICS)
        assert 0 <= result["confidence"] <= 100

    def test_ensemble_save_and_load(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ensemble.pkl")
            p1 = MLPredictor(model_path=path)
            p1.train(SAMPLE_TRAINING_DATA)
            p1.save_model()

            p2 = MLPredictor(model_path=path)
            assert p2.load_model() is True
            assert p2.is_trained is True
            assert len(p2.models) >= 2

            r1 = p1.predict(SAMPLE_ANALYTICS)
            r2 = p2.predict(SAMPLE_ANALYTICS)
            assert r1["prediction"] == r2["prediction"]

    def test_legacy_single_model_load(self):
        """Ensure an old single-model .pkl file is handled gracefully."""
        import tempfile, os, pickle
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pkl")
            # Create a minimal legacy pickle
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            X = np.random.rand(30, len(FEATURE_NAMES))
            y = np.array([0, 1, 2] * 10)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            rf.fit(X_scaled, y)
            with open(path, "wb") as f:
                pickle.dump({"model": rf, "scaler": scaler}, f)

            p = MLPredictor(model_path=path)
            assert p.load_model() is True
            assert p.is_trained is True
            result = p.predict(SAMPLE_ANALYTICS)
            assert result["prediction"] in (HOME_WIN, DRAW, AWAY_WIN)
