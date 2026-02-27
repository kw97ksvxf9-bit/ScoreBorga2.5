"""
engine/ml_model.py
Machine Learning model for match outcome prediction in ScoreBorga 2.5.

Uses scikit-learn to train an ensemble classifier (soft voting) on historical
match data from the past N seasons to predict match outcomes.

Ensemble members:
  - Logistic Regression
  - Random Forest classifier
  - XGBoost classifier (optional; gracefully skipped if not installed)
"""

import logging
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost is not installed. The ensemble will run without XGBoost. "
        "Install it with: pip install xgboost",
        ImportWarning,
        stacklevel=2,
    )

from config.settings import settings

logger = logging.getLogger(__name__)

# Feature names used for training (must match features from historical.py)
FEATURE_NAMES = [
    "home_wins",
    "home_draws",
    "home_losses",
    "home_goals_scored",
    "home_goals_conceded",
    "home_points",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_wins",
    "away_draws",
    "away_losses",
    "away_goals_scored",
    "away_goals_conceded",
    "away_points",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "home_form_score",
    "away_form_score",
    "goal_diff_home",
    "goal_diff_away",
]

# Outcome labels: 0 = Home Win, 1 = Draw, 2 = Away Win
OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

# Default model save path
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ml_predictor.pkl")


class MLPredictor:
    """
    Machine Learning predictor for match outcomes using a soft-voting ensemble.

    Ensemble members (in order of addition):
      - Logistic Regression
      - Random Forest classifier
      - XGBoost classifier (only when xgboost is installed)

    Trained on historical match data from past seasons.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML predictor.

        Args:
            model_path: Path to load/save the trained model. Defaults to MODEL_SAVE_PATH.
        """
        self.model_path = model_path or MODEL_SAVE_PATH
        # Individual ensemble members
        self.models: List[Any] = []
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

        # Keep a reference to the RF model for feature importances (backwards-compat)
        self.model: Optional[RandomForestClassifier] = None

    def _prepare_features(self, analytics: Dict) -> np.ndarray:
        """
        Convert analytics dict into a feature vector for prediction.

        Args:
            analytics: Analytics dict from engine/analytics.py

        Returns:
            numpy array of features
        """
        home_form = analytics.get("home_form", {})
        away_form = analytics.get("away_form", {})

        # Calculate derived features
        home_matches = (
            home_form.get("wins", 0) +
            home_form.get("draws", 0) +
            home_form.get("losses", 0)
        ) or 1
        away_matches = (
            away_form.get("wins", 0) +
            away_form.get("draws", 0) +
            away_form.get("losses", 0)
        ) or 1

        home_form_score = home_form.get("points", 0) / (home_matches * 3)
        away_form_score = away_form.get("points", 0) / (away_matches * 3)

        features = [
            home_form.get("wins", 0),
            home_form.get("draws", 0),
            home_form.get("losses", 0),
            home_form.get("goals_scored", 0),
            home_form.get("goals_conceded", 0),
            home_form.get("points", 0),
            home_form.get("avg_goals_scored", 0.0),
            home_form.get("avg_goals_conceded", 0.0),
            away_form.get("wins", 0),
            away_form.get("draws", 0),
            away_form.get("losses", 0),
            away_form.get("goals_scored", 0),
            away_form.get("goals_conceded", 0),
            away_form.get("points", 0),
            away_form.get("avg_goals_scored", 0.0),
            away_form.get("avg_goals_conceded", 0.0),
            home_form_score,
            away_form_score,
            home_form.get("goals_scored", 0) - home_form.get("goals_conceded", 0),
            away_form.get("goals_scored", 0) - away_form.get("goals_conceded", 0),
        ]

        return np.array(features).reshape(1, -1)

    def train(self, training_samples: List[Dict]) -> bool:
        """
        Train the ensemble ML model on historical match data.

        Args:
            training_samples: List of training sample dicts from HistoricalDataFetcher

        Returns:
            True if training succeeded, False otherwise
        """
        if not training_samples:
            logger.warning("No training samples provided, cannot train ML model")
            return False

        logger.info("Training ensemble ML model on %d samples...", len(training_samples))

        # Extract features and outcomes
        X = []
        y = []
        for sample in training_samples:
            features = sample.get("features", {})
            outcome = sample.get("outcome")
            if features and outcome is not None:
                feature_vector = [features.get(name, 0) for name in FEATURE_NAMES]
                X.append(feature_vector)
                y.append(outcome)

        if not X:
            logger.error("No valid training data after feature extraction")
            return False

        X = np.array(X)
        y = np.array(y)

        logger.info("Training data shape: X=%s, y=%s", X.shape, y.shape)
        logger.info("Outcome distribution: Home Win=%d, Draw=%d, Away Win=%d",
                    np.sum(y == 0), np.sum(y == 1), np.sum(y == 2))

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build ensemble members
        ensemble = []

        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )
        lr.fit(X_scaled, y)
        ensemble.append(("lr", lr))

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_scaled, y)
        ensemble.append(("rf", rf))
        self.model = rf  # backwards-compat reference

        if _XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
                verbosity=0,
            )
            xgb.fit(X_scaled, y)
            ensemble.append(("xgb", xgb))
        else:
            logger.warning("XGBoost not available — ensemble running without XGBoost")

        self.models = ensemble
        self.is_trained = True

        # Log RF feature importances for transparency
        importances = rf.feature_importances_
        feature_importance = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )
        logger.info("Top 5 RF feature importances:")
        for name, importance in feature_importance[:5]:
            logger.info("  %s: %.4f", name, importance)

        logger.info("Ensemble ML model training complete (%d members)", len(ensemble))
        return True

    def predict(self, analytics: Dict) -> Dict:
        """
        Predict match outcome using soft-voting across the trained ensemble.

        Args:
            analytics: Analytics dict from engine/analytics.py

        Returns:
            Dict with keys: prediction (str), confidence (float 0-100), probabilities (dict)
        """
        if not self.is_trained or not self.models:
            logger.warning("ML model not trained, returning neutral prediction")
            return {
                "prediction": "Draw",
                "confidence": 33.3,
                "probabilities": {"home": 0.333, "draw": 0.334, "away": 0.333},
            }

        features = self._prepare_features(analytics)

        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Soft voting: average predict_proba across all ensemble members
        # Each model may have seen classes [0, 1, 2]; we aggregate by class index
        avg_probs = np.zeros(3)  # indices: 0=Home Win, 1=Draw, 2=Away Win
        for _name, mdl in self.models:
            proba = mdl.predict_proba(features)[0]
            classes = mdl.classes_
            for i, cls in enumerate(classes):
                if 0 <= cls <= 2:
                    avg_probs[cls] += proba[i]
        avg_probs /= len(self.models)

        # Determine winning class
        prediction_idx = int(np.argmax(avg_probs))
        prediction = OUTCOME_LABELS.get(prediction_idx, "Draw")
        confidence = float(avg_probs[prediction_idx]) * 100

        prob_dict = {
            "home": float(avg_probs[0]),
            "draw": float(avg_probs[1]),
            "away": float(avg_probs[2]),
        }

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
        }

    def save_model(self, path: Optional[str] = None) -> bool:
        """
        Save the trained ensemble to disk.

        Args:
            path: File path to save to. Defaults to self.model_path.

        Returns:
            True if save succeeded, False otherwise
        """
        save_path = path or self.model_path
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump({"models": self.models, "scaler": self.scaler}, f)
            logger.info("Ensemble ML model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save ML model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load a trained ensemble from disk.

        Args:
            path: File path to load from. Defaults to self.model_path.

        Returns:
            True if load succeeded, False otherwise
        """
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            # Support both old single-model format and new ensemble format
            if "models" in data:
                self.models = data["models"]
                # Rebuild backwards-compat self.model reference (RF if present)
                for name, mdl in self.models:
                    if name == "rf":
                        self.model = mdl
                        break
            elif "model" in data:
                # Legacy single-model file — wrap as single-element ensemble
                legacy_model = data["model"]
                self.models = [("rf", legacy_model)]
                self.model = legacy_model
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("Ensemble ML model loaded from %s (%d members)", load_path, len(self.models))
            return True
        except FileNotFoundError:
            logger.warning("No saved ML model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load ML model: %s", exc)
            return False


# Global ML predictor instance
_ml_predictor: Optional[MLPredictor] = None


def get_ml_predictor() -> MLPredictor:
    """Get or create the global ML predictor instance."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor()
        # Try to load existing model
        _ml_predictor.load_model()
    return _ml_predictor


def predict_with_ml(analytics: Dict) -> Dict:
    """
    Convenience function to predict using the global ML predictor.

    Args:
        analytics: Analytics dict from engine/analytics.py

    Returns:
        Dict with prediction, confidence, and probabilities
    """
    predictor = get_ml_predictor()
    return predictor.predict(analytics)
