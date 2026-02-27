"""
engine/ml_model.py
Machine Learning model for match outcome prediction in ScoreBorga 2.5.

Uses scikit-learn to train a classifier on historical match data
from the past N seasons to predict match outcomes.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

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
    # Rate-based features (new)
    "home_btts_rate",
    "away_btts_rate",
    "home_over_2_5_rate",
    "away_over_2_5_rate",
    "h2h_btts_rate",
    "h2h_over_2_5_rate",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
]

# Outcome labels: 0 = Home Win, 1 = Draw, 2 = Away Win
OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

# Default model save paths
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ml_predictor.pkl")
GB_MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gb_predictor.pkl")
LR_MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lr_predictor.pkl")


def _prepare_features(analytics: Dict) -> np.ndarray:
    """
    Convert analytics dict into a feature vector matching FEATURE_NAMES.
    Shared by all predictor classes.
    """
    home_form = analytics.get("home_form", {})
    away_form = analytics.get("away_form", {})

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
        # Rate-based features — default to 0 if not present in analytics
        analytics.get("home_btts_rate", 0.0),
        analytics.get("away_btts_rate", 0.0),
        analytics.get("home_over_2_5_rate", 0.0),
        analytics.get("away_over_2_5_rate", 0.0),
        analytics.get("h2h_btts_rate", 0.0),
        analytics.get("h2h_over_2_5_rate", 0.0),
        analytics.get("home_clean_sheet_rate", 0.0),
        analytics.get("away_clean_sheet_rate", 0.0),
    ]

    return np.array(features).reshape(1, -1)


class MLPredictor:
    """
    Machine Learning predictor for match outcomes using Random Forest classifier.
    Trained on historical match data from past seasons.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML predictor.

        Args:
            model_path: Path to load/save the trained model. Defaults to MODEL_SAVE_PATH.
        """
        self.model_path = model_path or MODEL_SAVE_PATH
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def _prepare_features(self, analytics: Dict) -> np.ndarray:
        """Convert analytics dict into a feature vector for prediction."""
        return _prepare_features(analytics)

    def train(self, training_samples: List[Dict]) -> bool:
        """
        Train the ML model on historical match data.

        Args:
            training_samples: List of training sample dicts from HistoricalDataFetcher

        Returns:
            True if training succeeded, False otherwise
        """
        if not training_samples:
            logger.warning("No training samples provided, cannot train ML model")
            return False

        logger.info("Training ML model on %d samples...", len(training_samples))

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

        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Log feature importances
        importances = self.model.feature_importances_
        feature_importance = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )
        logger.info("Top 5 feature importances:")
        for name, importance in feature_importance[:5]:
            logger.info("  %s: %.4f", name, importance)

        logger.info("ML model training complete")
        return True

    def predict(self, analytics: Dict) -> Dict:
        """
        Predict match outcome using the trained ML model.

        Args:
            analytics: Analytics dict from engine/analytics.py

        Returns:
            Dict with keys: prediction (str), confidence (float 0-100), probabilities (dict)
        """
        if not self.is_trained or self.model is None:
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

        # Get prediction and probabilities
        prediction_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Map to outcome labels
        prediction = OUTCOME_LABELS.get(prediction_idx, "Draw")
        confidence = float(probabilities[prediction_idx]) * 100

        # Build probability dict (handle missing classes)
        prob_dict = {"home": 0.0, "draw": 0.0, "away": 0.0}
        classes = self.model.classes_
        for i, cls in enumerate(classes):
            if cls == 0:
                prob_dict["home"] = float(probabilities[i])
            elif cls == 1:
                prob_dict["draw"] = float(probabilities[i])
            elif cls == 2:
                prob_dict["away"] = float(probabilities[i])

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
        }

    def save_model(self, path: Optional[str] = None) -> bool:
        """
        Save the trained model to disk.

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
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            logger.info("ML model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save ML model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.

        Args:
            path: File path to load from. Defaults to self.model_path.

        Returns:
            True if load succeeded, False otherwise
        """
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("ML model loaded from %s", load_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved ML model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load ML model: %s", exc)
            return False


class GradientBoostingPredictor:
    """
    Match outcome predictor using Gradient Boosting (One-vs-Rest multiclass).
    Same interface as MLPredictor.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or GB_MODEL_SAVE_PATH
        self.model: Optional[OneVsRestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def train(self, training_samples: List[Dict]) -> bool:
        if not training_samples:
            logger.warning("No training samples provided, cannot train GB model")
            return False

        logger.info("Training GradientBoosting model on %d samples...", len(training_samples))

        X, y = [], []
        for sample in training_samples:
            features = sample.get("features", {})
            outcome = sample.get("outcome")
            if features and outcome is not None:
                X.append([features.get(name, 0) for name in FEATURE_NAMES])
                y.append(outcome)

        if not X:
            logger.error("No valid training data for GB model")
            return False

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = OneVsRestClassifier(
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("GradientBoosting model training complete")
        return True

    def predict(self, analytics: Dict) -> Dict:
        if not self.is_trained or self.model is None:
            logger.warning("GB model not trained, returning neutral prediction")
            return {
                "prediction": "Draw",
                "confidence": 33.3,
                "probabilities": {"home": 0.333, "draw": 0.334, "away": 0.333},
            }

        features = _prepare_features(analytics)
        if self.scaler is not None:
            features = self.scaler.transform(features)

        prediction_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        prediction = OUTCOME_LABELS.get(prediction_idx, "Draw")
        confidence = float(probabilities[prediction_idx]) * 100

        classes = self.model.classes_
        prob_dict = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for i, cls in enumerate(classes):
            if cls == 0:
                prob_dict["home"] = float(probabilities[i])
            elif cls == 1:
                prob_dict["draw"] = float(probabilities[i])
            elif cls == 2:
                prob_dict["away"] = float(probabilities[i])

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
        }

    def save_model(self, path: Optional[str] = None) -> bool:
        save_path = path or self.model_path
        if not self.is_trained:
            logger.warning("Cannot save untrained GB model")
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            logger.info("GB model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save GB model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("GB model loaded from %s", load_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved GB model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load GB model: %s", exc)
            return False


class LogisticRegressionPredictor:
    """
    Match outcome predictor using Logistic Regression with calibrated probabilities.
    Same interface as MLPredictor.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or LR_MODEL_SAVE_PATH
        self.model: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def train(self, training_samples: List[Dict]) -> bool:
        if not training_samples:
            logger.warning("No training samples provided, cannot train LR model")
            return False

        logger.info("Training LogisticRegression model on %d samples...", len(training_samples))

        X, y = [], []
        for sample in training_samples:
            features = sample.get("features", {})
            outcome = sample.get("outcome")
            if features and outcome is not None:
                X.append([features.get(name, 0) for name in FEATURE_NAMES])
                y.append(outcome)

        if not X:
            logger.error("No valid training data for LR model")
            return False

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        base_lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.model = CalibratedClassifierCV(base_lr, cv=3)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("LogisticRegression model training complete")
        return True

    def predict(self, analytics: Dict) -> Dict:
        if not self.is_trained or self.model is None:
            logger.warning("LR model not trained, returning neutral prediction")
            return {
                "prediction": "Draw",
                "confidence": 33.3,
                "probabilities": {"home": 0.333, "draw": 0.334, "away": 0.333},
            }

        features = _prepare_features(analytics)
        if self.scaler is not None:
            features = self.scaler.transform(features)

        prediction_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        prediction = OUTCOME_LABELS.get(prediction_idx, "Draw")
        confidence = float(probabilities[prediction_idx]) * 100

        classes = self.model.classes_
        prob_dict = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for i, cls in enumerate(classes):
            if cls == 0:
                prob_dict["home"] = float(probabilities[i])
            elif cls == 1:
                prob_dict["draw"] = float(probabilities[i])
            elif cls == 2:
                prob_dict["away"] = float(probabilities[i])

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
        }

    def save_model(self, path: Optional[str] = None) -> bool:
        save_path = path or self.model_path
        if not self.is_trained:
            logger.warning("Cannot save untrained LR model")
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            logger.info("LR model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save LR model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("LR model loaded from %s", load_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved LR model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load LR model: %s", exc)
            return False


class EnsemblePredictor:
    """
    Weighted ensemble of RF + GB + LR models.
    Weights: RF=0.40, GB=0.40, LR=0.20 (configurable via settings).
    Falls back gracefully if any individual model is not trained.
    """

    def __init__(self):
        self._rf = MLPredictor()
        self._gb = GradientBoostingPredictor()
        self._lr = LogisticRegressionPredictor()
        # Load any pre-trained models from disk
        self._rf.load_model()
        self._gb.load_model()
        self._lr.load_model()

    @property
    def _weights(self) -> Dict[str, float]:
        return {
            "rf": settings.ENSEMBLE_RF_WEIGHT,
            "gb": settings.ENSEMBLE_GB_WEIGHT,
            "lr": settings.ENSEMBLE_LR_WEIGHT,
        }

    def train(self, training_samples: List[Dict]) -> bool:
        """Train all three sub-models on the same training data."""
        rf_ok = self._rf.train(training_samples)
        gb_ok = self._gb.train(training_samples)
        lr_ok = self._lr.train(training_samples)
        return rf_ok or gb_ok or lr_ok

    def predict(self, analytics: Dict) -> Dict:
        """
        Weighted average of RF, GB and LR probability outputs.
        Sub-models that are not trained contribute 0 weight (renormalised).
        """
        weights = self._weights
        results = {
            "rf": (self._rf, weights["rf"]),
            "gb": (self._gb, weights["gb"]),
            "lr": (self._lr, weights["lr"]),
        }

        combined = {"home": 0.0, "draw": 0.0, "away": 0.0}
        active_weight = 0.0

        for name, (predictor, w) in results.items():
            if predictor.is_trained:
                pred = predictor.predict(analytics)
                probs = pred["probabilities"]
                combined["home"] += w * probs.get("home", 0.0)
                combined["draw"] += w * probs.get("draw", 0.0)
                combined["away"] += w * probs.get("away", 0.0)
                active_weight += w
            else:
                logger.debug("Ensemble: %s sub-model not trained, skipping", name)

        # Normalise in case some models were missing
        if active_weight > 0:
            combined = {k: v / active_weight for k, v in combined.items()}
        else:
            # All models untrained — fall back to neutral
            combined = {"home": 0.333, "draw": 0.334, "away": 0.333}

        max_prob = max(combined.values())
        if max_prob == combined["home"]:
            prediction = "Home Win"
        elif max_prob == combined["draw"]:
            prediction = "Draw"
        else:
            prediction = "Away Win"

        confidence = round(max_prob * 100, 1)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": combined,
        }


# ---------------------------------------------------------------------------
# Global predictor instances
# ---------------------------------------------------------------------------

_ml_predictor: Optional[MLPredictor] = None
_ensemble_predictor: Optional[EnsemblePredictor] = None


def get_ml_predictor() -> MLPredictor:
    """Get or create the global ML predictor instance."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor()
        # Try to load existing model
        _ml_predictor.load_model()
    return _ml_predictor


def get_ensemble_predictor() -> EnsemblePredictor:
    """Get or create the global Ensemble predictor instance."""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsemblePredictor()
    return _ensemble_predictor


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

