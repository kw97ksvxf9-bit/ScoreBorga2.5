"""
engine/ml_model.py
Machine Learning model for match outcome prediction in ScoreBorga 2.5.

Uses scikit-learn, XGBoost, and CatBoost to train classifiers on historical match data
from the past N seasons to predict match outcomes.

Supported predictors:
  - MLPredictor: Random Forest classifier
  - GradientBoostingPredictor: Gradient Boosting (One-vs-Rest)
  - LogisticRegressionPredictor: Calibrated Logistic Regression
  - XGBoostPredictor: XGBoost classifier with calibrated probabilities
  - CatBoostPredictor: CatBoost multiclass classifier
  - SVMPredictor: SVM (RBF kernel) with calibrated probabilities
  - EnsemblePredictor: Weighted ensemble of all six models with soft or hard voting
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
XGB_MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_predictor.pkl")
CB_MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cb_predictor.pkl")
SVM_MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "svm_predictor.pkl")


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


class XGBoostPredictor:
    """
    Match outcome predictor using XGBoost classifier with calibrated probabilities.
    Same interface as MLPredictor.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or XGB_MODEL_SAVE_PATH
        self.model: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def train(self, training_samples: List[Dict]) -> bool:
        if not training_samples:
            logger.warning("No training samples provided, cannot train XGB model")
            return False

        logger.info("Training XGBoost model on %d samples...", len(training_samples))

        X, y = [], []
        for sample in training_samples:
            features = sample.get("features", {})
            outcome = sample.get("outcome")
            if features and outcome is not None:
                X.append([features.get(name, 0) for name in FEATURE_NAMES])
                y.append(outcome)

        if not X:
            logger.error("No valid training data for XGB model")
            return False

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.model = CalibratedClassifierCV(xgb, cv=3)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("XGBoost model training complete")
        return True

    def predict(self, analytics: Dict) -> Dict:
        if not self.is_trained or self.model is None:
            logger.warning("XGB model not trained, returning neutral prediction")
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
            logger.warning("Cannot save untrained XGB model")
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            logger.info("XGB model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save XGB model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("XGB model loaded from %s", load_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved XGB model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load XGB model: %s", exc)
            return False


class CatBoostPredictor:
    """
    Match outcome predictor using CatBoost multiclass classifier.
    Does not require feature scaling. Same interface as MLPredictor.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or CB_MODEL_SAVE_PATH
        self.model: Optional[CatBoostClassifier] = None
        self.scaler: Optional[StandardScaler] = None  # Not used; kept for interface consistency
        self.is_trained = False

    def train(self, training_samples: List[Dict]) -> bool:
        if not training_samples:
            logger.warning("No training samples provided, cannot train CatBoost model")
            return False

        logger.info("Training CatBoost model on %d samples...", len(training_samples))

        X, y = [], []
        for sample in training_samples:
            features = sample.get("features", {})
            outcome = sample.get("outcome")
            if features and outcome is not None:
                X.append([features.get(name, 0) for name in FEATURE_NAMES])
                y.append(outcome)

        if not X:
            logger.error("No valid training data for CatBoost model")
            return False

        X = np.array(X)
        y = np.array(y)

        self.model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        )
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("CatBoost model training complete")
        return True

    def predict(self, analytics: Dict) -> Dict:
        if not self.is_trained or self.model is None:
            logger.warning("CatBoost model not trained, returning neutral prediction")
            return {
                "prediction": "Draw",
                "confidence": 33.3,
                "probabilities": {"home": 0.333, "draw": 0.334, "away": 0.333},
            }

        features = _prepare_features(analytics)

        prediction_idx = int(self.model.predict(features).flatten()[0])
        probabilities = self.model.predict_proba(features)[0]

        prediction = OUTCOME_LABELS.get(prediction_idx, "Draw")
        confidence = float(probabilities[prediction_idx]) * 100

        prob_dict = {
            "home": float(probabilities[0]),
            "draw": float(probabilities[1]),
            "away": float(probabilities[2]),
        }

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
        }

    def save_model(self, path: Optional[str] = None) -> bool:
        save_path = path or self.model_path
        if not self.is_trained:
            logger.warning("Cannot save untrained CatBoost model")
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            logger.info("CatBoost model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save CatBoost model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("CatBoost model loaded from %s", load_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved CatBoost model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load CatBoost model: %s", exc)
            return False


class SVMPredictor:
    """
    Match outcome predictor using SVM (RBF kernel) with calibrated probabilities.
    Same interface as MLPredictor.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or SVM_MODEL_SAVE_PATH
        self.model: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def train(self, training_samples: List[Dict]) -> bool:
        if not training_samples:
            logger.warning("No training samples provided, cannot train SVM model")
            return False

        logger.info("Training SVM model on %d samples...", len(training_samples))

        X, y = [], []
        for sample in training_samples:
            features = sample.get("features", {})
            outcome = sample.get("outcome")
            if features and outcome is not None:
                X.append([features.get(name, 0) for name in FEATURE_NAMES])
                y.append(outcome)

        if not X:
            logger.error("No valid training data for SVM model")
            return False

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        self.model = CalibratedClassifierCV(svm, cv=3)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("SVM model training complete")
        return True

    def predict(self, analytics: Dict) -> Dict:
        if not self.is_trained or self.model is None:
            logger.warning("SVM model not trained, returning neutral prediction")
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
            logger.warning("Cannot save untrained SVM model")
            return False
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            logger.info("SVM model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save SVM model: %s", exc)
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            logger.info("SVM model loaded from %s", load_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved SVM model found at %s", load_path)
            return False
        except Exception as exc:
            logger.error("Failed to load SVM model: %s", exc)
            return False


class EnsemblePredictor:
    """
    Weighted ensemble of RF + GB + LR + XGBoost + CatBoost + SVM models.
    Supports two voting modes:
      - "soft": weighted average of all six models' probabilities (default)
      - "hard": each trained model votes for its predicted class; winner is the class
                with the most votes (ties broken by soft-vote probabilities).
    Weights are configurable via settings. Falls back gracefully if any sub-model is not trained.
    """

    def __init__(self, ensemble_mode: Optional[str] = None):
        self._rf = MLPredictor()
        self._gb = GradientBoostingPredictor()
        self._lr = LogisticRegressionPredictor()
        self._xgb = XGBoostPredictor()
        self._cb = CatBoostPredictor()
        self._svm = SVMPredictor()
        self.ensemble_mode = ensemble_mode or settings.ENSEMBLE_MODE
        # Load any pre-trained models from disk
        self._rf.load_model()
        self._gb.load_model()
        self._lr.load_model()
        self._xgb.load_model()
        self._cb.load_model()
        self._svm.load_model()

    @property
    def is_trained(self) -> bool:
        """Return True if at least one sub-model is trained and can make predictions.
        The soft-vote fallback works with any subset of trained sub-models, so a
        partial ensemble is still useful. Use force_retrain=True to retrain all."""
        return any(
            m.is_trained
            for m in (self._rf, self._gb, self._lr, self._xgb, self._cb, self._svm)
        )

    def save_model(self) -> bool:
        """Save all trained sub-models to disk."""
        results = [
            self._rf.save_model(),
            self._gb.save_model(),
            self._lr.save_model(),
            self._xgb.save_model(),
            self._cb.save_model(),
            self._svm.save_model(),
        ]
        return any(results)

    @property
    def _weights(self) -> Dict[str, float]:
        return {
            "rf": settings.ENSEMBLE_RF_WEIGHT,
            "gb": settings.ENSEMBLE_GB_WEIGHT,
            "lr": settings.ENSEMBLE_LR_WEIGHT,
            "xgb": settings.ENSEMBLE_XGB_WEIGHT,
            "cb": settings.ENSEMBLE_CB_WEIGHT,
            "svm": settings.ENSEMBLE_SVM_WEIGHT,
        }

    def train(self, training_samples: List[Dict]) -> bool:
        """Train all six sub-models on the same training data."""
        rf_ok = self._rf.train(training_samples)
        gb_ok = self._gb.train(training_samples)
        lr_ok = self._lr.train(training_samples)
        xgb_ok = self._xgb.train(training_samples)
        cb_ok = self._cb.train(training_samples)
        svm_ok = self._svm.train(training_samples)
        return rf_ok or gb_ok or lr_ok or xgb_ok or cb_ok or svm_ok

    def predict(self, analytics: Dict) -> Dict:
        """
        Predict match outcome using soft or hard ensemble voting.

        Soft (default): weighted average of all six models' probability outputs.
        Hard: each trained model casts one vote; class with most votes wins.
              Ties are broken by soft-vote probabilities.
        Sub-models that are not trained contribute 0 weight (renormalised for soft)
        or are skipped (for hard).
        """
        weights = self._weights
        model_map = {
            "rf": (self._rf, weights["rf"]),
            "gb": (self._gb, weights["gb"]),
            "lr": (self._lr, weights["lr"]),
            "xgb": (self._xgb, weights["xgb"]),
            "cb": (self._cb, weights["cb"]),
            "svm": (self._svm, weights["svm"]),
        }

        if self.ensemble_mode == "hard":
            return self._hard_vote(model_map, analytics)
        return self._soft_vote(model_map, analytics)

    def _soft_vote(self, model_map: Dict, analytics: Dict) -> Dict:
        """Weighted average of all trained models' probabilities."""
        combined = {"home": 0.0, "draw": 0.0, "away": 0.0}
        active_weight = 0.0

        for name, (predictor, w) in model_map.items():
            if predictor.is_trained:
                pred = predictor.predict(analytics)
                probs = pred["probabilities"]
                combined["home"] += w * probs.get("home", 0.0)
                combined["draw"] += w * probs.get("draw", 0.0)
                combined["away"] += w * probs.get("away", 0.0)
                active_weight += w
            else:
                logger.debug("Ensemble: %s sub-model not trained, skipping", name)

        if active_weight > 0:
            combined = {k: v / active_weight for k, v in combined.items()}
        else:
            combined = {"home": 0.333, "draw": 0.334, "away": 0.333}

        max_prob = max(combined.values())
        if max_prob == combined["home"]:
            prediction = "Home Win"
        elif max_prob == combined["draw"]:
            prediction = "Draw"
        else:
            prediction = "Away Win"

        confidence = round(max_prob * 100, 1)
        return {"prediction": prediction, "confidence": confidence, "probabilities": combined}

    def _hard_vote(self, model_map: Dict, analytics: Dict) -> Dict:
        """Majority-class voting; ties broken by soft-vote probabilities."""
        votes: Dict[str, int] = {"Home Win": 0, "Draw": 0, "Away Win": 0}
        total_votes = 0

        for name, (predictor, _) in model_map.items():
            if predictor.is_trained:
                pred = predictor.predict(analytics)
                votes[pred["prediction"]] += 1
                total_votes += 1
            else:
                logger.debug("Ensemble: %s sub-model not trained, skipping", name)

        if total_votes == 0:
            return {
                "prediction": "Draw",
                "confidence": 33.3,
                "probabilities": {"home": 0.333, "draw": 0.334, "away": 0.333},
            }

        max_votes = max(votes.values())
        winners = [cls for cls, v in votes.items() if v == max_votes]

        if len(winners) == 1:
            prediction = winners[0]
        else:
            # Tie-break using soft-vote probabilities
            soft_result = self._soft_vote(model_map, analytics)
            prediction = soft_result["prediction"]

        confidence = round((votes[prediction] / total_votes) * 100, 1)
        # Use soft probabilities as the probability output
        soft = self._soft_vote(model_map, analytics)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": soft["probabilities"],
        }


# ---------------------------------------------------------------------------
# Global predictor instances
# ---------------------------------------------------------------------------

_ml_predictor: Optional[MLPredictor] = None
_ensemble_predictor: Optional[EnsemblePredictor] = None
_xgb_predictor: Optional[XGBoostPredictor] = None
_cb_predictor: Optional[CatBoostPredictor] = None


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


def get_xgb_predictor() -> XGBoostPredictor:
    """Get or create the global XGBoost predictor instance."""
    global _xgb_predictor
    if _xgb_predictor is None:
        _xgb_predictor = XGBoostPredictor()
        _xgb_predictor.load_model()
    return _xgb_predictor


def get_cb_predictor() -> CatBoostPredictor:
    """Get or create the global CatBoost predictor instance."""
    global _cb_predictor
    if _cb_predictor is None:
        _cb_predictor = CatBoostPredictor()
        _cb_predictor.load_model()
    return _cb_predictor


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

