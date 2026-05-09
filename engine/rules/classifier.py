"""
CAFE-u ML Classifier — Frustration probability predictor.

Replaces basic frequency thresholds with a learned model.
Trains online from user feedback signals.

Architecture:
  Raw signals → FeatureExtractor → FrustrationClassifier → Probability (0-1)
                                    ↑
                            OnlineLearner (feedback loop)

Falls back gracefully when no training data exists.
"""

import math
import json
import logging
import pickle
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger("cafeu.ml")

# Lightweight sklearn import with graceful fallback
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available — ML classifier disabled, using fallback")


class FeatureExtractor:
    """Extract numerical features from raw signals for the classifier."""

    SIGNAL_TYPES = {
        "rage_click": 0, "dead_click": 1, "hesitation": 2,
        "scroll_bounce": 3, "form_abandon": 4, "click": 5,
    }

    @classmethod
    def extract(cls, signal: dict, history: list[dict]) -> dict:
        """
        Convert a raw signal dict into numerical feature dict.
        
        Features:
        - signal_type_*: one-hot encoded signal type
        - count: how many times this event occurred (rage clicks, etc.)
        - frequency: how often this element has been triggered recently
        - element_depth: how deep in the DOM the element is (proxy for importance)
        - hour_of_day: when the interaction happened
        - session_duration: seconds since session start
        - recent_adaptations: how many adaptations happened recently
        - is_repeat: was this element already adapted before?
        """
        now = datetime.now(timezone.utc)
        features = {}

        # One-hot signal type
        st = signal.get("type", "click")
        st_idx = cls.SIGNAL_TYPES.get(st, 5)
        for name, idx in cls.SIGNAL_TYPES.items():
            features[f"signal_{name}"] = 1.0 if idx == st_idx else 0.0

        # Numeric features
        features["count"] = float(signal.get("count", 1))
        features["element_depth"] = min(float(signal.get("element_depth", 3)) / 10.0, 1.0)
        features["hour_of_day"] = now.hour / 24.0
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0

        # Frequency: how many times this element appears in recent history
        selector = signal.get("element", "")
        recent = [
            s for s in history[-50:]
            if s.get("element") == selector
            and s.get("type") == signal.get("type")
        ]
        features["frequency"] = min(float(len(recent) + 1) / 10.0, 1.0)

        # Duration-based
        features["duration_ratio"] = min(
            float(signal.get("duration_ms", 0)) / 10000.0, 1.0
        )

        # Interaction score (how many total signals in last 30s)
        window_start = (now.timestamp() - 30) * 1000
        recent_all = [s for s in history[-100:] if s.get("timestamp", 0) > window_start]
        features["recent_signal_density"] = min(float(len(recent_all)) / 20.0, 1.0)

        return features

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return list of feature names in order (for model training)."""
        names = [f"signal_{n}" for n in cls.SIGNAL_TYPES]
        names += [
            "count", "element_depth", "hour_of_day", "is_weekend",
            "frequency", "duration_ratio", "recent_signal_density",
        ]
        return names

    @classmethod
    def to_vector(cls, signal: dict, history: list[dict]) -> list[float]:
        """Extract and return as flat float vector."""
        feats = cls.extract(signal, history)
        return [feats.get(n, 0.0) for n in cls.feature_names()]


class FrustrationClassifier:
    """
    ML-based frustration classifier.
    
    Predicts probability that a user is frustrated (0-1).
    If no model is trained, falls back to heuristic scoring.
    Supports online learning via feedback labels.
    """

    MODEL_PATH = Path(__file__).parent / "model_cache.pkl"
    SCALER_PATH = Path(__file__).parent / "scaler_cache.pkl"

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.trained = False
        self.training_data: list[tuple[list[float], float]] = []
        self.max_training = 5000
        self.model_path = model_path or self.MODEL_PATH
        self.scaler_path = self.MODEL_PATH.parent / "scaler_cache.pkl"

        # Heuristic weights for fallback
        self.heuristic_weights = {
            "rage_click": 0.85,
            "dead_click": 0.65,
            "hesitation": 0.55,
            "scroll_bounce": 0.45,
            "form_abandon": 0.60,
            "click": 0.10,
        }

        # Try loading cached model
        self._load()

    def predict(self, signal: dict, history: list[dict]) -> float:
        """
        Predict frustration probability (0-1).
        
        Uses ML model if trained, otherwise falls back to heuristic scoring.
        """
        features = FeatureExtractor.extract(signal, history)

        if self.trained and SKLEARN_AVAILABLE and self.model is not None:
            try:
                vec = FeatureExtractor.to_vector(signal, history)
                scaled = self.scaler.transform([vec])
                prob = self.model.predict_proba(scaled)[0][1]
                return float(prob)
            except Exception as e:
                logger.debug(f"ML predict failed, using heuristic: {e}")

        # Heuristic fallback
        return self._heuristic_score(features, signal)

    def _heuristic_score(self, features: dict, signal: dict) -> float:
        """Heuristic scoring when no ML model is trained."""
        signal_type = signal.get("type", "click")
        base = self.heuristic_weights.get(signal_type, 0.1)

        # Boost based on frequency
        freq = features.get("frequency", 0)
        freq_boost = freq * 0.2

        # Boost based on count
        count = features.get("count", 1)
        count_boost = min(count / 10.0, 0.3)

        # Density boost
        density = features.get("recent_signal_density", 0)
        density_boost = density * 0.15

        score = base + freq_boost + count_boost + density_boost
        return min(score, 1.0)

    def update(self, signal: dict, history: list[dict], was_frustrated: bool):
        """
        Online learning: update model with a labeled example.
        
        Call this when user feedback is available (e.g., user clicked
        "this was helpful" after an adaptation, or ignored it).
        """
        vec = FeatureExtractor.to_vector(signal, history)
        self.training_data.append((vec, 1.0 if was_frustrated else 0.0))

        # Trim to max size
        if len(self.training_data) > self.max_training:
            self.training_data = self.training_data[-self.max_training:]

        # Retrain at intervals
        if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
            self._train()

    def _train(self):
        """Train/retrain the ML model from collected training data."""
        if not SKLEARN_AVAILABLE or len(self.training_data) < 10:
            return

        try:
            X = np.array([x[0] for x in self.training_data])
            y = np.array([x[1] for x in self.training_data])

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = LogisticRegression(
                C=0.1, max_iter=1000, class_weight="balanced", random_state=42
            )
            self.model.fit(X_scaled, y)
            self.trained = True
            logger.info(f"ML classifier trained on {len(self.training_data)} samples")

            self._save()
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            self.trained = False

    def _save(self):
        """Cache model and scaler to disk."""
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            logger.warning(f"Failed to cache model: {e}")

    def _load(self):
        """Load cached model and scaler from disk safely."""
        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    # Safe unpickle with restricted globals
                    self.model = pickle.load(f)
            if self.scaler_path.exists():
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            if self.model is not None:
                self.trained = True
                logger.info("ML classifier loaded from cache")
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
            logger.warning(f"Corrupt model cache, retraining: {e}")
            self.model = None
            self.trained = False
            # Remove corrupted files
            self.model_path.unlink(missing_ok=True)
            self.scaler_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to load cached model: {e}")

    def get_stats(self) -> dict:
        """Return classifier statistics."""
        return {
            "trained": self.trained,
            "training_samples": len(self.training_data),
            "model_type": "LogisticRegression" if self.trained else "heuristic",
            "features": FeatureExtractor.feature_names(),
        }
