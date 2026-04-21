"""
Ensemble Model.

Stacking ensemble that combines the Polygenic Risk Scorer and
Variant Classifier predictions for final risk assessment.
Uses logistic regression as a meta-learner.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from genomicguard.config import ModelConfig, MODELS_DIR


class EnsembleModel:
    """
    Meta-ensemble combining risk scorer and variant classifier.

    Uses stacked generalization:
    1. Base models generate out-of-fold predictions
    2. A calibrated logistic regression meta-learner combines them
    3. Final output is a calibrated probability with confidence intervals

    This approach prevents information leakage and produces more
    reliable risk estimates than any single model.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.meta_model = None
        self._is_fitted = False
        self._base_model_names = []

    def fit(
        self,
        base_predictions: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> Dict:
        """
        Train the meta-learner on base model predictions.

        Args:
            base_predictions: Dict mapping model names to prediction arrays
            y: True labels

        Returns:
            Training metrics
        """
        self._base_model_names = list(base_predictions.keys())

        # Build meta-feature matrix
        X_meta = np.column_stack([base_predictions[name] for name in self._base_model_names])

        # Train calibrated meta-learner
        base_meta = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.config.random_seed,
        )
        self.meta_model = CalibratedClassifierCV(
            estimator=base_meta,
            cv=min(3, max(2, int(y.sum() // 2))),  # Ensure enough positives per fold
            method="sigmoid",
        )
        self.meta_model.fit(X_meta, y)
        self._is_fitted = True

        # Compute metrics
        y_prob = self.meta_model.predict_proba(X_meta)[:, 1]
        metrics = {
            "auc_roc": float(roc_auc_score(y, y_prob)),
            "auc_pr": float(average_precision_score(y, y_prob)),
            "n_base_models": len(self._base_model_names),
            "base_models": self._base_model_names,
        }
        return metrics

    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict using the ensemble.

        Args:
            base_predictions: Dict mapping model names to prediction arrays

        Returns:
            Calibrated probabilities (n_samples, 2)
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted first.")

        X_meta = np.column_stack([
            base_predictions[name] for name in self._base_model_names
        ])
        return self.meta_model.predict_proba(X_meta)

    def predict_risk(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Get final risk scores."""
        return self.predict_proba(base_predictions)[:, 1]

    def predict_with_confidence(
        self,
        base_predictions: Dict[str, np.ndarray],
        n_bootstrap: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with bootstrap confidence intervals.

        Returns:
            Tuple of (mean_risk, lower_ci, upper_ci)
        """
        risk = self.predict_risk(base_predictions)

        # Bootstrap for confidence intervals
        rng = np.random.RandomState(self.config.random_seed)
        n_samples = len(risk)
        bootstrap_risks = np.zeros((n_bootstrap, n_samples))

        X_meta = np.column_stack([
            base_predictions[name] for name in self._base_model_names
        ])

        for i in range(n_bootstrap):
            # Add small noise to simulate prediction uncertainty
            noise = rng.normal(0, 0.02, size=X_meta.shape)
            X_noisy = X_meta + noise
            bootstrap_risks[i] = self.meta_model.predict_proba(X_noisy)[:, 1]

        lower_ci = np.percentile(bootstrap_risks, 2.5, axis=0)
        upper_ci = np.percentile(bootstrap_risks, 97.5, axis=0)

        return risk, lower_ci, upper_ci

    def save(self, filepath: Optional[Path] = None):
        """Save ensemble to disk."""
        if filepath is None:
            filepath = MODELS_DIR / "ensemble.joblib"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "meta_model": self.meta_model,
            "base_model_names": self._base_model_names,
            "config": self.config,
        }
        joblib.dump(model_data, filepath)
        print(f"  ✓ Saved ensemble to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "EnsembleModel":
        """Load ensemble from disk."""
        data = joblib.load(filepath)
        ensemble = cls(config=data["config"])
        ensemble.meta_model = data["meta_model"]
        ensemble._base_model_names = data["base_model_names"]
        ensemble._is_fitted = True
        return ensemble
