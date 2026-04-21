"""
Polygenic Risk Score Model.

Gradient Boosting classifier for disease risk prediction using
genomic and clinical features. Designed for clinical use with
calibrated probability outputs.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)

from genomicguard.config import ModelConfig, MODELS_DIR


class PolygenicRiskScorer:
    """
    Gradient Boosting model for polygenic risk scoring.

    This model combines genomic features (PRS, PCA components, interactions)
    with clinical risk factors to produce calibrated disease risk probabilities.

    Key design decisions for clinical use:
    - Probability calibration via Platt scaling for reliable risk estimates
    - Built-in cross-validation for robust performance estimation
    - Feature importance tracking for interpretability
    """

    def __init__(self, config: Optional[ModelConfig] = None, disease_name: str = "generic"):
        self.config = config or ModelConfig()
        self.disease_name = disease_name
        self.model = None
        self.calibrated_model = None
        self._feature_names = None
        self._is_fitted = False
        self._cv_scores = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        calibrate: bool = True,
    ) -> Dict:
        """
        Train the risk scoring model.

        Args:
            X: Feature DataFrame
            y: Binary disease labels
            calibrate: Whether to apply probability calibration

        Returns:
            Dictionary of training metrics
        """
        self._feature_names = list(X.columns)

        # Base model
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            max_depth=self.config.gb_max_depth,
            learning_rate=self.config.gb_learning_rate,
            min_samples_split=self.config.gb_min_samples_split,
            subsample=self.config.gb_subsample,
            random_state=self.config.random_seed,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        )

        if calibrate:
            # Calibrated model with cross-validation
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.model,
                cv=3,
                method="sigmoid",
            )
            self.calibrated_model.fit(X.values, y)

            # Also fit base model for feature importances
            self.model.fit(X.values, y)
        else:
            self.model.fit(X.values, y)

        self._is_fitted = True

        # Compute training metrics
        metrics = self._compute_metrics(X, y)
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated disease risk probabilities.

        Returns:
            Array of shape (n_samples, 2) with [prob_negative, prob_positive]
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X.values)
        return self.model.predict_proba(X.values)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get risk scores (probability of disease).

        Returns:
            Array of risk scores in [0, 1]
        """
        return self.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction at given threshold."""
        return (self.predict_risk(X) >= threshold).astype(int)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict:
        """
        Perform stratified cross-validation.

        Returns:
            Dictionary with per-fold and aggregate metrics
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Clone model
            fold_model = GradientBoostingClassifier(
                n_estimators=self.config.gb_n_estimators,
                max_depth=self.config.gb_max_depth,
                learning_rate=self.config.gb_learning_rate,
                min_samples_split=self.config.gb_min_samples_split,
                subsample=self.config.gb_subsample,
                random_state=self.config.random_seed + fold,
                validation_fraction=0.1,
                n_iter_no_change=20,
                tol=1e-4,
            )
            fold_model.fit(X_train.values, y_train)
            y_prob = fold_model.predict_proba(X_val.values)[:, 1]

            fold_metrics.append({
                "fold": fold + 1,
                "auc_roc": roc_auc_score(y_val, y_prob),
                "auc_pr": average_precision_score(y_val, y_prob),
                "brier_score": brier_score_loss(y_val, y_prob),
            })

        self._cv_scores = {
            "folds": fold_metrics,
            "mean_auc_roc": np.mean([m["auc_roc"] for m in fold_metrics]),
            "std_auc_roc": np.std([m["auc_roc"] for m in fold_metrics]),
            "mean_auc_pr": np.mean([m["auc_pr"] for m in fold_metrics]),
            "mean_brier": np.mean([m["brier_score"] for m in fold_metrics]),
        }
        return self._cv_scores

    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances from the base model."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted first.")

        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def _compute_metrics(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Compute comprehensive metrics."""
        y_prob = self.predict_risk(X)
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "disease": self.disease_name,
            "auc_roc": float(roc_auc_score(y, y_prob)),
            "auc_pr": float(average_precision_score(y, y_prob)),
            "brier_score": float(brier_score_loss(y, y_prob)),
            "n_samples": int(len(y)),
            "prevalence": float(y.mean()),
        }

    def save(self, filepath: Optional[Path] = None):
        """Save model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / f"risk_scorer_{self.disease_name}.joblib"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_names": self._feature_names,
            "disease_name": self.disease_name,
            "cv_scores": self._cv_scores,
            "config": self.config,
        }
        joblib.dump(model_data, filepath)
        print(f"  ✓ Saved risk scorer to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "PolygenicRiskScorer":
        """Load model from disk."""
        data = joblib.load(filepath)
        scorer = cls(config=data["config"], disease_name=data["disease_name"])
        scorer.model = data["model"]
        scorer.calibrated_model = data["calibrated_model"]
        scorer._feature_names = data["feature_names"]
        scorer._cv_scores = data["cv_scores"]
        scorer._is_fitted = True
        return scorer
