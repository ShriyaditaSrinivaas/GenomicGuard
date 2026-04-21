"""
SHAP-based Model Explainer.

Provides local and global explanations for risk predictions using
SHAP (SHapley Additive exPlanations) values.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SHAPExplainer:
    """
    SHAP-based explainability engine for GenomicGuard models.

    Provides:
    - Per-patient SHAP waterfall explanations (why this patient got this score)
    - Global SHAP summary (which features matter most overall)
    - Population-stratified explanations (do feature importances differ by group?)
    - Feature interaction analysis
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize the SHAP explainer.

        Args:
            model: Fitted sklearn model (must support predict/predict_proba)
            feature_names: List of feature column names
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._shap_values = None
        self._base_value = None

    def compute_shap_values(
        self, X: pd.DataFrame, max_samples: int = 500
    ) -> np.ndarray:
        """
        Compute SHAP values for the given data.

        Uses TreeExplainer for tree-based models, otherwise falls back
        to KernelExplainer with a sampled background.

        Args:
            X: Feature DataFrame
            max_samples: Max samples for background (KernelExplainer)

        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        try:
            # Try TreeExplainer (fast, exact for tree models)
            self._explainer = shap.TreeExplainer(self.model)
            shap_values = self._explainer.shap_values(X_values)

            # For binary classification, TreeExplainer may return [neg, pos]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class

            self._base_value = self._explainer.expected_value
            if isinstance(self._base_value, (list, np.ndarray)):
                self._base_value = self._base_value[1] if len(self._base_value) > 1 else self._base_value[0]

        except Exception:
            # Fallback: KernelExplainer (slower, model-agnostic)
            n_bg = min(max_samples, len(X_values))
            background = shap.sample(X_values, n_bg)

            def predict_fn(x):
                return self.model.predict_proba(x)[:, 1]

            self._explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = self._explainer.shap_values(X_values, nsamples=100)
            self._base_value = self._explainer.expected_value

        self._shap_values = shap_values
        return shap_values

    def get_patient_explanation(
        self, patient_idx: int, X: pd.DataFrame
    ) -> Dict:
        """
        Get a detailed explanation for a single patient.

        Args:
            patient_idx: Index of the patient in X
            X: Feature DataFrame

        Returns:
            Dictionary with feature contributions and risk breakdown
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        patient_shap = self._shap_values[patient_idx]
        patient_features = X.iloc[patient_idx].values if isinstance(X, pd.DataFrame) else X[patient_idx]

        # Sort features by absolute SHAP value
        sorted_idx = np.argsort(np.abs(patient_shap))[::-1]

        contributions = []
        for idx in sorted_idx:
            contributions.append({
                "feature": self.feature_names[idx],
                "value": float(patient_features[idx]),
                "shap_value": float(patient_shap[idx]),
                "direction": "increases risk" if patient_shap[idx] > 0 else "decreases risk",
                "magnitude": abs(float(patient_shap[idx])),
            })

        return {
            "patient_idx": patient_idx,
            "base_risk": float(self._base_value) if self._base_value is not None else 0.5,
            "predicted_risk": float(self._base_value + patient_shap.sum()) if self._base_value is not None else float(patient_shap.sum()),
            "top_risk_factors": contributions[:10],
            "top_protective_factors": [c for c in contributions if c["shap_value"] < 0][:5],
            "all_contributions": contributions,
        }

    def get_global_importance(self) -> pd.DataFrame:
        """
        Get global feature importance based on mean |SHAP|.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self._shap_values is None:
            raise RuntimeError("Must compute SHAP values first.")

        mean_abs_shap = np.abs(self._shap_values).mean(axis=0)

        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
            "importance_pct": (mean_abs_shap / mean_abs_shap.sum() * 100).round(2),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def get_population_stratified_importance(
        self,
        X: pd.DataFrame,
        populations: np.ndarray,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute feature importance stratified by population.

        This reveals whether the model relies on different features
        for different populations — a critical fairness signal.
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        results = {}
        for pop in np.unique(populations):
            mask = populations == pop
            pop_shap = self._shap_values[mask]
            mean_abs = np.abs(pop_shap).mean(axis=0)

            results[pop] = pd.DataFrame({
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs,
                "importance_pct": (mean_abs / (mean_abs.sum() + 1e-8) * 100).round(2),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return results

    def get_shap_values_df(self) -> pd.DataFrame:
        """Return SHAP values as a DataFrame."""
        if self._shap_values is None:
            raise RuntimeError("Must compute SHAP values first.")
        return pd.DataFrame(self._shap_values, columns=self.feature_names)

    @property
    def base_value(self):
        """Return the base (expected) value."""
        return self._base_value
