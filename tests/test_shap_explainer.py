"""Tests for the SHAP explainer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from genomicguard.interpretability.shap_explainer import SHAPExplainer


class TestSHAPExplainer:
    """Tests for SHAPExplainer."""

    @pytest.fixture
    def fitted_model_and_data(self):
        """Create a fitted model with sample data."""
        rng = np.random.RandomState(42)
        n, d = 100, 10
        X = pd.DataFrame(rng.randn(n, d), columns=[f"feat_{i}" for i in range(d)])
        y = rng.randint(0, 2, size=n)

        model = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, random_state=42
        )
        model.fit(X.values, y)

        return model, X, y

    def test_compute_shap_values(self, fitted_model_and_data):
        """Test SHAP value computation."""
        model, X, y = fitted_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        shap_values = explainer.compute_shap_values(X)

        assert shap_values.shape == X.shape

    def test_patient_explanation(self, fitted_model_and_data):
        """Test individual patient explanation."""
        model, X, y = fitted_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        explainer.compute_shap_values(X)

        explanation = explainer.get_patient_explanation(0, X)

        assert "patient_idx" in explanation
        assert "top_risk_factors" in explanation
        assert "all_contributions" in explanation
        assert len(explanation["all_contributions"]) == X.shape[1]

    def test_global_importance(self, fitted_model_and_data):
        """Test global feature importance."""
        model, X, y = fitted_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        explainer.compute_shap_values(X)

        importance = explainer.get_global_importance()

        assert len(importance) == X.shape[1]
        assert "feature" in importance.columns
        assert "mean_abs_shap" in importance.columns
        assert all(importance["mean_abs_shap"] >= 0)

    def test_population_stratified_importance(self, fitted_model_and_data):
        """Test population-stratified importance."""
        model, X, y = fitted_model_and_data
        rng = np.random.RandomState(42)
        pops = rng.choice(["A", "B"], size=len(X))

        explainer = SHAPExplainer(model, list(X.columns))
        explainer.compute_shap_values(X)

        pop_importance = explainer.get_population_stratified_importance(X, pops)

        assert "A" in pop_importance
        assert "B" in pop_importance
        assert len(pop_importance["A"]) == X.shape[1]
