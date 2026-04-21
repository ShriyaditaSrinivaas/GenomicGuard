"""Tests for the risk scorer model."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.config import ModelConfig


class TestPolygenicRiskScorer:
    """Tests for PolygenicRiskScorer."""

    def test_fit_and_predict(self, sample_binary_data, model_config):
        """Test basic fit and predict workflow."""
        X, y, _ = sample_binary_data
        scorer = PolygenicRiskScorer(config=model_config, disease_name="test")
        metrics = scorer.fit(X, y, calibrate=False)

        assert "auc_roc" in metrics
        assert metrics["auc_roc"] >= 0.0

        predictions = scorer.predict_risk(X)
        assert len(predictions) == len(X)
        assert all(0 <= p <= 1 for p in predictions)

    def test_calibrated_predictions(self, sample_binary_data, model_config):
        """Test that calibrated model produces valid probabilities."""
        X, y, _ = sample_binary_data
        scorer = PolygenicRiskScorer(config=model_config)
        scorer.fit(X, y, calibrate=True)

        proba = scorer.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_cross_validate(self, sample_binary_data, model_config):
        """Test cross-validation."""
        X, y, _ = sample_binary_data
        scorer = PolygenicRiskScorer(config=model_config)
        cv_results = scorer.cross_validate(X, y, n_folds=3)

        assert "mean_auc_roc" in cv_results
        assert "folds" in cv_results
        assert len(cv_results["folds"]) == 3

    def test_feature_importances(self, sample_binary_data, model_config):
        """Test feature importance extraction."""
        X, y, _ = sample_binary_data
        scorer = PolygenicRiskScorer(config=model_config)
        scorer.fit(X, y, calibrate=False)

        importances = scorer.get_feature_importances()
        assert len(importances) == X.shape[1]
        assert "feature" in importances.columns
        assert "importance" in importances.columns

    def test_save_and_load(self, sample_binary_data, model_config):
        """Test model serialization."""
        X, y, _ = sample_binary_data
        scorer = PolygenicRiskScorer(config=model_config, disease_name="test_save")
        scorer.fit(X, y, calibrate=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            scorer.save(filepath)

            loaded = PolygenicRiskScorer.load(filepath)
            original_preds = scorer.predict_risk(X)
            loaded_preds = loaded.predict_risk(X)
            np.testing.assert_allclose(original_preds, loaded_preds, atol=1e-6)

    def test_unfitted_model_raises(self, sample_binary_data):
        """Test that predicting before fitting raises an error."""
        X, _, _ = sample_binary_data
        scorer = PolygenicRiskScorer()
        with pytest.raises(RuntimeError):
            scorer.predict_risk(X)
