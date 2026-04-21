"""Tests for the fairness auditor."""

import numpy as np
import pytest
from genomicguard.fairness.auditor import FairnessAuditor
from genomicguard.fairness.bias_detector import BiasDetector
from genomicguard.fairness.mitigation import BiasMitigator


class TestFairnessAuditor:
    """Tests for FairnessAuditor."""

    def test_audit_returns_all_sections(self, sample_risk_data):
        """Test that audit returns all expected sections."""
        y_true, y_prob, groups = sample_risk_data
        auditor = FairnessAuditor()
        results = auditor.audit(y_true, y_prob, groups)

        assert "group_metrics" in results
        assert "fairness_metrics" in results
        assert "statistical_tests" in results
        assert "fairness_summary" in results
        assert "overall_metrics" in results

    def test_group_metrics(self, sample_risk_data):
        """Test that group metrics are computed for all groups."""
        y_true, y_prob, groups = sample_risk_data
        auditor = FairnessAuditor()
        results = auditor.audit(y_true, y_prob, groups)

        for group in np.unique(groups):
            assert group in results["group_metrics"]
            metrics = results["group_metrics"][group]
            assert "n_samples" in metrics
            assert "prevalence" in metrics

    def test_fairness_metrics_structure(self, sample_risk_data):
        """Test fairness metrics structure."""
        y_true, y_prob, groups = sample_risk_data
        auditor = FairnessAuditor()
        results = auditor.audit(y_true, y_prob, groups)

        fm = results["fairness_metrics"]
        assert "demographic_parity" in fm
        assert "equalized_odds" in fm
        assert "calibration" in fm
        assert "passes_threshold" in fm["demographic_parity"]

    def test_to_dataframe(self, sample_risk_data):
        """Test conversion to DataFrame."""
        y_true, y_prob, groups = sample_risk_data
        auditor = FairnessAuditor()
        auditor.audit(y_true, y_prob, groups)

        df = auditor.to_dataframe()
        assert len(df) == len(np.unique(groups))


class TestBiasDetector:
    """Tests for BiasDetector."""

    def test_detect_bias(self, sample_risk_data):
        """Test bias detection."""
        y_true, y_prob, groups = sample_risk_data
        detector = BiasDetector()
        results = detector.detect_bias(
            y_true, y_prob,
            {"population": groups},
        )

        assert "single_attribute_analysis" in results
        assert "overall_bias_risk" in results
        assert results["overall_bias_risk"] in ["LOW", "MODERATE", "HIGH"]

    def test_intersectional_analysis(self, sample_risk_data):
        """Test intersectional bias analysis."""
        y_true, y_prob, groups = sample_risk_data
        rng = np.random.RandomState(42)
        sex = rng.choice(["M", "F"], size=len(y_true))

        detector = BiasDetector()
        results = detector.detect_bias(
            y_true, y_prob,
            {"population": groups, "sex": sex},
        )

        assert "intersectional_analysis" in results


class TestBiasMitigator:
    """Tests for BiasMitigator."""

    def test_optimize_thresholds(self, sample_risk_data):
        """Test threshold optimization."""
        y_true, y_prob, groups = sample_risk_data
        mitigator = BiasMitigator()
        thresholds = mitigator.optimize_thresholds(y_true, y_prob, groups)

        assert len(thresholds) == len(np.unique(groups))
        for group, threshold in thresholds.items():
            assert 0.1 <= threshold <= 0.9

    def test_sample_weights(self, sample_risk_data):
        """Test sample weight computation."""
        y_true, _, groups = sample_risk_data
        mitigator = BiasMitigator()
        weights = mitigator.compute_sample_weights(y_true, groups)

        assert len(weights) == len(y_true)
        assert all(w > 0 for w in weights)

    def test_apply_group_thresholds(self, sample_risk_data):
        """Test group-specific threshold application."""
        y_true, y_prob, groups = sample_risk_data
        mitigator = BiasMitigator()
        thresholds = mitigator.optimize_thresholds(y_true, y_prob, groups)
        predictions = mitigator.apply_group_thresholds(y_prob, groups, thresholds)

        assert len(predictions) == len(y_prob)
        assert set(predictions).issubset({0, 1})
