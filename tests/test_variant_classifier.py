"""Tests for the variant classifier."""

import numpy as np
import pytest
from genomicguard.models.variant_classifier import VariantClassifier
from genomicguard.config import ModelConfig, PATHOGENICITY_CLASSES


class TestVariantClassifier:
    """Tests for VariantClassifier."""

    def test_prepare_features(self, small_dataset):
        """Test variant feature preparation."""
        classifier = VariantClassifier()
        features = classifier.prepare_variant_features(small_dataset["snp_metadata"])

        assert len(features) == len(small_dataset["snp_metadata"])
        assert "conservation_score" in features.columns
        assert "cadd_score" in features.columns
        assert features.isna().sum().sum() == 0

    def test_generate_labels(self, small_dataset):
        """Test label generation."""
        classifier = VariantClassifier()
        labels = classifier.generate_variant_labels(small_dataset["snp_metadata"])

        assert len(labels) == len(small_dataset["snp_metadata"])
        for label in labels:
            assert label in PATHOGENICITY_CLASSES

    def test_fit_and_predict(self, small_dataset, model_config):
        """Test fitting and prediction."""
        classifier = VariantClassifier(config=model_config)
        features = classifier.prepare_variant_features(small_dataset["snp_metadata"])
        labels = classifier.generate_variant_labels(small_dataset["snp_metadata"])

        metrics = classifier.fit(features, labels)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0

        predictions = classifier.predict(features)
        assert len(predictions) == len(features)
        for pred in predictions:
            assert pred in PATHOGENICITY_CLASSES

    def test_predict_proba(self, small_dataset, model_config):
        """Test probability predictions."""
        classifier = VariantClassifier(config=model_config)
        features = classifier.prepare_variant_features(small_dataset["snp_metadata"])
        labels = classifier.generate_variant_labels(small_dataset["snp_metadata"])
        classifier.fit(features, labels)

        probs = classifier.predict_proba(features)
        assert probs.shape[0] == len(features)
        # Probabilities should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1).values, 1.0, atol=1e-5)
