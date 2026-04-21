"""Integration tests for the full GenomicGuard pipeline."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from genomicguard.config import DataConfig, ModelConfig
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.data.preprocessor import GenomicPreprocessor
from genomicguard.data.feature_engineering import GenomicFeatureEngineer
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.models.variant_classifier import VariantClassifier
from genomicguard.models.ensemble import EnsembleModel
from genomicguard.fairness.auditor import FairnessAuditor
from genomicguard.interpretability.shap_explainer import SHAPExplainer
from genomicguard.interpretability.clinical_report import ClinicalReportGenerator


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    @pytest.fixture(scope="class")
    def pipeline_data(self):
        """Run the full pipeline and return results."""
        config = DataConfig(n_samples=150, n_snps=20, random_seed=42)
        model_config = ModelConfig(
            gb_n_estimators=30, gb_max_depth=3,
            mlp_hidden_layers=(16, 8), mlp_max_iter=50,
            random_seed=42,
        )

        # Generate data
        gen = GenomicDataGenerator(config)
        dataset = gen.generate_full_dataset()

        # Preprocess
        preprocessor = GenomicPreprocessor()
        geno, clinical = preprocessor.fit_transform(
            dataset["genotypes"], dataset["clinical"], dataset["sample_metadata"]
        )

        # Feature engineering
        fe = GenomicFeatureEngineer(n_pca_components=5, n_interaction_pairs=5)
        features = fe.fit_transform(geno, clinical, dataset["snp_metadata"])

        # Train risk scorer
        disease_key = "type_2_diabetes"
        y = dataset["phenotypes"][f"{disease_key}_label"].values
        scorer = PolygenicRiskScorer(config=model_config, disease_name=disease_key)
        scorer.fit(features, y, calibrate=True)

        # Train variant classifier
        vc = VariantClassifier(config=model_config)
        vc_features = vc.prepare_variant_features(dataset["snp_metadata"])
        vc_labels = vc.generate_variant_labels(dataset["snp_metadata"])
        vc.fit(vc_features, vc_labels)

        return {
            "dataset": dataset,
            "features": features,
            "scorer": scorer,
            "variant_classifier": vc,
            "y": y,
            "disease_key": disease_key,
        }

    def test_data_generation(self, pipeline_data):
        """Test that data generation produces valid output."""
        dataset = pipeline_data["dataset"]
        assert dataset["genotypes"].shape[0] == 150
        assert dataset["genotypes"].shape[1] == 20

    def test_feature_engineering(self, pipeline_data):
        """Test that feature engineering produces valid features."""
        features = pipeline_data["features"]
        assert features.shape[0] == 150
        assert features.shape[1] > 20  # Should have more features from engineering
        assert not features.isna().any().any()

    def test_risk_scorer_performance(self, pipeline_data):
        """Test that risk scorer achieves reasonable performance."""
        scorer = pipeline_data["scorer"]
        features = pipeline_data["features"]
        y = pipeline_data["y"]

        risk = scorer.predict_risk(features)
        assert all(0 <= r <= 1 for r in risk)

        # AUC should be better than random (0.5)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, risk)
        assert auc > 0.5, f"AUC ({auc:.3f}) should be > 0.5"

    def test_variant_classifier_performance(self, pipeline_data):
        """Test variant classifier output."""
        vc = pipeline_data["variant_classifier"]
        dataset = pipeline_data["dataset"]
        features = vc.prepare_variant_features(dataset["snp_metadata"])

        preds = vc.predict(features)
        assert len(preds) == len(dataset["snp_metadata"])

    def test_ensemble(self, pipeline_data):
        """Test ensemble model."""
        scorer = pipeline_data["scorer"]
        features = pipeline_data["features"]
        y = pipeline_data["y"]

        risk_preds = scorer.predict_risk(features)
        base_predictions = {"risk_scorer": risk_preds}

        ensemble = EnsembleModel()
        metrics = ensemble.fit(base_predictions, y)
        assert "auc_roc" in metrics

        risk = ensemble.predict_risk(base_predictions)
        assert len(risk) == len(features)

    def test_fairness_audit(self, pipeline_data):
        """Test fairness audit on model predictions."""
        scorer = pipeline_data["scorer"]
        features = pipeline_data["features"]
        y = pipeline_data["y"]
        populations = pipeline_data["dataset"]["sample_metadata"]["population"].values

        risk = scorer.predict_risk(features)
        auditor = FairnessAuditor()
        results = auditor.audit(y, risk, populations)

        assert "fairness_summary" in results
        assert results["fairness_summary"]["overall_assessment"] in ["PASS", "NEEDS ATTENTION"]

    def test_shap_explanations(self, pipeline_data):
        """Test SHAP explanations."""
        scorer = pipeline_data["scorer"]
        features = pipeline_data["features"]

        explainer = SHAPExplainer(scorer.model, list(features.columns))
        shap_vals = explainer.compute_shap_values(features)
        assert shap_vals.shape == features.shape

        explanation = explainer.get_patient_explanation(0, features)
        assert "top_risk_factors" in explanation

    def test_clinical_report(self, pipeline_data):
        """Test clinical report generation."""
        scorer = pipeline_data["scorer"]
        features = pipeline_data["features"]

        risk = scorer.predict_risk(features.iloc[[0]])[0]

        explainer = SHAPExplainer(scorer.model, list(features.columns))
        explainer.compute_shap_values(features)
        explanation = explainer.get_patient_explanation(0, features)

        report_gen = ClinicalReportGenerator()
        report = report_gen.generate_patient_report(
            patient_id="TEST_001",
            risk_scores={pipeline_data["disease_key"]: risk},
            shap_explanation=explanation,
            patient_features=features.iloc[0].to_dict(),
        )

        assert "risk_summary" in report
        assert "contributing_factors" in report
        assert "recommendations" in report
        assert report["patient_id"] == "TEST_001"

    def test_save_load_roundtrip(self, pipeline_data):
        """Test full save/load roundtrip."""
        scorer = pipeline_data["scorer"]
        features = pipeline_data["features"]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "scorer.joblib"
            scorer.save(filepath)

            loaded = PolygenicRiskScorer.load(filepath)
            original = scorer.predict_risk(features)
            reloaded = loaded.predict_risk(features)
            np.testing.assert_allclose(original, reloaded, atol=1e-6)
