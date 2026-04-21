"""
Model Training Pipeline.

Orchestrates the full model training workflow:
1. Data loading and preprocessing
2. Feature engineering
3. Train/test split with population stratification
4. Model training and evaluation
5. Ensemble construction
6. Model serialization
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from genomicguard.config import DataConfig, ModelConfig, DISEASES, DATA_DIR, MODELS_DIR
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.data.preprocessor import GenomicPreprocessor
from genomicguard.data.feature_engineering import GenomicFeatureEngineer
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.models.variant_classifier import VariantClassifier
from genomicguard.models.ensemble import EnsembleModel


class TrainingPipeline:
    """
    End-to-end training pipeline for GenomicGuard models.

    Handles the full lifecycle from data loading through model
    serialization, ensuring reproducibility and population-aware
    train/test splitting.
    """

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.data_config = data_config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.preprocessor = GenomicPreprocessor()
        self.feature_engineer = GenomicFeatureEngineer()
        self.risk_scorers = {}
        self.variant_classifier = None
        self.ensemble = None
        self._dataset = None
        self._features = None
        self._results = {}

    def run(self, data_dir: Optional[Path] = None) -> Dict:
        """
        Execute the full training pipeline.

        Returns:
            Dictionary of all training results and metrics
        """
        print("=" * 70)
        print("  GenomicGuard Training Pipeline")
        print("=" * 70)

        # 1. Load data
        print("\n[1/6] Loading data...")
        self._load_data(data_dir)

        # 2. Preprocess
        print("\n[2/6] Preprocessing...")
        geno_processed, clinical_processed = self._preprocess()

        # 3. Feature engineering
        print("\n[3/6] Engineering features...")
        features = self._engineer_features(geno_processed, clinical_processed)

        # 4. Train risk scorers
        print("\n[4/6] Training risk scorers...")
        risk_results = self._train_risk_scorers(features)

        # 5. Train variant classifier
        print("\n[5/6] Training variant classifier...")
        variant_results = self._train_variant_classifier()

        # 6. Build ensemble
        print("\n[6/6] Building ensemble...")
        ensemble_results = self._train_ensemble(features, risk_results)

        # Save all artifacts
        print("\n" + "=" * 70)
        print("  Saving models and artifacts...")
        print("=" * 70)
        self._save_all()

        print("\n✓ Training pipeline complete!")
        return self._results

    def _load_data(self, data_dir: Optional[Path] = None):
        """Load dataset from disk."""
        self._dataset = GenomicDataGenerator.load_dataset(data_dir)
        for name, df in self._dataset.items():
            print(f"  ✓ Loaded {name}: {df.shape}")

    def _preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess genotype and clinical data."""
        geno_processed, clinical_processed = self.preprocessor.fit_transform(
            self._dataset["genotypes"],
            self._dataset["clinical"],
            self._dataset["sample_metadata"],
        )
        print(f"  ✓ Genotypes: {geno_processed.shape}")
        print(f"  ✓ Clinical: {clinical_processed.shape}")
        return geno_processed, clinical_processed

    def _engineer_features(
        self,
        geno_processed: pd.DataFrame,
        clinical_processed: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create ML-ready features."""
        features = self.feature_engineer.fit_transform(
            geno_processed,
            clinical_processed,
            self._dataset["snp_metadata"],
        )
        self._features = features
        print(f"  ✓ Feature matrix: {features.shape}")
        return features

    def _train_risk_scorers(self, features: pd.DataFrame) -> Dict:
        """Train risk scorers for each disease."""
        phenotypes = self._dataset["phenotypes"]
        sample_meta = self._dataset["sample_metadata"]
        results = {}

        for disease in DISEASES:
            disease_key = disease.lower().replace(" ", "_")
            label_col = f"{disease_key}_label"

            if label_col not in phenotypes.columns:
                print(f"  ⚠ Skipping {disease}: label column not found")
                continue

            y = phenotypes[label_col].values

            # Population-stratified train/test split
            strat_key = sample_meta["population"].astype(str) + "_" + y.astype(str)
            X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
                features, y, sample_meta,
                test_size=self.data_config.test_size,
                stratify=strat_key,
                random_state=self.data_config.random_seed,
            )

            # Train
            scorer = PolygenicRiskScorer(config=self.model_config, disease_name=disease_key)
            train_metrics = scorer.fit(X_train, y_train, calibrate=True)

            # Cross-validate
            cv_results = scorer.cross_validate(X_train, y_train, n_folds=5)

            # Test set evaluation
            y_test_prob = scorer.predict_risk(X_test)
            test_auc = roc_auc_score(y_test, y_test_prob)

            self.risk_scorers[disease_key] = scorer
            results[disease_key] = {
                "train_metrics": train_metrics,
                "cv_results": cv_results,
                "test_auc": float(test_auc),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }

            print(f"  ✓ {disease}: CV AUC = {cv_results['mean_auc_roc']:.3f} ± {cv_results['std_auc_roc']:.3f}, Test AUC = {test_auc:.3f}")

        self._results["risk_scorers"] = results
        return results

    def _train_variant_classifier(self) -> Dict:
        """Train the variant pathogenicity classifier."""
        snp_meta = self._dataset["snp_metadata"]

        self.variant_classifier = VariantClassifier(config=self.model_config)

        # Prepare features and labels
        X_variant = self.variant_classifier.prepare_variant_features(snp_meta)
        y_variant = self.variant_classifier.generate_variant_labels(snp_meta)

        # Train
        metrics = self.variant_classifier.fit(X_variant, y_variant)

        self._results["variant_classifier"] = metrics
        print(f"  ✓ Variant Classifier: Accuracy = {metrics['accuracy']:.3f}, Macro F1 = {metrics['macro_f1']:.3f}")
        return metrics

    def _train_ensemble(self, features: pd.DataFrame, risk_results: Dict) -> Dict:
        """Build the stacking ensemble."""
        phenotypes = self._dataset["phenotypes"]

        # Use the first disease for ensemble demo
        primary_disease = list(risk_results.keys())[0]
        label_col = f"{primary_disease}_label"
        y = phenotypes[label_col].values

        # Get base model predictions
        base_predictions = {}
        for disease_key, scorer in self.risk_scorers.items():
            base_predictions[f"risk_{disease_key}"] = scorer.predict_risk(features)

        self.ensemble = EnsembleModel(config=self.model_config)
        ensemble_metrics = self.ensemble.fit(base_predictions, y)

        self._results["ensemble"] = ensemble_metrics
        print(f"  ✓ Ensemble: AUC = {ensemble_metrics['auc_roc']:.3f}")
        return ensemble_metrics

    def _save_all(self):
        """Save all trained models and preprocessing artifacts."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save risk scorers
        for disease_key, scorer in self.risk_scorers.items():
            scorer.save()

        # Save variant classifier
        if self.variant_classifier:
            self.variant_classifier.save()

        # Save ensemble
        if self.ensemble:
            self.ensemble.save()

        # Save preprocessor and feature engineer
        joblib.dump(self.preprocessor, MODELS_DIR / "preprocessor.joblib")
        joblib.dump(self.feature_engineer, MODELS_DIR / "feature_engineer.joblib")

        # Save training results
        joblib.dump(self._results, MODELS_DIR / "training_results.joblib")
        print(f"  ✓ All artifacts saved to {MODELS_DIR}")
