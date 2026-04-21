"""
Pathogenic Variant Classifier.

MLP-based multi-class classifier for variant pathogenicity prediction.
Classifies variants into ClinVar categories: Benign, Likely Benign,
VUS, Likely Pathogenic, Pathogenic.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from genomicguard.config import ModelConfig, PATHOGENICITY_CLASSES, MODELS_DIR


class VariantClassifier:
    """
    Multi-class classifier for variant pathogenicity.

    Uses an MLP to classify genomic variants based on:
    - Conservation scores (cross-species)
    - CADD deleteriousness scores
    - Functional annotation
    - Allele frequency
    - Protein impact predictions

    Output: 5-class pathogenicity prediction matching ClinVar categories.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._feature_names = None
        self._is_fitted = False
        self._classes = PATHOGENICITY_CLASSES

    def prepare_variant_features(self, snp_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for variant classification from SNP metadata.

        Creates features that approximate those used in real variant
        pathogenicity predictors (REVEL, ClinPred, etc.).
        """
        features = pd.DataFrame()

        # Conservation score
        features["conservation_score"] = snp_metadata["conservation_score"].values

        # CADD score
        features["cadd_score"] = snp_metadata["cadd_score"].values

        # Absolute effect size (proxy for functional impact)
        features["abs_beta"] = np.abs(snp_metadata["beta"].values)

        # Squared effect (captures non-linearity)
        features["beta_squared"] = snp_metadata["beta"].values ** 2

        # Annotation encoding
        annotation_dummies = pd.get_dummies(
            snp_metadata["annotation"], prefix="annot"
        )
        features = pd.concat([features, annotation_dummies], axis=1)

        # Log-transformed CADD
        features["log_cadd"] = np.log1p(snp_metadata["cadd_score"].values)

        # Conservation × CADD interaction
        features["consv_cadd_interact"] = (
            snp_metadata["conservation_score"].values * snp_metadata["cadd_score"].values
        )

        # Chromosome encoding (numeric)
        chrom_map = {str(i): i for i in range(1, 23)}
        chrom_map["X"] = 23
        features["chromosome_num"] = snp_metadata["chromosome"].map(chrom_map).fillna(0).values

        return features

    def generate_variant_labels(
        self, snp_metadata: pd.DataFrame, seed: int = 42
    ) -> np.ndarray:
        """
        Generate synthetic pathogenicity labels.

        Uses a combination of conservation, CADD, and effect size
        to create plausible pathogenicity assignments.
        """
        rng = np.random.RandomState(seed)
        n = len(snp_metadata)

        # Compute a pathogenicity score
        path_score = (
            0.3 * snp_metadata["conservation_score"].values
            + 0.4 * (snp_metadata["cadd_score"].values / 40.0)
            + 0.3 * np.abs(snp_metadata["beta"].values) / (np.abs(snp_metadata["beta"].values).max() + 1e-8)
        )

        # Add noise
        path_score += rng.normal(0, 0.1, size=n)
        path_score = np.clip(path_score, 0, 1)

        # Assign classes based on score thresholds
        labels = np.empty(n, dtype=object)
        labels[path_score < 0.20] = "Benign"
        labels[(path_score >= 0.20) & (path_score < 0.35)] = "Likely Benign"
        labels[(path_score >= 0.35) & (path_score < 0.55)] = "VUS"
        labels[(path_score >= 0.55) & (path_score < 0.75)] = "Likely Pathogenic"
        labels[path_score >= 0.75] = "Pathogenic"

        return labels

    def fit(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Dict:
        """
        Train the variant classifier.

        Args:
            X: Variant feature DataFrame
            y: Pathogenicity labels (string categories)

        Returns:
            Training metrics dictionary
        """
        self._feature_names = list(X.columns)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X.values)

        # Train MLP
        self.model = MLPClassifier(
            hidden_layer_sizes=self.config.mlp_hidden_layers,
            activation=self.config.mlp_activation,
            max_iter=self.config.mlp_max_iter,
            learning_rate_init=self.config.mlp_learning_rate_init,
            early_stopping=self.config.mlp_early_stopping,
            validation_fraction=self.config.mlp_validation_fraction,
            random_state=self.config.random_seed,
            batch_size=min(64, len(X)),
        )
        self.model.fit(X_scaled, y_encoded)
        self._is_fitted = True

        # Metrics
        y_pred = self.model.predict(X_scaled)
        report = classification_report(
            y_encoded, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
        )

        return {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class": {
                cls: {
                    "precision": float(report[cls]["precision"]),
                    "recall": float(report[cls]["recall"]),
                    "f1": float(report[cls]["f1-score"]),
                }
                for cls in self.label_encoder.classes_
            },
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict pathogenicity class."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        X_scaled = self.scaler.transform(X.values)
        y_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict pathogenicity probabilities for all classes."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        X_scaled = self.scaler.transform(X.values)
        probs = self.model.predict_proba(X_scaled)
        return pd.DataFrame(probs, columns=self.label_encoder.classes_)

    def save(self, filepath: Optional[Path] = None):
        """Save model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / "variant_classifier.joblib"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self._feature_names,
            "classes": self._classes,
        }
        joblib.dump(model_data, filepath)
        print(f"  ✓ Saved variant classifier to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "VariantClassifier":
        """Load model from disk."""
        data = joblib.load(filepath)
        classifier = cls()
        classifier.model = data["model"]
        classifier.scaler = data["scaler"]
        classifier.label_encoder = data["label_encoder"]
        classifier._feature_names = data["feature_names"]
        classifier._classes = data["classes"]
        classifier._is_fitted = True
        return classifier
