"""
Genomic Feature Engineering.

Transforms raw genotype and clinical data into ML-ready features, including:
- Polygenic Risk Scores (PRS) from SNP weights
- Principal Component Analysis for ancestry adjustment
- Gene-gene interaction features
- Clinical risk factor composites
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA


class GenomicFeatureEngineer:
    """
    Engineers features from genomic and clinical data.

    Creates a unified feature matrix combining:
    1. Raw genotype features (SNP dosages)
    2. Polygenic Risk Scores (weighted sums)
    3. PCA components for population structure
    4. Clinical risk composites
    5. Gene-gene interaction features
    """

    def __init__(self, n_pca_components: int = 10, n_interaction_pairs: int = 20):
        self.n_pca_components = n_pca_components
        self.n_interaction_pairs = n_interaction_pairs
        self.pca = PCA(n_components=n_pca_components)
        self._fitted = False
        self._top_snp_pairs = None

    def fit_transform(
        self,
        genotypes: pd.DataFrame,
        clinical: pd.DataFrame,
        snp_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit feature engineering pipeline and transform data.

        Args:
            genotypes: Preprocessed genotype DataFrame
            clinical: Preprocessed clinical DataFrame
            snp_metadata: SNP annotations with effect sizes

        Returns:
            Combined feature DataFrame
        """
        features = {}

        # 1. PCA for population structure adjustment
        pca_features = self._compute_pca(genotypes, fit=True)
        for col in pca_features.columns:
            features[col] = pca_features[col].values

        # 2. Polygenic Risk Scores
        prs_features = self._compute_prs(genotypes, snp_metadata)
        for col in prs_features.columns:
            features[col] = prs_features[col].values

        # 3. Gene-gene interactions (top SNP pairs by variance)
        interaction_features = self._compute_interactions(genotypes, fit=True)
        for col in interaction_features.columns:
            features[col] = interaction_features[col].values

        # 4. Clinical composites
        composite_features = self._compute_clinical_composites(clinical)
        for col in composite_features.columns:
            features[col] = composite_features[col].values

        # 5. Raw clinical features
        for col in clinical.columns:
            features[f"clinical_{col}"] = clinical[col].values

        self._fitted = True
        return pd.DataFrame(features)

    def transform(
        self,
        genotypes: pd.DataFrame,
        clinical: pd.DataFrame,
        snp_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform new data using fitted feature engineer."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted first.")

        features = {}

        pca_features = self._compute_pca(genotypes, fit=False)
        for col in pca_features.columns:
            features[col] = pca_features[col].values

        prs_features = self._compute_prs(genotypes, snp_metadata)
        for col in prs_features.columns:
            features[col] = prs_features[col].values

        interaction_features = self._compute_interactions(genotypes, fit=False)
        for col in interaction_features.columns:
            features[col] = interaction_features[col].values

        composite_features = self._compute_clinical_composites(clinical)
        for col in composite_features.columns:
            features[col] = composite_features[col].values

        for col in clinical.columns:
            features[f"clinical_{col}"] = clinical[col].values

        return pd.DataFrame(features)

    def _compute_pca(self, genotypes: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Compute PCA components from genotype data.

        These components capture population structure and are critical
        for adjusting out ancestry-related confounding in risk models.
        """
        n_components = min(self.n_pca_components, genotypes.shape[1], genotypes.shape[0])
        if fit:
            self.pca = PCA(n_components=n_components)
            components = self.pca.fit_transform(genotypes.values.astype(np.float64))
        else:
            components = self.pca.transform(genotypes.values.astype(np.float64))

        columns = [f"pc_{i+1}" for i in range(n_components)]
        return pd.DataFrame(components, columns=columns)

    def _compute_prs(
        self, genotypes: pd.DataFrame, snp_metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute Polygenic Risk Scores.

        PRS = Σ (genotype_i × beta_i)

        Creates multiple PRS variants:
        - Full PRS (all SNPs)
        - Top-10% PRS (strongest effect SNPs only)
        - Annotation-weighted PRS (adjusted by functional impact)
        """
        betas = snp_metadata["beta"].values
        # Align betas with genotype columns
        n_snps = min(len(betas), genotypes.shape[1])
        betas_aligned = betas[:n_snps]
        geno_values = genotypes.iloc[:, :n_snps].values.astype(np.float64)

        # Full PRS
        full_prs = geno_values @ betas_aligned

        # Top-effect PRS (top 10% by absolute beta)
        top_k = max(1, n_snps // 10)
        top_idx = np.argsort(np.abs(betas_aligned))[-top_k:]
        top_prs = geno_values[:, top_idx] @ betas_aligned[top_idx]

        # CADD-weighted PRS
        cadd_scores = snp_metadata["cadd_score"].values[:n_snps]
        cadd_weights = cadd_scores / (cadd_scores.sum() + 1e-8)
        weighted_betas = betas_aligned * cadd_weights * n_snps  # scale back
        cadd_prs = geno_values @ weighted_betas

        return pd.DataFrame({
            "prs_full": full_prs,
            "prs_top_effects": top_prs,
            "prs_cadd_weighted": cadd_prs,
        })

    def _compute_interactions(
        self, genotypes: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Compute gene-gene interaction features.

        Selects top SNP pairs by variance of their product term
        and creates multiplicative interaction features.
        """
        n_snps = genotypes.shape[1]
        n_pairs = min(self.n_interaction_pairs, n_snps * (n_snps - 1) // 2)

        if fit:
            # Find top pairs by variance of product
            geno_vals = genotypes.values.astype(np.float64)
            variances = []
            pairs = []

            # Sample random pairs for efficiency
            n_candidates = min(200, n_snps * (n_snps - 1) // 2)
            rng = np.random.RandomState(42)
            for _ in range(n_candidates):
                i, j = rng.choice(n_snps, size=2, replace=False)
                product = geno_vals[:, i] * geno_vals[:, j]
                variances.append(product.var())
                pairs.append((i, j))

            # Keep top-variance pairs
            top_idx = np.argsort(variances)[-n_pairs:]
            self._top_snp_pairs = [pairs[i] for i in top_idx]

        interactions = {}
        geno_vals = genotypes.values.astype(np.float64)
        for k, (i, j) in enumerate(self._top_snp_pairs):
            if i < geno_vals.shape[1] and j < geno_vals.shape[1]:
                interactions[f"interact_{k}"] = geno_vals[:, i] * geno_vals[:, j]

        return pd.DataFrame(interactions)

    def _compute_clinical_composites(self, clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically meaningful composite risk scores.

        These composites reflect established medical risk indices:
        - Metabolic syndrome score
        - Cardiovascular risk composite
        - Lipid ratio
        """
        composites = {}

        # Metabolic syndrome composite (z-score sum of components)
        if all(c in clinical.columns for c in ["bmi", "fasting_glucose", "systolic_bp", "triglycerides", "hdl_cholesterol"]):
            composites["metabolic_composite"] = (
                clinical["bmi"].values
                + clinical["fasting_glucose"].values
                + clinical["systolic_bp"].values
                + clinical["triglycerides"].values
                - clinical["hdl_cholesterol"].values  # lower HDL = higher risk
            ) / 5.0

        # Cardiovascular composite
        if all(c in clinical.columns for c in ["systolic_bp", "ldl_cholesterol", "hdl_cholesterol", "smoking_status"]):
            composites["cv_composite"] = (
                clinical["systolic_bp"].values
                + clinical["ldl_cholesterol"].values
                - clinical["hdl_cholesterol"].values
                + clinical["smoking_status"].values
            ) / 4.0

        # Age-BMI interaction
        if all(c in clinical.columns for c in ["age", "bmi"]):
            composites["age_bmi_interaction"] = clinical["age"].values * clinical["bmi"].values

        return pd.DataFrame(composites)

    def get_feature_names(self) -> List[str]:
        """Return list of all engineered feature names."""
        if not self._fitted:
            return []
        return list(self._feature_names) if hasattr(self, '_feature_names') else []
