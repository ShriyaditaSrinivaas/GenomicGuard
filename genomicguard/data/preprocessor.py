"""
Genomic Data Preprocessor.

Handles data cleaning, quality control, missing data imputation,
and standardization of genomic and clinical data for downstream ML.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class GenomicPreprocessor:
    """
    Preprocesses genomic and clinical data for model training.

    Includes:
    - Genotype quality control (MAF filtering, call rate filtering)
    - Population-aware imputation
    - Clinical feature standardization
    - Encoding of categorical variables
    """

    def __init__(self, maf_threshold: float = 0.01, call_rate_threshold: float = 0.95):
        self.maf_threshold = maf_threshold
        self.call_rate_threshold = call_rate_threshold
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False
        self._selected_snps = None
        self._clinical_columns = None

    def fit_transform(
        self,
        genotypes: pd.DataFrame,
        clinical: pd.DataFrame,
        sample_metadata: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit preprocessor and transform data.

        Args:
            genotypes: Raw genotype matrix (0, 1, 2 encoding)
            clinical: Clinical features
            sample_metadata: Sample-level metadata including population

        Returns:
            Tuple of (processed genotypes, processed clinical features)
        """
        # ── Genotype QC ────────────────────────────────────────────────────
        geno_clean = self._genotype_qc(genotypes)

        # ── Genotype imputation (mean imputation per population) ───────────
        geno_imputed = self._impute_genotypes(geno_clean, sample_metadata["population"])

        # ── Clinical preprocessing ─────────────────────────────────────────
        clinical_processed = self._process_clinical(clinical)

        self._fitted = True
        return geno_imputed, clinical_processed

    def transform(
        self,
        genotypes: pd.DataFrame,
        clinical: pd.DataFrame,
        sample_metadata: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform new data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit_transform first.")

        # Filter to selected SNPs
        available_snps = [s for s in self._selected_snps if s in genotypes.columns]
        geno_filtered = genotypes[available_snps]

        # Impute
        geno_imputed = self._impute_genotypes(geno_filtered, sample_metadata["population"])

        # Transform clinical
        clinical_processed = clinical[self._clinical_columns].copy()
        numeric_cols = clinical_processed.select_dtypes(include=[np.number]).columns
        clinical_processed[numeric_cols] = self.scaler.transform(clinical_processed[numeric_cols])

        return geno_imputed, clinical_processed

    def _genotype_qc(self, genotypes: pd.DataFrame) -> pd.DataFrame:
        """
        Quality control filtering for genotype data.

        Removes SNPs with:
        - Minor Allele Frequency (MAF) below threshold
        - Call rate below threshold
        """
        # Calculate MAF
        allele_freq = genotypes.mean(axis=0) / 2.0
        maf = np.minimum(allele_freq, 1 - allele_freq)

        # Calculate call rate (proportion of non-missing)
        call_rate = genotypes.notna().mean(axis=0)

        # Filter
        keep_mask = (maf >= self.maf_threshold) & (call_rate >= self.call_rate_threshold)
        self._selected_snps = genotypes.columns[keep_mask].tolist()

        filtered = genotypes[self._selected_snps]
        n_removed = len(genotypes.columns) - len(self._selected_snps)
        if n_removed > 0:
            print(f"  ⚠ Removed {n_removed} SNPs failing QC (MAF < {self.maf_threshold} or call rate < {self.call_rate_threshold})")

        return filtered

    def _impute_genotypes(
        self, genotypes: pd.DataFrame, populations: pd.Series
    ) -> pd.DataFrame:
        """
        Population-aware genotype imputation.

        Uses mean imputation within each population group, which is more
        appropriate than global imputation for diverse cohorts.
        """
        result = genotypes.copy()

        for pop in populations.unique():
            mask = populations.values == pop
            pop_data = result.loc[mask]

            if pop_data.isna().any().any():
                # Impute with population-specific means
                pop_means = pop_data.mean()
                result.loc[mask] = pop_data.fillna(pop_means)

        # Final fallback: global mean for any remaining NAs
        if result.isna().any().any():
            global_means = result.mean()
            result = result.fillna(global_means)

        return result

    def _process_clinical(self, clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Process clinical features: impute, encode, and standardize.
        """
        processed = clinical.copy()
        self._clinical_columns = processed.columns.tolist()

        # Impute numeric columns with median
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="median")
        processed[numeric_cols] = imputer.fit_transform(processed[numeric_cols])

        # Standardize numeric features
        processed[numeric_cols] = self.scaler.fit_transform(processed[numeric_cols])

        return processed

    def get_quality_report(self, genotypes: pd.DataFrame) -> Dict:
        """Generate a quality control summary report."""
        allele_freq = genotypes.mean(axis=0) / 2.0
        maf = np.minimum(allele_freq, 1 - allele_freq)
        call_rate = genotypes.notna().mean(axis=0)

        return {
            "total_snps": len(genotypes.columns),
            "passing_snps": len(self._selected_snps) if self._selected_snps else "N/A",
            "maf_stats": {
                "mean": float(maf.mean()),
                "median": float(maf.median()),
                "min": float(maf.min()),
                "max": float(maf.max()),
            },
            "call_rate_stats": {
                "mean": float(call_rate.mean()),
                "min": float(call_rate.min()),
            },
            "missing_rate": float(1 - call_rate.mean()),
        }
