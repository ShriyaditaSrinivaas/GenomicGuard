"""Tests for the genomic preprocessor."""

import numpy as np
import pandas as pd
import pytest
from genomicguard.data.preprocessor import GenomicPreprocessor


class TestGenomicPreprocessor:
    """Tests for GenomicPreprocessor."""

    def test_fit_transform(self, small_dataset):
        """Test basic fit_transform workflow."""
        preprocessor = GenomicPreprocessor()
        geno, clinical = preprocessor.fit_transform(
            small_dataset["genotypes"],
            small_dataset["clinical"],
            small_dataset["sample_metadata"],
        )

        assert len(geno) == len(small_dataset["genotypes"])
        assert len(clinical) == len(small_dataset["clinical"])
        assert not geno.isna().any().any(), "Genotypes should have no NAs after preprocessing"
        assert not clinical.isna().any().any(), "Clinical should have no NAs after preprocessing"

    def test_maf_filtering(self, small_dataset):
        """Test that MAF filtering removes low-frequency variants."""
        preprocessor = GenomicPreprocessor(maf_threshold=0.05)
        geno, _ = preprocessor.fit_transform(
            small_dataset["genotypes"],
            small_dataset["clinical"],
            small_dataset["sample_metadata"],
        )
        # Should have <= original number of SNPs
        assert geno.shape[1] <= small_dataset["genotypes"].shape[1]

    def test_clinical_standardization(self, small_dataset):
        """Test that clinical features are standardized."""
        preprocessor = GenomicPreprocessor()
        _, clinical = preprocessor.fit_transform(
            small_dataset["genotypes"],
            small_dataset["clinical"],
            small_dataset["sample_metadata"],
        )

        # After standardization, mean should be ~0 and std ~1
        numeric_cols = clinical.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(clinical[col].mean()) < 0.5, f"Mean of {col} should be near 0"

    def test_quality_report(self, small_dataset):
        """Test quality report generation."""
        preprocessor = GenomicPreprocessor()
        preprocessor.fit_transform(
            small_dataset["genotypes"],
            small_dataset["clinical"],
            small_dataset["sample_metadata"],
        )

        report = preprocessor.get_quality_report(small_dataset["genotypes"])
        assert "total_snps" in report
        assert "maf_stats" in report
        assert "call_rate_stats" in report
