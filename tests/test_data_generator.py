"""Tests for the synthetic data generator."""

import numpy as np
import pandas as pd
import pytest
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.config import DataConfig, POPULATIONS, DISEASES


class TestGenomicDataGenerator:
    """Tests for GenomicDataGenerator."""

    def test_generate_full_dataset(self, small_config):
        """Test that full dataset generation produces all expected components."""
        gen = GenomicDataGenerator(small_config)
        dataset = gen.generate_full_dataset()

        assert "genotypes" in dataset
        assert "clinical" in dataset
        assert "phenotypes" in dataset
        assert "snp_metadata" in dataset
        assert "sample_metadata" in dataset

    def test_correct_dimensions(self, small_dataset, small_config):
        """Test that dataset dimensions match configuration."""
        assert len(small_dataset["genotypes"]) == small_config.n_samples
        assert small_dataset["genotypes"].shape[1] == small_config.n_snps
        assert len(small_dataset["clinical"]) == small_config.n_samples
        assert len(small_dataset["phenotypes"]) == small_config.n_samples
        assert len(small_dataset["sample_metadata"]) == small_config.n_samples
        assert len(small_dataset["snp_metadata"]) == small_config.n_snps

    def test_genotype_values(self, small_dataset):
        """Test that genotypes contain only valid values (0, 1, 2)."""
        geno = small_dataset["genotypes"]
        unique_vals = set(geno.values.flatten())
        assert unique_vals.issubset({0, 1, 2})

    def test_populations_present(self, small_dataset):
        """Test that all configured populations are present."""
        pops = small_dataset["sample_metadata"]["population"].unique()
        for pop in POPULATIONS:
            assert pop in pops, f"Population {pop} not found"

    def test_clinical_features(self, small_dataset):
        """Test that expected clinical features are present."""
        clinical = small_dataset["clinical"]
        expected_cols = ["age", "bmi", "systolic_bp", "fasting_glucose", "hba1c"]
        for col in expected_cols:
            assert col in clinical.columns, f"Missing clinical feature: {col}"

    def test_clinical_ranges(self, small_dataset):
        """Test that clinical values are in reasonable ranges."""
        clinical = small_dataset["clinical"]
        assert clinical["age"].min() >= 18
        assert clinical["age"].max() <= 90
        assert clinical["bmi"].min() >= 15
        assert clinical["bmi"].max() <= 50

    def test_phenotype_labels(self, small_dataset):
        """Test that disease labels are binary."""
        phenotypes = small_dataset["phenotypes"]
        for disease in DISEASES:
            dk = disease.lower().replace(" ", "_")
            label_col = f"{dk}_label"
            assert label_col in phenotypes.columns
            assert set(phenotypes[label_col].unique()).issubset({0, 1})

    def test_risk_scores_range(self, small_dataset):
        """Test that risk scores are between 0 and 1."""
        phenotypes = small_dataset["phenotypes"]
        for col in phenotypes.columns:
            if col.endswith("_risk_score"):
                assert phenotypes[col].min() >= 0
                assert phenotypes[col].max() <= 1

    def test_reproducibility(self, small_config):
        """Test that same seed produces identical data."""
        gen1 = GenomicDataGenerator(small_config)
        gen2 = GenomicDataGenerator(small_config)
        d1 = gen1.generate_full_dataset()
        d2 = gen2.generate_full_dataset()

        pd.testing.assert_frame_equal(d1["genotypes"], d2["genotypes"])
        pd.testing.assert_frame_equal(d1["clinical"], d2["clinical"])

    def test_sex_distribution(self, small_dataset):
        """Test that sex distribution is roughly balanced."""
        sex = small_dataset["sample_metadata"]["sex"]
        assert set(sex.unique()) == {"M", "F"}
        ratio = (sex == "M").mean()
        assert 0.3 < ratio < 0.7  # Roughly balanced
