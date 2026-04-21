"""Shared test fixtures for GenomicGuard tests."""

import pytest
import numpy as np
import pandas as pd

from genomicguard.config import DataConfig, ModelConfig
from genomicguard.data.generator import GenomicDataGenerator


@pytest.fixture(scope="session")
def small_config():
    """Small dataset config for fast testing."""
    return DataConfig(n_samples=200, n_snps=30, random_seed=42)


@pytest.fixture(scope="session")
def model_config():
    """Model config for testing."""
    return ModelConfig(
        gb_n_estimators=50,
        gb_max_depth=3,
        mlp_hidden_layers=(32, 16),
        mlp_max_iter=100,
        random_seed=42,
    )


@pytest.fixture(scope="session")
def small_dataset(small_config):
    """Generate a small synthetic dataset for testing."""
    generator = GenomicDataGenerator(small_config)
    return generator.generate_full_dataset()


@pytest.fixture
def sample_binary_data():
    """Simple binary classification data."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"feat_{i}" for i in range(5)])
    y = rng.randint(0, 2, size=n)
    groups = rng.choice(["A", "B", "C"], size=n)
    return X, y, groups


@pytest.fixture
def sample_risk_data():
    """Sample data with probabilities and groups."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.randint(0, 2, size=n)
    y_prob = rng.beta(2, 5, size=n)
    # Make probabilities somewhat correlated with labels
    y_prob[y_true == 1] += 0.3
    y_prob = np.clip(y_prob, 0, 1)
    groups = rng.choice(["EUR", "AFR", "EAS", "SAS", "AMR"], size=n)
    return y_true, y_prob, groups
