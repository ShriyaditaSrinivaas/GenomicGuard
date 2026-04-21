"""
Configuration management for GenomicGuard.

Centralizes all project paths, model hyperparameters, and genomic constants.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Project Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Genomic Constants ──────────────────────────────────────────────────────────

# Superpopulation codes (1000 Genomes Project convention)
POPULATIONS = ["EUR", "AFR", "EAS", "SAS", "AMR"]

POPULATION_LABELS = {
    "EUR": "European",
    "AFR": "African",
    "EAS": "East Asian",
    "SAS": "South Asian",
    "AMR": "Admixed American",
}

# Disease phenotypes modeled
DISEASES = [
    "Type 2 Diabetes",
    "Coronary Artery Disease",
    "Breast Cancer",
]

# Variant pathogenicity classes (ClinVar convention)
PATHOGENICITY_CLASSES = [
    "Benign",
    "Likely Benign",
    "VUS",
    "Likely Pathogenic",
    "Pathogenic",
]

# Chromosomes
CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X"]


# ── Data Generation Config ─────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""

    n_samples: int = 2000
    n_snps: int = 150  # Number of SNPs per disease
    n_clinical_features: int = 12
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1

    # Population distribution (realistic global proportions)
    population_weights: Dict[str, float] = field(default_factory=lambda: {
        "EUR": 0.30,
        "AFR": 0.25,
        "EAS": 0.20,
        "SAS": 0.15,
        "AMR": 0.10,
    })

    # Disease prevalence (approximate real-world rates)
    disease_prevalence: Dict[str, float] = field(default_factory=lambda: {
        "Type 2 Diabetes": 0.12,
        "Coronary Artery Disease": 0.08,
        "Breast Cancer": 0.05,
    })


# ── Model Hyperparameters ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # Gradient Boosting (Risk Scorer)
    gb_n_estimators: int = 200
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.05
    gb_min_samples_split: int = 10
    gb_subsample: float = 0.8

    # MLP (Variant Classifier)
    mlp_hidden_layers: Tuple[int, ...] = (128, 64, 32)
    mlp_activation: str = "relu"
    mlp_max_iter: int = 500
    mlp_learning_rate_init: float = 0.001
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1

    # Ensemble
    ensemble_meta_model: str = "logistic_regression"
    ensemble_cv_folds: int = 5

    # General
    random_seed: int = 42
    n_jobs: int = -1


# ── Fairness Config ────────────────────────────────────────────────────────────

@dataclass
class FairnessConfig:
    """Configuration for fairness auditing."""

    # Protected attributes
    protected_attributes: List[str] = field(default_factory=lambda: [
        "population", "sex", "age_group"
    ])

    # Fairness metric thresholds
    demographic_parity_threshold: float = 0.10  # max allowed disparity
    equalized_odds_threshold: float = 0.10
    calibration_threshold: float = 0.10

    # Statistical significance
    significance_level: float = 0.05

    # Bias severity levels
    bias_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.05,
        "moderate": 0.10,
        "high": 0.20,
    })


# ── Dashboard Config ───────────────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    """Configuration for the Streamlit dashboard."""

    page_title: str = "GenomicGuard"
    page_icon: str = "🧬"
    layout: str = "wide"

    # Risk level thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3,
        "moderate": 0.6,
        "high": 0.8,
        "very_high": 1.0,
    })

    risk_colors: Dict[str, str] = field(default_factory=lambda: {
        "low": "#22c55e",
        "moderate": "#f59e0b",
        "high": "#ef4444",
        "very_high": "#991b1b",
    })

    # Population colors
    population_colors: Dict[str, str] = field(default_factory=lambda: {
        "EUR": "#6366f1",
        "AFR": "#f59e0b",
        "EAS": "#10b981",
        "SAS": "#ef4444",
        "AMR": "#8b5cf6",
    })
