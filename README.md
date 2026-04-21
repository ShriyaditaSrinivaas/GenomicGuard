<div align="center">

# 🧬 GenomicGuard

### Interpretable Clinical Genomics Risk Assessment with Fairness Auditing

[![CI](https://github.com/ShriyaditaSrinivaas/GenomicGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/ShriyaditaSrinivaas/GenomicGuard/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Building AI systems for precision medicine that clinicians can trust, explain, and audit — across every population.*

</div>

---

## 🎯 Problem Statement

Current clinical workflows cannot combine genomic, phenotypic, and historical data at the speed and scale needed for accurate diagnosis of complex diseases. Polygenic Risk Scores (PRS) have shown promise, but two critical gaps remain:

1. **Interpretability**: Clinicians need to understand *why* a model produces a given risk score — not just the score itself
2. **Equity**: Models trained on predominantly European-ancestry cohorts often underperform for other populations, perpetuating healthcare disparities

**GenomicGuard** addresses both gaps by integrating SHAP-based interpretability and multi-dimensional fairness auditing directly into the clinical genomics pipeline.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Polygenic Risk Scoring** | Gradient Boosting models with calibrated probabilities for T2D, CAD, and Breast Cancer |
| 🧪 **Variant Classification** | MLP classifier for pathogenicity prediction (Benign → Pathogenic) |
| 🏗️ **Stacking Ensemble** | Meta-learner combining base model predictions with bootstrap confidence intervals |
| 🔍 **SHAP Interpretability** | Per-patient explanations, global feature importance, population-stratified analysis |
| ⚖️ **Fairness Auditing** | Demographic parity, equalized odds, calibration — with statistical testing |
| 🚨 **Bias Detection** | Automated intersectional bias scanning with severity classification |
| 🛠️ **Bias Mitigation** | Group-specific thresholds, sample reweighting, per-population recalibration |
| 📊 **Clinical Dashboard** | Interactive Streamlit UI with 4 pages for clinical exploration |
| 📝 **Clinical Reports** | Structured JSON reports with risk factors, recommendations, and transparency |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     GenomicGuard Pipeline                        │
├──────────┬──────────┬───────────────┬──────────────┬────────────┤
│  Data    │  Models  │Interpretability│  Fairness    │ Dashboard  │
│          │          │               │              │            │
│ Generate │ Risk     │ SHAP          │ Auditor      │ Patient    │
│ Preproc  │ Scorer   │ Explainer     │ Bias         │ Risk View  │
│ Feature  │ Variant  │ Feature       │ Detector     │ SHAP View  │
│ Engineer │ Classif. │ Importance    │ Mitigator    │ Fairness   │
│          │ Ensemble │ Clinical      │ Equity       │ Population │
│          │ Trainer  │ Reports       │ Reports      │ Analytics  │
└──────────┴──────────┴───────────────┴──────────────┴────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ShriyaditaSrinivaas/GenomicGuard.git
cd GenomicGuard

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# 1. Generate synthetic genomic data (2000 samples, 5 populations)
python scripts/generate_data.py

# 2. Train all models (risk scorers, variant classifier, ensemble)
python scripts/train_models.py

# 3. Run fairness audit across populations
python scripts/run_fairness_audit.py

# 4. Generate clinical reports for sample patients
python scripts/generate_reports.py

# 5. Launch the interactive dashboard
streamlit run dashboard/app.py
```

Or use the Makefile:

```bash
make install    # Install dependencies
make pipeline   # Run steps 1-3
make dashboard  # Launch dashboard
make test       # Run tests
```

## 📊 Results

### Model Performance

| Disease | CV AUC-ROC | Test AUC-ROC | Brier Score |
|---------|-----------|-------------|-------------|
| Type 2 Diabetes | 0.706 ± 0.040 | 0.720 | 0.092 |
| Coronary Artery Disease | 0.710 ± 0.025 | 0.708 | 0.063 |
| Breast Cancer | 0.593 ± 0.068 | 0.750 | 0.045 |
| **Ensemble** | — | **0.945** | — |

### Fairness Audit Summary

| Population | N | AUC | Sensitivity | Specificity | PPV |
|-----------|---|-----|-------------|-------------|-----|
| EUR (European) | 615 | 0.961 | — | 1.000 | 1.000 |
| AFR (African) | 474 | 0.935 | 0.198 | 1.000 | 1.000 |
| EAS (East Asian) | 410 | 0.935 | — | 1.000 | — |
| SAS (South Asian) | 304 | 0.876 | — | 1.000 | 1.000 |
| AMR (Admixed American) | 197 | 0.906 | 0.156 | 0.994 | 0.833 |

## 🗂️ Project Structure

```
GenomicGuard/
├── genomicguard/                  # Core Python package
│   ├── config.py                  # Configuration management
│   ├── data/
│   │   ├── generator.py           # Synthetic genomic data generation
│   │   ├── preprocessor.py        # QC, imputation, standardization
│   │   └── feature_engineering.py # PRS, PCA, interactions
│   ├── models/
│   │   ├── risk_scorer.py         # Gradient Boosting risk scorer
│   │   ├── variant_classifier.py  # MLP pathogenicity classifier
│   │   ├── ensemble.py            # Stacking ensemble
│   │   └── trainer.py             # Training pipeline
│   ├── interpretability/
│   │   ├── shap_explainer.py      # SHAP explanations
│   │   ├── feature_importance.py  # Permutation & clinical importance
│   │   └── clinical_report.py     # Clinical report generator
│   ├── fairness/
│   │   ├── auditor.py             # Fairness metrics engine
│   │   ├── bias_detector.py       # Automated bias detection
│   │   ├── mitigation.py          # Bias mitigation strategies
│   │   └── equity_report.py       # Equity report generator
│   └── utils/
│       ├── metrics.py             # Clinical evaluation metrics
│       └── visualization.py       # Plotting utilities
├── dashboard/                     # Streamlit dashboard
│   ├── app.py                     # Main app
│   └── pages/
│       ├── 1_Patient_Risk_Profile.py
│       ├── 2_Model_Interpretability.py
│       ├── 3_Fairness_Audit.py
│       └── 4_Population_Analytics.py
├── scripts/                       # Pipeline scripts
│   ├── generate_data.py
│   ├── train_models.py
│   ├── run_fairness_audit.py
│   └── generate_reports.py
├── tests/                         # Test suite (46 tests)
│   ├── test_data_generator.py
│   ├── test_preprocessor.py
│   ├── test_risk_scorer.py
│   ├── test_variant_classifier.py
│   ├── test_fairness_auditor.py
│   ├── test_shap_explainer.py
│   └── test_integration.py
├── docs/                          # Documentation
├── .github/workflows/ci.yml       # CI/CD pipeline
├── pyproject.toml                 # Python project config
├── requirements.txt               # Dependencies
└── Makefile                       # Convenience commands
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=genomicguard --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_fairness_auditor.py -v
```

## 🔬 Technical Details

### Data Pipeline

- **Synthetic Data**: Generates biologically plausible genomic data using the Balding-Nichols model for population-specific allele frequencies, with realistic clinical feature distributions
- **Population Stratification**: 5 superpopulations (EUR, AFR, EAS, SAS, AMR) with Fst-based allele frequency differentiation
- **Disease Modeling**: Combines polygenic risk (SNP effect sizes) with clinical risk factors and population-specific baselines

### Interpretability

- **SHAP TreeExplainer**: Exact Shapley values for the Gradient Boosting model
- **Per-Patient Waterfall**: Top contributing factors for each individual patient
- **Population Heatmap**: Reveals whether the model relies on different features for different populations — a critical fairness signal

### Fairness Framework

- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Calibration**: Predicted probabilities match observed rates per group
- **Kruskal-Wallis Testing**: Statistical significance of inter-group differences
- **Intersectional Analysis**: Bias detection across combinations of protected attributes

## 📚 Clinical Context

This project addresses the challenge highlighted by recent research: PRS models trained on European-ancestry cohorts transfer poorly to other populations, with prediction accuracy dropping by 50-80% for non-European groups (Martin et al., 2019; Sirugo et al., 2019). GenomicGuard demonstrates how fairness-aware ML can help bridge this gap.

**Disclaimer**: This is a research demonstration using synthetic data. It is not intended for clinical use. All medical decisions should be made by qualified healthcare providers.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Shriyadita Srinivaas**

---

<div align="center">
<i>Built for equitable precision medicine 🧬</i>
</div>
