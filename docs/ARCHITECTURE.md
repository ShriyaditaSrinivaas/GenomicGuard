# Architecture Documentation

## System Components

### 1. Data Pipeline (`genomicguard/data/`)

**Generator** (`generator.py`): Creates synthetic genomic data mimicking real cohort studies.

- Uses the **Balding-Nichols model** to generate population-specific allele frequencies
- Fst values calibrated from 1000 Genomes Project data
- Clinical features follow epidemiologically accurate distributions
- Disease labels computed from combined genetic + clinical risk models

**Preprocessor** (`preprocessor.py`): Quality control and data cleaning.

- **MAF filtering**: Removes rare variants (MAF < 0.01)
- **Call rate filtering**: Removes poorly-genotyped SNPs
- **Population-aware imputation**: Uses within-population mean imputation
- **Standardization**: Z-score normalization for clinical features

**Feature Engineer** (`feature_engineering.py`): Transforms raw data into ML-ready features.

- **PCA components**: Captures population structure (ancestry adjustment)
- **Polygenic Risk Scores**: Full PRS, top-effect PRS, CADD-weighted PRS
- **Gene-gene interactions**: Multiplicative interaction features
- **Clinical composites**: Metabolic syndrome score, cardiovascular composite

### 2. Models (`genomicguard/models/`)

**Risk Scorer**: Gradient Boosting with probability calibration (Platt scaling).

- Stratified cross-validation for robust performance estimation
- Feature importance tracking for interpretability
- Serialization with joblib

**Variant Classifier**: MLP for 5-class pathogenicity prediction.

- Feature engineering from conservation, CADD, and annotation
- ClinVar-compatible output categories

**Ensemble**: Stacking meta-learner with calibrated logistic regression.

- Prevents information leakage through out-of-fold predictions
- Bootstrap confidence intervals for uncertainty quantification

### 3. Interpretability (`genomicguard/interpretability/`)

**SHAP Explainer**: TreeExplainer for exact Shapley values.

- Per-patient waterfall explanations
- Global mean |SHAP| importance ranking
- Population-stratified importance (fairness signal)

**Clinical Reports**: Structured JSON reports matching clinical workflows.

- Risk summary with confidence intervals
- Contributing factor analysis (genetic vs. clinical)
- Actionable recommendations

### 4. Fairness (`genomicguard/fairness/`)

**Auditor**: Computes fairness metrics per protected group.

- Demographic Parity, Equalized Odds, Calibration
- Per-group AUC, sensitivity, specificity, PPV, NPV
- Statistical significance testing (Kruskal-Wallis)

**Bias Detector**: Automated bias scanning with intersectionality.

- Single-attribute analysis
- Intersectional analysis (population × sex, population × age)
- Underperforming subgroup identification

**Mitigator**: Post-hoc and in-processing bias mitigation.

- Group-specific threshold optimization
- Sample reweighting for balanced training
- Per-group probability recalibration

### 5. Dashboard (`dashboard/`)

Four-page Streamlit application:

1. **Patient Risk Profile**: Individual risk gauges, SHAP waterfall, clinical values
2. **Model Interpretability**: Global importance, category breakdown, population heatmap
3. **Fairness Audit**: Performance comparison, fairness checks, bias findings, mitigation
4. **Population Analytics**: Distributions, demographics, prevalence, PCA plot

## Design Decisions

1. **scikit-learn over PyTorch**: Lighter dependencies, faster training, better interpretability support via SHAP TreeExplainer
2. **Synthetic data**: Enables public sharing without HIPAA concerns; mimics real-world patterns via Balding-Nichols model
3. **Calibrated probabilities**: Clinical use requires well-calibrated risk estimates, not just discriminative accuracy
4. **Post-processing fairness**: More practical for clinical deployment; doesn't require model retraining
