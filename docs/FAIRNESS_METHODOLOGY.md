# Fairness Methodology

## Overview

GenomicGuard implements a comprehensive fairness assessment framework designed specifically for clinical genomics models. The framework evaluates model performance across protected population groups and provides actionable mitigation strategies.

## Fairness Metrics

### 1. Demographic Parity

**Definition**: The positive prediction rate should be similar across all groups.

$$P(\hat{Y} = 1 | A = a) \approx P(\hat{Y} = 1 | A = b) \quad \forall a, b \in \text{groups}$$

**Threshold**: Maximum disparity ≤ 10%

**Clinical relevance**: Ensures that the model doesn't systematically screen certain populations at higher rates, which could lead to inequitable resource allocation.

### 2. Equalized Odds

**Definition**: True positive rates (sensitivity) and false positive rates should be similar across groups.

- **TPR Parity**: $P(\hat{Y} = 1 | Y = 1, A = a) \approx P(\hat{Y} = 1 | Y = 1, A = b)$
- **FPR Parity**: $P(\hat{Y} = 1 | Y = 0, A = a) \approx P(\hat{Y} = 1 | Y = 0, A = b)$

**Threshold**: Maximum TPR/FPR disparity ≤ 10%

**Clinical relevance**: Ensures the model is equally good at detecting disease (sensitivity) and avoiding false alarms across all populations.

### 3. Calibration

**Definition**: Predicted probabilities should match observed rates within each group.

$$E[Y | P = p, A = a] \approx p \quad \forall a \in \text{groups}$$

**Threshold**: Maximum calibration gap ≤ 10%

**Clinical relevance**: A 30% risk prediction should mean a ~30% actual risk, regardless of the patient's population group.

## Bias Detection Methodology

### Single-Attribute Analysis
Evaluates performance disparities for each protected attribute independently (population, sex, age group).

### Intersectional Analysis
Examines performance at the intersection of multiple attributes (e.g., African-ancestry females, South Asian seniors), which can reveal disparities hidden by single-attribute analysis.

### Severity Classification
- **Negligible**: Disparity < 5%
- **Low**: 5% ≤ Disparity < 10%
- **Moderate**: 10% ≤ Disparity < 20%
- **High**: Disparity ≥ 20%

## Mitigation Strategies

### 1. Group-Specific Thresholds
Instead of applying a single classification threshold, optimizes thresholds per group to equalize a chosen performance metric (F1, sensitivity, or specificity).

### 2. Sample Reweighting
Computes inverse-frequency weights to balance group representation during training, upweighting underrepresented groups and underrepresented class labels within groups.

### 3. Per-Group Recalibration
Applies Platt scaling or isotonic regression per population group to align predicted probabilities with observed rates.

## Limitations

1. **Synthetic data**: The fairness patterns in synthetic data may not fully capture real-world disparities
2. **Missing confounders**: Real clinical data has socioeconomic and environmental confounders not modeled here
3. **Group definitions**: Population categories are simplifications; real genetic ancestry is continuous
4. **Metric trade-offs**: Satisfying all fairness metrics simultaneously is generally impossible (impossibility theorem)
