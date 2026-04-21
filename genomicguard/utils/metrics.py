"""
Custom evaluation metrics for clinical genomics models.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive suite of clinical metrics.

    Includes metrics that clinicians care about:
    - Sensitivity/Specificity
    - PPV/NPV (Positive/Negative Predictive Values)
    - Number Needed to Screen
    - Discrimination (AUC)
    - Calibration (Brier score)
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "prevalence": float(y_true.mean()),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Number Needed to Screen = 1 / PPV (clinically useful)
    metrics["nns"] = float(1 / metrics["ppv"]) if metrics["ppv"] > 0 else float("inf")

    return metrics


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute net benefit (decision curve analysis).

    Net Benefit = (TP/N) - (FP/N) × (threshold / (1 - threshold))
    """
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    odds = threshold / (1 - threshold) if threshold < 1 else float("inf")
    return float((tp / n) - (fp / n) * odds)
