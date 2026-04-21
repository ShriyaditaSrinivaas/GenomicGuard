"""
Bias Mitigation Strategies.

Implements post-hoc and in-processing mitigation techniques
for reducing bias in genomic risk models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import calibration_curve

from genomicguard.config import FairnessConfig


class BiasMitigator:
    """
    Bias mitigation engine for genomic risk models.

    Implements:
    1. Group-specific threshold optimization
    2. Sample reweighting for training data
    3. Post-processing probability recalibration
    """

    def __init__(self, config: Optional[FairnessConfig] = None):
        self.config = config or FairnessConfig()
        self._optimal_thresholds = {}
        self._reweighting_factors = {}

    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
        metric: str = "f1",
    ) -> Dict[str, float]:
        """
        Find optimal classification thresholds per group.

        Instead of using a single threshold for all groups, finds
        group-specific thresholds that equalize a chosen metric.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            groups: Group assignments
            metric: Metric to optimize ('f1', 'sensitivity', 'specificity')

        Returns:
            Dict mapping group → optimal threshold
        """
        optimal_thresholds = {}

        for group in np.unique(groups):
            mask = groups == group
            yt = y_true[mask]
            yp = y_prob[mask]

            if len(np.unique(yt)) < 2:
                optimal_thresholds[str(group)] = 0.5
                continue

            best_threshold = 0.5
            best_score = 0.0

            for t in np.arange(0.1, 0.9, 0.01):
                yd = (yp >= t).astype(int)

                if metric == "f1":
                    score = f1_score(yt, yd, zero_division=0)
                elif metric == "sensitivity":
                    tp = ((yd == 1) & (yt == 1)).sum()
                    fn = ((yd == 0) & (yt == 1)).sum()
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0
                elif metric == "specificity":
                    tn = ((yd == 0) & (yt == 0)).sum()
                    fp = ((yd == 1) & (yt == 0)).sum()
                    score = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    score = f1_score(yt, yd, zero_division=0)

                if score > best_score:
                    best_score = score
                    best_threshold = t

            optimal_thresholds[str(group)] = round(best_threshold, 2)

        self._optimal_thresholds = optimal_thresholds
        return optimal_thresholds

    def compute_sample_weights(
        self,
        y_true: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sample weights to balance group representation.

        Uses inverse frequency weighting to up-weight underrepresented
        groups during training.
        """
        n = len(y_true)
        weights = np.ones(n)

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        for group in unique_groups:
            mask = groups == group

            # Within-group class balance
            for label in [0, 1]:
                label_mask = mask & (y_true == label)
                n_group_label = label_mask.sum()

                if n_group_label > 0:
                    # Weight = total / (n_groups * n_classes * n_group_label)
                    weight = n / (n_groups * 2 * n_group_label)
                    weights[label_mask] = weight

        # Normalize
        weights = weights / weights.mean()

        self._reweighting_factors = {
            str(g): float(weights[groups == g].mean())
            for g in unique_groups
        }

        return weights

    def recalibrate_probabilities(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
        method: str = "platt",
    ) -> np.ndarray:
        """
        Recalibrate probabilities per group.

        Adjusts predicted probabilities so they better match observed
        rates within each group (post-processing fairness).
        """
        recalibrated = y_prob.copy()

        for group in np.unique(groups):
            mask = groups == group
            yt = y_true[mask]
            yp = y_prob[mask]

            if len(np.unique(yt)) < 2:
                continue

            if method == "platt":
                # Platt scaling per group
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(max_iter=1000)
                lr.fit(yp.reshape(-1, 1), yt)
                recalibrated[mask] = lr.predict_proba(yp.reshape(-1, 1))[:, 1]

            elif method == "isotonic":
                from sklearn.isotonic import IsotonicRegression

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(yp, yt)
                recalibrated[mask] = iso.predict(yp)

        return recalibrated

    def apply_group_thresholds(
        self,
        y_prob: np.ndarray,
        groups: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Apply group-specific thresholds for prediction.

        Args:
            y_prob: Predicted probabilities
            groups: Group assignments
            thresholds: Group → threshold mapping (uses optimized if None)

        Returns:
            Binary predictions with group-specific thresholds
        """
        if thresholds is None:
            thresholds = self._optimal_thresholds

        y_pred = np.zeros(len(y_prob), dtype=int)

        for group in np.unique(groups):
            mask = groups == group
            t = thresholds.get(str(group), 0.5)
            y_pred[mask] = (y_prob[mask] >= t).astype(int)

        return y_pred

    def evaluate_mitigation(
        self,
        y_true: np.ndarray,
        y_prob_original: np.ndarray,
        y_prob_mitigated: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """
        Compare performance before and after mitigation.

        Returns detailed comparison of fairness metrics.
        """
        results = {"before": {}, "after": {}, "improvement": {}}

        for label, yp in [("before", y_prob_original), ("after", y_prob_mitigated)]:
            group_aucs = {}
            for g in np.unique(groups):
                mask = groups == g
                yt = y_true[mask]
                if len(np.unique(yt)) >= 2:
                    group_aucs[str(g)] = float(roc_auc_score(yt, yp[mask]))

            results[label] = {
                "overall_auc": float(roc_auc_score(y_true, yp)),
                "group_aucs": group_aucs,
                "auc_gap": float(max(group_aucs.values()) - min(group_aucs.values())) if group_aucs else 0,
            }

        # Compute improvement
        if results["before"]["auc_gap"] > 0:
            gap_reduction = results["before"]["auc_gap"] - results["after"]["auc_gap"]
            results["improvement"] = {
                "auc_gap_reduction": float(gap_reduction),
                "pct_reduction": float(gap_reduction / results["before"]["auc_gap"] * 100),
                "auc_change": float(results["after"]["overall_auc"] - results["before"]["overall_auc"]),
            }

        return results

    def get_mitigation_summary(self) -> Dict:
        """Return summary of mitigation configuration."""
        return {
            "optimal_thresholds": self._optimal_thresholds,
            "reweighting_factors": self._reweighting_factors,
        }
