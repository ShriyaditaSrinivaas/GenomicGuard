"""
Fairness Auditor.

Computes comprehensive fairness metrics across protected groups,
including demographic parity, equalized odds, and calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from scipy import stats

from genomicguard.config import FairnessConfig, POPULATIONS, POPULATION_LABELS


class FairnessAuditor:
    """
    Comprehensive fairness auditing engine.

    Computes and reports fairness metrics across population groups:
    - Performance metrics per group (AUC, sensitivity, specificity, PPV, NPV)
    - Demographic parity (equal positive prediction rates)
    - Equalized odds (equal TPR and FPR)
    - Calibration (predicted probabilities match observed rates)
    - Statistical significance of performance gaps
    """

    def __init__(self, config: Optional[FairnessConfig] = None):
        self.config = config or FairnessConfig()
        self._audit_results = None

    def audit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
        threshold: float = 0.5,
        group_name: str = "population",
    ) -> Dict:
        """
        Run a comprehensive fairness audit.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities (positive class)
            groups: Group assignments (e.g., population codes)
            threshold: Classification threshold
            group_name: Name of the protected attribute

        Returns:
            Comprehensive audit results dictionary
        """
        y_pred = (y_prob >= threshold).astype(int)

        # Per-group metrics
        group_metrics = self._compute_group_metrics(y_true, y_prob, y_pred, groups)

        # Fairness metrics
        fairness_metrics = self._compute_fairness_metrics(y_true, y_prob, y_pred, groups)

        # Statistical tests
        statistical_tests = self._run_statistical_tests(y_true, y_prob, groups)

        # Overall summary
        overall_metrics = self._compute_overall_metrics(y_true, y_prob, y_pred)

        self._audit_results = {
            "group_name": group_name,
            "n_groups": len(np.unique(groups)),
            "groups": list(np.unique(groups)),
            "threshold": threshold,
            "n_samples": len(y_true),
            "overall_metrics": overall_metrics,
            "group_metrics": group_metrics,
            "fairness_metrics": fairness_metrics,
            "statistical_tests": statistical_tests,
            "fairness_summary": self._summarize_fairness(fairness_metrics),
        }

        return self._audit_results

    def _compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, Dict]:
        """Compute performance metrics for each group."""
        results = {}

        for group in np.unique(groups):
            mask = groups == group
            yt = y_true[mask]
            yp = y_prob[mask]
            yd = y_pred[mask]

            n = mask.sum()
            n_pos = int(yt.sum())
            n_neg = int(n - n_pos)

            metrics = {
                "n_samples": int(n),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "prevalence": float(yt.mean()),
                "positive_prediction_rate": float(yd.mean()),
            }

            if len(np.unique(yt)) >= 2:
                metrics["auc_roc"] = float(roc_auc_score(yt, yp))
                metrics["auc_pr"] = float(average_precision_score(yt, yp))
                metrics["brier_score"] = float(brier_score_loss(yt, yp))

                if n_pos > 0 and n_neg > 0:
                    tn, fp, fn, tp = confusion_matrix(yt, yd, labels=[0, 1]).ravel()
                    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                    metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
                    metrics["f1"] = float(f1_score(yt, yd, zero_division=0))
            else:
                metrics["auc_roc"] = None
                metrics["note"] = "Only one class present in this group"

            group_label = POPULATION_LABELS.get(group, group)
            results[group] = {**metrics, "label": group_label}

        return results

    def _compute_overall_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """Compute overall (ungrouped) metrics."""
        return {
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "auc_pr": float(average_precision_score(y_true, y_prob)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "prevalence": float(y_true.mean()),
            "positive_prediction_rate": float(y_pred.mean()),
            "n_samples": int(len(y_true)),
        }

    def _compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """Compute key fairness metrics."""
        unique_groups = np.unique(groups)

        # Demographic Parity: Equal positive prediction rates
        pred_rates = {}
        for g in unique_groups:
            mask = groups == g
            pred_rates[g] = float(y_pred[mask].mean())

        dp_disparity = max(pred_rates.values()) - min(pred_rates.values())

        # Equalized Odds: Equal TPR and FPR
        tpr_by_group = {}
        fpr_by_group = {}
        for g in unique_groups:
            mask = groups == g
            yt, yd = y_true[mask], y_pred[mask]
            if len(np.unique(yt)) >= 2:
                tn, fp, fn, tp = confusion_matrix(yt, yd, labels=[0, 1]).ravel()
                tpr_by_group[g] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                fpr_by_group[g] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        eo_tpr_disparity = (max(tpr_by_group.values()) - min(tpr_by_group.values())) if tpr_by_group else 0.0
        eo_fpr_disparity = (max(fpr_by_group.values()) - min(fpr_by_group.values())) if fpr_by_group else 0.0

        # Calibration: Do predicted probabilities match observed rates?
        calibration = {}
        for g in unique_groups:
            mask = groups == g
            yt, yp = y_true[mask], y_prob[mask]
            # Simple calibration: mean predicted vs mean actual
            calibration[g] = {
                "mean_predicted": float(yp.mean()),
                "mean_actual": float(yt.mean()),
                "calibration_gap": float(abs(yp.mean() - yt.mean())),
            }

        max_cal_gap = max(c["calibration_gap"] for c in calibration.values())

        # AUC disparity
        auc_by_group = {}
        for g in unique_groups:
            mask = groups == g
            yt, yp = y_true[mask], y_prob[mask]
            if len(np.unique(yt)) >= 2:
                auc_by_group[g] = float(roc_auc_score(yt, yp))

        auc_disparity = (max(auc_by_group.values()) - min(auc_by_group.values())) if auc_by_group else 0.0

        return {
            "demographic_parity": {
                "prediction_rates": pred_rates,
                "max_disparity": float(dp_disparity),
                "passes_threshold": dp_disparity <= self.config.demographic_parity_threshold,
            },
            "equalized_odds": {
                "tpr_by_group": tpr_by_group,
                "fpr_by_group": fpr_by_group,
                "tpr_disparity": float(eo_tpr_disparity),
                "fpr_disparity": float(eo_fpr_disparity),
                "passes_threshold": (
                    eo_tpr_disparity <= self.config.equalized_odds_threshold
                    and eo_fpr_disparity <= self.config.equalized_odds_threshold
                ),
            },
            "calibration": {
                "per_group": calibration,
                "max_gap": float(max_cal_gap),
                "passes_threshold": max_cal_gap <= self.config.calibration_threshold,
            },
            "auc_parity": {
                "auc_by_group": auc_by_group,
                "max_disparity": float(auc_disparity),
            },
        }

    def _run_statistical_tests(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
    ) -> Dict:
        """Run statistical tests for performance differences."""
        unique_groups = np.unique(groups)

        # Chi-squared test for prediction rate differences
        pred_by_group = []
        for g in unique_groups:
            mask = groups == g
            pred_by_group.append(y_prob[mask])

        # Kruskal-Wallis test (non-parametric) for risk score differences
        if len(pred_by_group) >= 2:
            try:
                h_stat, p_value = stats.kruskal(*pred_by_group)
                kw_result = {
                    "test": "Kruskal-Wallis",
                    "h_statistic": float(h_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.config.significance_level,
                    "interpretation": (
                        "Significant differences in risk score distributions across groups"
                        if p_value < self.config.significance_level
                        else "No significant differences in risk score distributions"
                    ),
                }
            except Exception:
                kw_result = {"test": "Kruskal-Wallis", "error": "Insufficient data"}
        else:
            kw_result = {"test": "Kruskal-Wallis", "error": "Need at least 2 groups"}

        return {"kruskal_wallis": kw_result}

    def _summarize_fairness(self, fairness_metrics: Dict) -> Dict:
        """Generate a human-readable fairness summary."""
        issues = []
        passed = []

        # Check demographic parity
        dp = fairness_metrics["demographic_parity"]
        if dp["passes_threshold"]:
            passed.append("Demographic Parity")
        else:
            issues.append({
                "metric": "Demographic Parity",
                "severity": self._classify_severity(dp["max_disparity"]),
                "disparity": dp["max_disparity"],
                "description": f"Positive prediction rate varies by {dp['max_disparity']:.1%} across groups.",
            })

        # Check equalized odds
        eo = fairness_metrics["equalized_odds"]
        if eo["passes_threshold"]:
            passed.append("Equalized Odds")
        else:
            issues.append({
                "metric": "Equalized Odds",
                "severity": self._classify_severity(max(eo["tpr_disparity"], eo["fpr_disparity"])),
                "tpr_disparity": eo["tpr_disparity"],
                "fpr_disparity": eo["fpr_disparity"],
                "description": f"TPR disparity: {eo['tpr_disparity']:.1%}, FPR disparity: {eo['fpr_disparity']:.1%}.",
            })

        # Check calibration
        cal = fairness_metrics["calibration"]
        if cal["passes_threshold"]:
            passed.append("Calibration")
        else:
            issues.append({
                "metric": "Calibration",
                "severity": self._classify_severity(cal["max_gap"]),
                "gap": cal["max_gap"],
                "description": f"Maximum calibration gap of {cal['max_gap']:.1%} detected.",
            })

        overall_fair = len(issues) == 0
        return {
            "overall_assessment": "PASS" if overall_fair else "NEEDS ATTENTION",
            "issues_found": len(issues),
            "checks_passed": len(passed),
            "issues": issues,
            "passed": passed,
        }

    def _classify_severity(self, disparity: float) -> str:
        """Classify the severity of a bias finding."""
        thresholds = self.config.bias_thresholds
        if disparity < thresholds["low"]:
            return "Low"
        elif disparity < thresholds["moderate"]:
            return "Moderate"
        elif disparity < thresholds["high"]:
            return "High"
        return "Critical"

    def get_audit_results(self) -> Optional[Dict]:
        """Return the most recent audit results."""
        return self._audit_results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert group metrics to a pandas DataFrame."""
        if self._audit_results is None:
            raise RuntimeError("Must run audit() first.")

        rows = []
        for group, metrics in self._audit_results["group_metrics"].items():
            row = {"group": group, **metrics}
            rows.append(row)

        return pd.DataFrame(rows).set_index("group")
