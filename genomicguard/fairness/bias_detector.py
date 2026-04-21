"""
Bias Detection Engine.

Automated bias detection with configurable thresholds,
intersectional analysis, and drift monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from sklearn.metrics import roc_auc_score

from genomicguard.config import FairnessConfig


class BiasDetector:
    """
    Automated bias detection for genomic risk models.

    Features:
    - Multi-attribute bias scanning
    - Intersectional analysis (e.g., population × sex)
    - Subgroup performance analysis
    - Bias severity classification
    """

    def __init__(self, config: Optional[FairnessConfig] = None):
        self.config = config or FairnessConfig()
        self._findings = []

    def detect_bias(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        protected_attrs: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> Dict:
        """
        Run comprehensive bias detection.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            protected_attrs: Dict mapping attribute names to group arrays
            threshold: Classification threshold

        Returns:
            Bias detection report
        """
        self._findings = []
        y_pred = (y_prob >= threshold).astype(int)

        # Single-attribute analysis
        single_attr_results = {}
        for attr_name, groups in protected_attrs.items():
            result = self._analyze_single_attribute(
                y_true, y_prob, y_pred, groups, attr_name
            )
            single_attr_results[attr_name] = result

        # Intersectional analysis
        intersectional_results = self._analyze_intersections(
            y_true, y_prob, y_pred, protected_attrs
        )

        # Underperforming subgroups
        underperforming = self._find_underperforming_subgroups(
            y_true, y_prob, protected_attrs
        )

        return {
            "single_attribute_analysis": single_attr_results,
            "intersectional_analysis": intersectional_results,
            "underperforming_subgroups": underperforming,
            "total_findings": len(self._findings),
            "findings": self._findings,
            "overall_bias_risk": self._assess_overall_risk(),
        }

    def _analyze_single_attribute(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        attr_name: str,
    ) -> Dict:
        """Analyze bias for a single protected attribute."""
        unique_groups = np.unique(groups)
        group_aucs = {}
        group_pred_rates = {}

        for g in unique_groups:
            mask = groups == g
            yt = y_true[mask]
            yp = y_prob[mask]
            yd = y_pred[mask]

            if len(np.unique(yt)) >= 2:
                group_aucs[str(g)] = float(roc_auc_score(yt, yp))
            group_pred_rates[str(g)] = float(yd.mean())

        # Detect issues
        if group_aucs:
            auc_gap = max(group_aucs.values()) - min(group_aucs.values())
            if auc_gap > self.config.bias_thresholds["moderate"]:
                best_group = max(group_aucs, key=group_aucs.get)
                worst_group = min(group_aucs, key=group_aucs.get)
                self._findings.append({
                    "type": "Performance Disparity",
                    "attribute": attr_name,
                    "severity": self._severity(auc_gap),
                    "description": (
                        f"AUC gap of {auc_gap:.3f} between {best_group} "
                        f"({group_aucs[best_group]:.3f}) and {worst_group} "
                        f"({group_aucs[worst_group]:.3f})"
                    ),
                    "gap": float(auc_gap),
                })

        pred_rate_gap = max(group_pred_rates.values()) - min(group_pred_rates.values())
        if pred_rate_gap > self.config.bias_thresholds["moderate"]:
            self._findings.append({
                "type": "Prediction Rate Disparity",
                "attribute": attr_name,
                "severity": self._severity(pred_rate_gap),
                "description": (
                    f"Positive prediction rate varies by {pred_rate_gap:.1%} across {attr_name} groups"
                ),
                "gap": float(pred_rate_gap),
            })

        return {
            "auc_by_group": group_aucs,
            "prediction_rates": group_pred_rates,
            "auc_gap": float(auc_gap) if group_aucs else None,
            "pred_rate_gap": float(pred_rate_gap),
        }

    def _analyze_intersections(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        protected_attrs: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Analyze intersectional bias (e.g., population × sex).

        Intersectional analysis can reveal disparities hidden by
        single-attribute analysis.
        """
        attr_names = list(protected_attrs.keys())
        results = {}

        for attr1, attr2 in combinations(attr_names, 2):
            intersection_key = f"{attr1}_x_{attr2}"
            groups1 = protected_attrs[attr1]
            groups2 = protected_attrs[attr2]

            # Create intersection groups
            intersection = np.array([
                f"{g1}_{g2}" for g1, g2 in zip(groups1, groups2)
            ])

            # Compute AUC per intersection
            group_aucs = {}
            for g in np.unique(intersection):
                mask = intersection == g
                yt = y_true[mask]
                yp = y_prob[mask]

                if len(yt) >= 10 and len(np.unique(yt)) >= 2:
                    group_aucs[g] = float(roc_auc_score(yt, yp))

            if group_aucs:
                gap = max(group_aucs.values()) - min(group_aucs.values())
                results[intersection_key] = {
                    "group_aucs": group_aucs,
                    "max_gap": float(gap),
                    "n_subgroups": len(group_aucs),
                }

                if gap > self.config.bias_thresholds["high"]:
                    self._findings.append({
                        "type": "Intersectional Disparity",
                        "attribute": intersection_key,
                        "severity": self._severity(gap),
                        "description": (
                            f"Intersectional AUC gap of {gap:.3f} across {intersection_key} subgroups"
                        ),
                        "gap": float(gap),
                    })

        return results

    def _find_underperforming_subgroups(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        protected_attrs: Dict[str, np.ndarray],
        min_size: int = 20,
    ) -> List[Dict]:
        """Identify subgroups where the model significantly underperforms."""
        # Overall AUC
        overall_auc = roc_auc_score(y_true, y_prob)

        underperforming = []
        for attr_name, groups in protected_attrs.items():
            for g in np.unique(groups):
                mask = groups == g
                yt = y_true[mask]
                yp = y_prob[mask]

                if len(yt) >= min_size and len(np.unique(yt)) >= 2:
                    group_auc = roc_auc_score(yt, yp)
                    gap = overall_auc - group_auc

                    if gap > self.config.bias_thresholds["low"]:
                        underperforming.append({
                            "attribute": attr_name,
                            "group": str(g),
                            "group_auc": float(group_auc),
                            "overall_auc": float(overall_auc),
                            "gap": float(gap),
                            "n_samples": int(mask.sum()),
                            "severity": self._severity(gap),
                        })

        return sorted(underperforming, key=lambda x: -x["gap"])

    def _severity(self, gap: float) -> str:
        """Map disparity to severity level."""
        if gap < self.config.bias_thresholds["low"]:
            return "Negligible"
        elif gap < self.config.bias_thresholds["moderate"]:
            return "Low"
        elif gap < self.config.bias_thresholds["high"]:
            return "Moderate"
        return "High"

    def _assess_overall_risk(self) -> str:
        """Assess overall bias risk based on findings."""
        if not self._findings:
            return "LOW"

        severities = [f["severity"] for f in self._findings]
        if "High" in severities or "Critical" in severities:
            return "HIGH"
        elif "Moderate" in severities:
            return "MODERATE"
        return "LOW"
