"""
Equity Report Generator.

Produces comprehensive equity reports analyzing model performance
and fairness across population groups.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from genomicguard.config import REPORTS_DIR, POPULATION_LABELS


class EquityReportGenerator:
    """
    Generates detailed equity reports from fairness audit results.

    Reports include:
    - Population performance comparison
    - Bias severity assessment
    - Disparity visualizations data
    - Actionable recommendations
    """

    def generate_report(
        self,
        audit_results: Dict,
        bias_results: Optional[Dict] = None,
        mitigation_results: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a comprehensive equity report.

        Args:
            audit_results: Results from FairnessAuditor.audit()
            bias_results: Optional results from BiasDetector.detect_bias()
            mitigation_results: Optional results from BiasMitigator

        Returns:
            Structured equity report dictionary
        """
        report = {
            "report_type": "Equity Assessment",
            "generated_at": datetime.now().isoformat(),
            "report_id": f"EQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",

            # Executive Summary
            "executive_summary": self._generate_summary(audit_results, bias_results),

            # Performance by Population
            "population_performance": self._format_population_performance(
                audit_results.get("group_metrics", {})
            ),

            # Fairness Assessment
            "fairness_assessment": self._format_fairness_assessment(
                audit_results.get("fairness_metrics", {})
            ),

            # Bias Findings
            "bias_findings": self._format_bias_findings(bias_results) if bias_results else None,

            # Mitigation Recommendations
            "recommendations": self._generate_equity_recommendations(
                audit_results, bias_results
            ),

            # Mitigation Results (if applied)
            "mitigation_results": mitigation_results,
        }

        return report

    def _generate_summary(self, audit_results: Dict, bias_results: Optional[Dict]) -> Dict:
        """Generate executive summary."""
        summary = audit_results.get("fairness_summary", {})

        return {
            "overall_status": summary.get("overall_assessment", "UNKNOWN"),
            "total_populations_analyzed": audit_results.get("n_groups", 0),
            "fairness_checks_passed": summary.get("checks_passed", 0),
            "issues_identified": summary.get("issues_found", 0),
            "bias_risk_level": (
                bias_results.get("overall_bias_risk", "N/A") if bias_results else "N/A"
            ),
            "overall_auc": audit_results.get("overall_metrics", {}).get("auc_roc", None),
            "key_message": self._get_key_message(summary),
        }

    def _get_key_message(self, summary: Dict) -> str:
        """Generate a key message based on fairness status."""
        status = summary.get("overall_assessment", "UNKNOWN")
        if status == "PASS":
            return (
                "The model demonstrates equitable performance across all evaluated "
                "population groups, meeting all fairness thresholds."
            )
        else:
            n_issues = summary.get("issues_found", 0)
            return (
                f"The model shows {n_issues} fairness concern(s) that require attention. "
                "Review the detailed findings and consider applying recommended mitigations "
                "before clinical deployment."
            )

    def _format_population_performance(self, group_metrics: Dict) -> Dict:
        """Format population performance for the report."""
        formatted = {}
        for group, metrics in group_metrics.items():
            label = POPULATION_LABELS.get(group, group)
            formatted[group] = {
                "population": label,
                "sample_size": metrics.get("n_samples", 0),
                "prevalence": f"{metrics.get('prevalence', 0):.1%}",
                "auc_roc": metrics.get("auc_roc"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "ppv": metrics.get("ppv"),
                "npv": metrics.get("npv"),
                "f1": metrics.get("f1"),
            }
        return formatted

    def _format_fairness_assessment(self, fairness_metrics: Dict) -> Dict:
        """Format fairness assessment section."""
        assessments = {}

        # Demographic Parity
        dp = fairness_metrics.get("demographic_parity", {})
        assessments["demographic_parity"] = {
            "status": "PASS" if dp.get("passes_threshold", False) else "FAIL",
            "max_disparity": f"{dp.get('max_disparity', 0):.1%}",
            "interpretation": (
                "Positive prediction rates are consistent across groups."
                if dp.get("passes_threshold", False)
                else "Significant differences in positive prediction rates detected."
            ),
        }

        # Equalized Odds
        eo = fairness_metrics.get("equalized_odds", {})
        assessments["equalized_odds"] = {
            "status": "PASS" if eo.get("passes_threshold", False) else "FAIL",
            "tpr_disparity": f"{eo.get('tpr_disparity', 0):.1%}",
            "fpr_disparity": f"{eo.get('fpr_disparity', 0):.1%}",
            "interpretation": (
                "True positive and false positive rates are consistent across groups."
                if eo.get("passes_threshold", False)
                else "Disparities in error rates detected across groups."
            ),
        }

        # Calibration
        cal = fairness_metrics.get("calibration", {})
        assessments["calibration"] = {
            "status": "PASS" if cal.get("passes_threshold", False) else "FAIL",
            "max_gap": f"{cal.get('max_gap', 0):.1%}",
            "interpretation": (
                "Model is well-calibrated across all groups."
                if cal.get("passes_threshold", False)
                else "Calibration gaps detected for some population groups."
            ),
        }

        return assessments

    def _format_bias_findings(self, bias_results: Dict) -> List:
        """Format bias findings."""
        findings = bias_results.get("findings", [])
        return [
            {
                "type": f.get("type"),
                "attribute": f.get("attribute"),
                "severity": f.get("severity"),
                "description": f.get("description"),
            }
            for f in findings
        ]

    def _generate_equity_recommendations(
        self, audit_results: Dict, bias_results: Optional[Dict]
    ) -> list:
        """Generate actionable equity recommendations."""
        recommendations = []
        fairness_summary = audit_results.get("fairness_summary", {})
        issues = fairness_summary.get("issues", [])

        for issue in issues:
            metric = issue.get("metric", "")
            severity = issue.get("severity", "Low")

            if metric == "Demographic Parity":
                recommendations.append({
                    "priority": severity.upper(),
                    "area": "Prediction Rates",
                    "action": "Consider group-specific threshold optimization to equalize prediction rates.",
                    "rationale": issue.get("description", ""),
                })
            elif metric == "Equalized Odds":
                recommendations.append({
                    "priority": severity.upper(),
                    "area": "Error Rates",
                    "action": "Review training data balance and consider reweighting underperforming groups.",
                    "rationale": issue.get("description", ""),
                })
            elif metric == "Calibration":
                recommendations.append({
                    "priority": severity.upper(),
                    "area": "Calibration",
                    "action": "Apply group-specific probability recalibration (Platt scaling or isotonic regression).",
                    "rationale": issue.get("description", ""),
                })

        # General recommendations
        recommendations.append({
            "priority": "STANDARD",
            "area": "Monitoring",
            "action": "Implement continuous fairness monitoring in production deployment.",
            "rationale": "Bias patterns may shift over time as population distributions change.",
        })

        recommendations.append({
            "priority": "STANDARD",
            "area": "Data Collection",
            "action": "Increase representation of underrepresented populations in training data.",
            "rationale": "Larger, more diverse training sets generally improve cross-population generalization.",
        })

        return recommendations

    def save_report(self, report: Dict, output_dir: Optional[Path] = None) -> Path:
        """Save report as JSON."""
        out = output_dir or REPORTS_DIR
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{report['report_id']}.json"

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return filepath
