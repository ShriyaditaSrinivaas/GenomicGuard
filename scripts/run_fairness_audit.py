#!/usr/bin/env python3
"""
Run fairness audit on trained models.

Usage:
    python scripts/run_fairness_audit.py
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genomicguard.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, DISEASES
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.data.preprocessor import GenomicPreprocessor
from genomicguard.data.feature_engineering import GenomicFeatureEngineer
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.fairness.auditor import FairnessAuditor
from genomicguard.fairness.bias_detector import BiasDetector
from genomicguard.fairness.mitigation import BiasMitigator
from genomicguard.fairness.equity_report import EquityReportGenerator


def main():
    print("=" * 70)
    print("  GenomicGuard - Fairness Audit")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data and models...")
    dataset = GenomicDataGenerator.load_dataset()
    sample_meta = dataset["sample_metadata"]
    phenotypes = dataset["phenotypes"]
    populations = sample_meta["population"].values

    # Load preprocessor and feature engineer
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")

    # Preprocess
    geno_processed, clinical_processed = preprocessor.fit_transform(
        dataset["genotypes"], dataset["clinical"], sample_meta
    )
    features = feature_engineer.fit_transform(
        geno_processed, clinical_processed, dataset["snp_metadata"]
    )

    # Load first available risk scorer
    disease_key = DISEASES[0].lower().replace(" ", "_")
    scorer_path = MODELS_DIR / f"risk_scorer_{disease_key}.joblib"

    if not scorer_path.exists():
        print(f"  ✗ Model not found: {scorer_path}")
        print("  Run 'python scripts/train_models.py' first.")
        sys.exit(1)

    scorer = PolygenicRiskScorer.load(scorer_path)
    y_true = phenotypes[f"{disease_key}_label"].values
    y_prob = scorer.predict_risk(features)

    # 2. Fairness Audit
    print("\n[2/5] Running fairness audit...")
    auditor = FairnessAuditor()
    audit_results = auditor.audit(y_true, y_prob, populations)

    print(f"\n  Overall Assessment: {audit_results['fairness_summary']['overall_assessment']}")
    print(f"  Checks Passed: {audit_results['fairness_summary']['checks_passed']}")
    print(f"  Issues Found: {audit_results['fairness_summary']['issues_found']}")

    # Print group metrics
    print("\n  Performance by Population:")
    print(f"  {'Pop':<6} {'N':>6} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8}")
    print(f"  {'-'*44}")
    for group, metrics in audit_results["group_metrics"].items():
        auc = metrics.get("auc_roc", "N/A")
        sens = metrics.get("sensitivity", "N/A")
        spec = metrics.get("specificity", "N/A")
        ppv = metrics.get("ppv", "N/A")
        auc_str = f"{auc:.3f}" if isinstance(auc, float) else auc
        sens_str = f"{sens:.3f}" if isinstance(sens, float) else sens
        spec_str = f"{spec:.3f}" if isinstance(spec, float) else spec
        ppv_str = f"{ppv:.3f}" if isinstance(ppv, float) else ppv
        print(f"  {group:<6} {metrics['n_samples']:>6} {auc_str:>8} {sens_str:>8} {spec_str:>8} {ppv_str:>8}")

    # 3. Bias Detection
    print("\n[3/5] Running bias detection...")
    detector = BiasDetector()

    # Create age groups for intersectional analysis
    clinical = dataset["clinical"]
    age_groups = np.where(clinical["age"] < 40, "young",
                 np.where(clinical["age"] < 60, "middle", "senior"))
    sex = sample_meta["sex"].values

    protected_attrs = {
        "population": populations,
        "sex": sex,
        "age_group": age_groups,
    }
    bias_results = detector.detect_bias(y_true, y_prob, protected_attrs)

    print(f"  Overall Bias Risk: {bias_results['overall_bias_risk']}")
    print(f"  Total Findings: {bias_results['total_findings']}")

    for finding in bias_results["findings"]:
        print(f"    [{finding['severity']}] {finding['type']}: {finding['description']}")

    # 4. Mitigation
    print("\n[4/5] Computing mitigation strategies...")
    mitigator = BiasMitigator()

    # Threshold optimization
    thresholds = mitigator.optimize_thresholds(y_true, y_prob, populations)
    print(f"  Optimized thresholds: {thresholds}")

    # Sample weights
    weights = mitigator.compute_sample_weights(y_true, populations)
    print(f"  Reweighting factors: {mitigator.get_mitigation_summary()['reweighting_factors']}")

    # Recalibration
    y_prob_recal = mitigator.recalibrate_probabilities(y_true, y_prob, populations)
    mitigation_eval = mitigator.evaluate_mitigation(y_true, y_prob, y_prob_recal, populations)

    print(f"  AUC gap before: {mitigation_eval['before']['auc_gap']:.3f}")
    print(f"  AUC gap after:  {mitigation_eval['after']['auc_gap']:.3f}")

    # 5. Generate equity report
    print("\n[5/5] Generating equity report...")
    report_gen = EquityReportGenerator()
    equity_report = report_gen.generate_report(audit_results, bias_results, mitigation_eval)
    report_path = report_gen.save_report(equity_report)
    print(f"  ✓ Report saved to {report_path}")

    # Save audit results
    audit_save_path = REPORTS_DIR / "fairness_audit_results.json"
    with open(audit_save_path, "w") as f:
        json.dump(audit_results, f, indent=2, default=str)
    print(f"  ✓ Audit results saved to {audit_save_path}")

    print("\n✓ Fairness audit complete!")


if __name__ == "__main__":
    main()
