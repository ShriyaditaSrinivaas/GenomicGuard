#!/usr/bin/env python3
"""
Train all GenomicGuard models.

Usage:
    python scripts/train_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genomicguard.models.trainer import TrainingPipeline


def main():
    pipeline = TrainingPipeline()
    results = pipeline.run()

    # Print summary
    print("\n" + "=" * 70)
    print("  Training Summary")
    print("=" * 70)

    if "risk_scorers" in results:
        print("\n  Risk Scorers:")
        for disease, metrics in results["risk_scorers"].items():
            cv = metrics.get("cv_results", {})
            print(f"    {disease}:")
            print(f"      CV AUC:   {cv.get('mean_auc_roc', 0):.3f} ± {cv.get('std_auc_roc', 0):.3f}")
            print(f"      Test AUC: {metrics.get('test_auc', 0):.3f}")

    if "variant_classifier" in results:
        vc = results["variant_classifier"]
        print(f"\n  Variant Classifier:")
        print(f"    Accuracy:    {vc.get('accuracy', 0):.3f}")
        print(f"    Macro F1:    {vc.get('macro_f1', 0):.3f}")

    if "ensemble" in results:
        ens = results["ensemble"]
        print(f"\n  Ensemble:")
        print(f"    AUC:         {ens.get('auc_roc', 0):.3f}")


if __name__ == "__main__":
    main()
