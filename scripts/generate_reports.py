#!/usr/bin/env python3
"""
Generate clinical reports for sample patients.

Usage:
    python scripts/generate_reports.py [--n-patients 5]
"""

import sys
import json
import argparse
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genomicguard.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, DISEASES
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.interpretability.shap_explainer import SHAPExplainer
from genomicguard.interpretability.clinical_report import ClinicalReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate clinical reports")
    parser.add_argument("--n-patients", type=int, default=5, help="Number of reports to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("  GenomicGuard - Clinical Report Generation")
    print("=" * 60)

    # Load data and models
    print("\n  Loading data and models...")
    dataset = GenomicDataGenerator.load_dataset()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")

    sample_meta = dataset["sample_metadata"]
    geno_processed, clinical_processed = preprocessor.fit_transform(
        dataset["genotypes"], dataset["clinical"], sample_meta
    )
    features = feature_engineer.fit_transform(
        geno_processed, clinical_processed, dataset["snp_metadata"]
    )

    # Load risk scorers
    risk_scorers = {}
    for disease in DISEASES:
        disease_key = disease.lower().replace(" ", "_")
        scorer_path = MODELS_DIR / f"risk_scorer_{disease_key}.joblib"
        if scorer_path.exists():
            risk_scorers[disease_key] = PolygenicRiskScorer.load(scorer_path)

    if not risk_scorers:
        print("  ✗ No trained models found. Run 'python scripts/train_models.py' first.")
        sys.exit(1)

    # Compute SHAP for the first disease
    first_disease = list(risk_scorers.keys())[0]
    scorer = risk_scorers[first_disease]
    print(f"  Computing SHAP explanations for {first_disease}...")
    explainer = SHAPExplainer(scorer.model, list(features.columns))
    explainer.compute_shap_values(features)

    # Generate reports
    report_gen = ClinicalReportGenerator()
    n_patients = min(args.n_patients, len(features))

    # Select diverse patients
    rng = np.random.RandomState(42)
    patient_indices = rng.choice(len(features), size=n_patients, replace=False)

    print(f"\n  Generating {n_patients} clinical reports...")
    for idx in patient_indices:
        patient_id = sample_meta.iloc[idx]["sample_id"]

        # Get risk scores for all diseases
        risk_scores = {}
        for disease_key, s in risk_scorers.items():
            risk = s.predict_risk(features.iloc[[idx]])[0]
            risk_scores[disease_key] = float(risk)

        # Get SHAP explanation
        explanation = explainer.get_patient_explanation(idx, features)

        # Get patient features as dict
        patient_features = features.iloc[idx].to_dict()

        # Generate report
        report = report_gen.generate_patient_report(
            patient_id=patient_id,
            risk_scores=risk_scores,
            shap_explanation=explanation,
            patient_features=patient_features,
        )

        # Save
        filepath = report_gen.save_report(report)
        risk_level = report["risk_summary"]["overall_risk_level"]
        print(f"  ✓ {patient_id}: Risk = {risk_level} → {filepath}")

    print(f"\n✓ Generated {n_patients} reports in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
