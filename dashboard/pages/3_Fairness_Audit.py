"""
Fairness Audit Dashboard Page.

Population equity analysis, bias detection, and mitigation insights.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genomicguard.config import DISEASES, MODELS_DIR, POPULATION_LABELS, DashboardConfig
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.fairness.auditor import FairnessAuditor
from genomicguard.fairness.bias_detector import BiasDetector
from genomicguard.fairness.mitigation import BiasMitigator

st.set_page_config(page_title="Fairness Audit", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); }
    h1 { background: linear-gradient(135deg, #818cf8, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px; padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

config = DashboardConfig()

POP_COLORS = config.population_colors


@st.cache_data
def load_and_audit():
    try:
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

        # Load first disease model
        disease_key = DISEASES[0].lower().replace(" ", "_")
        scorer = PolygenicRiskScorer.load(MODELS_DIR / f"risk_scorer_{disease_key}.joblib")

        y_true = dataset["phenotypes"][f"{disease_key}_label"].values
        y_prob = scorer.predict_risk(features)
        populations = sample_meta["population"].values

        # Run audit
        auditor = FairnessAuditor()
        audit_results = auditor.audit(y_true, y_prob, populations)

        # Run bias detection
        detector = BiasDetector()
        age_groups = np.where(
            dataset["clinical"]["age"] < 40, "young",
            np.where(dataset["clinical"]["age"] < 60, "middle", "senior")
        )
        protected_attrs = {
            "population": populations,
            "sex": sample_meta["sex"].values,
            "age_group": age_groups,
        }
        bias_results = detector.detect_bias(y_true, y_prob, protected_attrs)

        # Mitigation
        mitigator = BiasMitigator()
        thresholds = mitigator.optimize_thresholds(y_true, y_prob, populations)

        return audit_results, bias_results, thresholds, disease_key
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None


st.markdown("# ⚖️ Fairness Audit Dashboard")
st.markdown("Population equity analysis across the genomic risk model.")
st.markdown("---")

audit_results, bias_results, opt_thresholds, disease_key = load_and_audit()

if audit_results is None:
    st.error("⚠️ Run the pipeline first.")
    st.stop()

# ── Overall Status ─────────────────────────────────────────────────────────
summary = audit_results["fairness_summary"]
status = summary["overall_assessment"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    status_color = "🟢" if status == "PASS" else "🟡"
    st.metric("Overall Status", f"{status_color} {status}")
with col2:
    st.metric("Checks Passed", f"{summary['checks_passed']}/3")
with col3:
    st.metric("Issues Found", summary["issues_found"])
with col4:
    overall_auc = audit_results["overall_metrics"]["auc_roc"]
    st.metric("Overall AUC", f"{overall_auc:.3f}")

st.markdown("---")

# ── Performance by Population ──────────────────────────────────────────────
st.markdown("## 📊 Performance by Population")

group_metrics = audit_results["group_metrics"]

# AUC comparison
auc_data = {
    g: m.get("auc_roc", 0) or 0
    for g, m in group_metrics.items()
}
colors_list = [POP_COLORS.get(g, "#888888") for g in auc_data.keys()]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(auc_data.keys()),
    y=list(auc_data.values()),
    marker_color=colors_list,
    text=[f"{v:.3f}" for v in auc_data.values()],
    textposition="outside",
    textfont=dict(size=14, color="#e2e8f0"),
))

# Overall line
fig.add_hline(
    y=overall_auc, line_dash="dash", line_color="#94a3b8",
    annotation_text=f"Overall: {overall_auc:.3f}",
    annotation_position="top right",
    annotation_font_color="#94a3b8",
)

fig.update_layout(
    title=f"AUC-ROC by Population — {disease_key.replace('_', ' ').title()}",
    yaxis_title="AUC-ROC",
    yaxis_range=[max(0, min(auc_data.values()) - 0.1), 1.0],
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
)
st.plotly_chart(fig, use_container_width=True)

# Metrics table
st.markdown("### Detailed Metrics")
table_data = []
for g, m in group_metrics.items():
    table_data.append({
        "Population": POPULATION_LABELS.get(g, g),
        "N": m.get("n_samples", 0),
        "Prevalence": f"{m.get('prevalence', 0):.1%}",
        "AUC-ROC": f"{m.get('auc_roc', 0):.3f}" if m.get("auc_roc") else "N/A",
        "Sensitivity": f"{m.get('sensitivity', 0):.3f}" if m.get("sensitivity") is not None else "N/A",
        "Specificity": f"{m.get('specificity', 0):.3f}" if m.get("specificity") is not None else "N/A",
        "PPV": f"{m.get('ppv', 0):.3f}" if m.get("ppv") is not None else "N/A",
        "NPV": f"{m.get('npv', 0):.3f}" if m.get("npv") is not None else "N/A",
    })
st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ── Fairness Checks ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔍 Fairness Checks")

fm = audit_results["fairness_metrics"]

check_col1, check_col2, check_col3 = st.columns(3)

with check_col1:
    dp = fm["demographic_parity"]
    icon = "✅" if dp["passes_threshold"] else "⚠️"
    st.markdown(f"### {icon} Demographic Parity")
    st.markdown(f"**Max Disparity:** {dp['max_disparity']:.1%}")
    st.markdown("Positive prediction rates should be similar across groups.")

    # Mini bar chart
    pred_rates = dp["prediction_rates"]
    fig_dp = go.Figure(go.Bar(
        x=list(pred_rates.keys()),
        y=list(pred_rates.values()),
        marker_color=[POP_COLORS.get(g, "#888") for g in pred_rates.keys()],
    ))
    fig_dp.update_layout(
        height=200, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", yaxis_title="Prediction Rate",
        yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
    )
    st.plotly_chart(fig_dp, use_container_width=True)

with check_col2:
    eo = fm["equalized_odds"]
    icon = "✅" if eo["passes_threshold"] else "⚠️"
    st.markdown(f"### {icon} Equalized Odds")
    st.markdown(f"**TPR Disparity:** {eo['tpr_disparity']:.1%}")
    st.markdown(f"**FPR Disparity:** {eo['fpr_disparity']:.1%}")
    st.markdown("Error rates should be balanced across groups.")

    # TPR/FPR comparison
    if eo["tpr_by_group"]:
        pops_eo = list(eo["tpr_by_group"].keys())
        fig_eo = go.Figure()
        fig_eo.add_trace(go.Bar(name="TPR", x=pops_eo, y=[eo["tpr_by_group"][g] for g in pops_eo], marker_color="#6366f1"))
        fig_eo.add_trace(go.Bar(name="FPR", x=pops_eo, y=[eo["fpr_by_group"].get(g, 0) for g in pops_eo], marker_color="#ef4444"))
        fig_eo.update_layout(
            barmode="group", height=200, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig_eo, use_container_width=True)

with check_col3:
    cal = fm["calibration"]
    icon = "✅" if cal["passes_threshold"] else "⚠️"
    st.markdown(f"### {icon} Calibration")
    st.markdown(f"**Max Gap:** {cal['max_gap']:.1%}")
    st.markdown("Predicted risk should match observed rates.")

    # Calibration comparison
    cal_data = cal["per_group"]
    pops_cal = list(cal_data.keys())
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(name="Predicted", x=pops_cal, y=[cal_data[g]["mean_predicted"] for g in pops_cal], marker_color="#6366f1"))
    fig_cal.add_trace(go.Bar(name="Observed", x=pops_cal, y=[cal_data[g]["mean_actual"] for g in pops_cal], marker_color="#10b981"))
    fig_cal.update_layout(
        barmode="group", height=200, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ── Bias Detection ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🚨 Bias Detection Findings")

if bias_results:
    n_findings = bias_results["total_findings"]
    risk_level = bias_results["overall_bias_risk"]

    risk_color = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")
    st.markdown(f"**Overall Bias Risk:** {risk_color} {risk_level} | **{n_findings} findings**")

    if n_findings > 0:
        for finding in bias_results["findings"]:
            severity = finding["severity"]
            sev_icon = {"Negligible": "⚪", "Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(severity, "⚪")
            st.markdown(
                f"- {sev_icon} **[{severity}]** {finding['type']} ({finding['attribute']}): {finding['description']}"
            )
    else:
        st.success("No significant bias detected across analyzed attributes.")

    # Underperforming subgroups
    if bias_results.get("underperforming_subgroups"):
        st.markdown("### ⚠️ Underperforming Subgroups")
        under_df = pd.DataFrame(bias_results["underperforming_subgroups"])
        if not under_df.empty:
            st.dataframe(under_df[["attribute", "group", "group_auc", "overall_auc", "gap", "severity"]],
                        use_container_width=True, hide_index=True)

# ── Mitigation ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🛠️ Mitigation Strategies")

st.markdown("### Group-Specific Thresholds")
st.markdown("Instead of a single threshold, optimize per-population thresholds to equalize performance.")

if opt_thresholds:
    thresh_df = pd.DataFrame([
        {"Population": POPULATION_LABELS.get(g, g), "Default (0.5)": 0.5, "Optimized": t}
        for g, t in opt_thresholds.items()
    ])
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)

st.info(
    "💡 **Key Insight**: Group-specific thresholds and per-population recalibration "
    "can reduce performance disparities without requiring model retraining. These "
    "post-processing approaches are recommended as a first intervention."
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "⚠️ This is a research demonstration. Not for clinical use."
    "</div>",
    unsafe_allow_html=True,
)
