"""
Patient Risk Profile Page.

Individual patient risk assessment with SHAP-based explanations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genomicguard.config import DISEASES, MODELS_DIR, DashboardConfig
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.interpretability.shap_explainer import SHAPExplainer

st.set_page_config(page_title="Patient Risk Profile", page_icon="🏥", layout="wide")

# Custom CSS
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


@st.cache_data
def load_data():
    """Load dataset and models."""
    try:
        dataset = GenomicDataGenerator.load_dataset()
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")

        geno_processed, clinical_processed = preprocessor.fit_transform(
            dataset["genotypes"], dataset["clinical"], dataset["sample_metadata"]
        )
        features = feature_engineer.fit_transform(
            geno_processed, clinical_processed, dataset["snp_metadata"]
        )

        # Load risk scorers
        risk_scorers = {}
        for disease in DISEASES:
            dk = disease.lower().replace(" ", "_")
            path = MODELS_DIR / f"risk_scorer_{dk}.joblib"
            if path.exists():
                risk_scorers[dk] = PolygenicRiskScorer.load(path)

        return dataset, features, risk_scorers
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


st.markdown("# 🏥 Patient Risk Profile")
st.markdown("Individual patient risk assessment with interpretable AI explanations.")
st.markdown("---")

dataset, features, risk_scorers = load_data()

if dataset is None:
    st.error("⚠️ Data or models not found. Please run the pipeline first:")
    st.code("python scripts/generate_data.py && python scripts/train_models.py")
    st.stop()

sample_meta = dataset["sample_metadata"]
clinical = dataset["clinical"]
phenotypes = dataset["phenotypes"]

# Sidebar: Patient selection
st.sidebar.markdown("### 🔍 Select Patient")
patient_idx = st.sidebar.selectbox(
    "Patient",
    range(len(sample_meta)),
    format_func=lambda i: f"{sample_meta.iloc[i]['sample_id']} ({sample_meta.iloc[i]['population']})",
)

# Patient info
patient = sample_meta.iloc[patient_idx]
patient_clinical = clinical.iloc[patient_idx]

# ── Patient Demographics ───────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Patient ID", patient["sample_id"])
with col2:
    st.metric("Population", patient["population"])
with col3:
    st.metric("Sex", patient["sex"])
with col4:
    st.metric("Age", int(patient_clinical["age"]))
with col5:
    st.metric("BMI", f"{patient_clinical['bmi']:.1f}")

st.markdown("---")

# ── Risk Scores ────────────────────────────────────────────────────────────
st.markdown("## 📊 Disease Risk Assessment")

risk_cols = st.columns(len(risk_scorers))
patient_risks = {}

for i, (disease_key, scorer) in enumerate(risk_scorers.items()):
    risk = scorer.predict_risk(features.iloc[[patient_idx]])[0]
    patient_risks[disease_key] = float(risk)

    risk_pct = risk * 100
    disease_label = disease_key.replace("_", " ").title()

    # Risk color
    if risk < 0.3:
        color, level = "#22c55e", "LOW"
    elif risk < 0.6:
        color, level = "#f59e0b", "MODERATE"
    elif risk < 0.8:
        color, level = "#ef4444", "HIGH"
    else:
        color, level = "#991b1b", "VERY HIGH"

    with risk_cols[i]:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": disease_label, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [60, 80], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [80, 100], "color": "rgba(153, 27, 27, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": risk_pct,
                },
            },
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div style='text-align:center'><span style='color:{color}; font-weight:bold; font-size:1.2rem'>● {level}</span></div>", unsafe_allow_html=True)

# ── SHAP Explanation ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔍 Risk Factor Breakdown (SHAP)")

# Use first disease for SHAP
first_disease = list(risk_scorers.keys())[0]
scorer = risk_scorers[first_disease]

with st.spinner("Computing SHAP explanations..."):
    explainer = SHAPExplainer(scorer.model, list(features.columns))
    explainer.compute_shap_values(features)
    explanation = explainer.get_patient_explanation(patient_idx, features)

# Waterfall-style bar chart
top_factors = explanation["top_risk_factors"][:15]
factor_names = [f["feature"].replace("clinical_", "").replace("_", " ").title() for f in top_factors]
shap_values = [f["shap_value"] for f in top_factors]
colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_values]

fig = go.Figure(go.Bar(
    y=factor_names[::-1],
    x=shap_values[::-1],
    orientation="h",
    marker_color=colors[::-1],
    text=[f"{v:+.3f}" for v in shap_values[::-1]],
    textposition="auto",
))

fig.update_layout(
    title=f"Top Risk Factors — {first_disease.replace('_', ' ').title()}",
    xaxis_title="SHAP Value (Impact on Risk)",
    yaxis_title="",
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    xaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Clinical Values ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🩺 Clinical Profile")

lab_col1, lab_col2, lab_col3, lab_col4 = st.columns(4)

with lab_col1:
    st.metric("Fasting Glucose", f"{patient_clinical['fasting_glucose']:.0f} mg/dL")
    st.metric("HbA1c", f"{patient_clinical['hba1c']:.1f}%")
with lab_col2:
    st.metric("Systolic BP", f"{patient_clinical['systolic_bp']:.0f} mmHg")
    st.metric("Total Cholesterol", f"{patient_clinical['total_cholesterol']:.0f} mg/dL")
with lab_col3:
    st.metric("HDL Cholesterol", f"{patient_clinical['hdl_cholesterol']:.0f} mg/dL")
    st.metric("LDL Cholesterol", f"{patient_clinical['ldl_cholesterol']:.0f} mg/dL")
with lab_col4:
    st.metric("Triglycerides", f"{patient_clinical['triglycerides']:.0f} mg/dL")
    family_hx = "Yes" if patient_clinical["family_history"] else "No"
    st.metric("Family History", family_hx)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "⚠️ This is a research demonstration. Not for clinical use."
    "</div>",
    unsafe_allow_html=True,
)
