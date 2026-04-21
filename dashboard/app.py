"""
GenomicGuard Dashboard - Main Application.

A clinical-grade Streamlit dashboard for interactive exploration of
genomic risk predictions, model interpretability, and fairness auditing.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomicguard.config import DashboardConfig

config = DashboardConfig()

st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout=config.layout,
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Headers */
    h1 {
        background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    h2, h3 {
        color: #e2e8f0 !important;
    }

    /* Metric containers */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ── Main Page ──────────────────────────────────────────────────────────────────
st.markdown("# 🧬 GenomicGuard")
st.markdown("### Interpretable Clinical Genomics Risk Assessment with Fairness Auditing")

st.markdown("---")

# Overview columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Pipeline Modules",
        value="4",
        help="Data, Models, Interpretability, Fairness"
    )

with col2:
    st.metric(
        label="🔬 Diseases Modeled",
        value="3",
        help="T2D, CAD, Breast Cancer"
    )

with col3:
    st.metric(
        label="🌍 Populations",
        value="5",
        help="EUR, AFR, EAS, SAS, AMR"
    )

with col4:
    st.metric(
        label="⚖️ Fairness Metrics",
        value="4",
        help="Demographic Parity, Equalized Odds, Calibration, AUC Parity"
    )

st.markdown("---")

# Architecture overview
st.markdown("## 📐 System Architecture")

st.markdown("""
**GenomicGuard** is an end-to-end clinical genomics risk assessment system that prioritizes
**interpretability** and **healthcare equity**. It combines polygenic risk scoring with
built-in fairness auditing to ensure equitable precision medicine across diverse populations.

### 🔄 Pipeline Overview

1. **Data Generation** → Synthetic genomic data with population stratification
2. **Preprocessing** → QC, imputation, feature engineering (PRS, PCA, interactions)
3. **Model Training** → Gradient Boosting risk scorers + MLP variant classifier + Ensemble
4. **Interpretability** → SHAP explanations for every prediction
5. **Fairness Audit** → Population-stratified performance, bias detection, mitigation

### 📄 Pages

Use the sidebar to navigate:
- **Patient Risk Profile** — Individual patient risk assessment with SHAP explanations
- **Model Interpretability** — Global feature importance and model explanations
- **Fairness Audit** — Population equity dashboard with bias detection
- **Population Analytics** — Population-level risk distributions and demographics
""")

# Status check
st.markdown("---")
st.markdown("## 🔧 System Status")

data_dir = Path(__file__).parent.parent / "data" / "synthetic"
models_dir = Path(__file__).parent.parent / "models"
reports_dir = Path(__file__).parent.parent / "reports"

col1, col2, col3 = st.columns(3)

with col1:
    data_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    if data_files:
        st.success(f"✅ Data: {len(data_files)} files loaded")
    else:
        st.error("❌ No data found. Run: `python scripts/generate_data.py`")

with col2:
    model_files = list(models_dir.glob("*.joblib")) if models_dir.exists() else []
    if model_files:
        st.success(f"✅ Models: {len(model_files)} artifacts loaded")
    else:
        st.error("❌ No models found. Run: `python scripts/train_models.py`")

with col3:
    report_files = list(reports_dir.glob("*.json")) if reports_dir.exists() else []
    if report_files:
        st.success(f"✅ Reports: {len(report_files)} reports available")
    else:
        st.warning("⚠️ No reports. Run: `python scripts/run_fairness_audit.py`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "GenomicGuard v1.0.0 • Built for equitable precision medicine • "
    "<a href='https://github.com/shriyaditasrinivaas/GenomicGuard' style='color: #818cf8;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
