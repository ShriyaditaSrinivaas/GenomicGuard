"""
Model Interpretability Page.

Global feature importance and model explanation visualizations.
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

from genomicguard.config import DISEASES, MODELS_DIR, POPULATION_LABELS
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.models.risk_scorer import PolygenicRiskScorer
from genomicguard.interpretability.shap_explainer import SHAPExplainer

st.set_page_config(page_title="Model Interpretability", page_icon="🔬", layout="wide")

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


@st.cache_data
def load_data_and_models():
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

        risk_scorers = {}
        for disease in DISEASES:
            dk = disease.lower().replace(" ", "_")
            path = MODELS_DIR / f"risk_scorer_{dk}.joblib"
            if path.exists():
                risk_scorers[dk] = PolygenicRiskScorer.load(path)

        training_results = None
        tr_path = MODELS_DIR / "training_results.joblib"
        if tr_path.exists():
            training_results = joblib.load(tr_path)

        return dataset, features, risk_scorers, training_results
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None


st.markdown("# 🔬 Model Interpretability")
st.markdown("Understanding what drives predictions and why the model makes its decisions.")
st.markdown("---")

dataset, features, risk_scorers, training_results = load_data_and_models()

if dataset is None:
    st.error("⚠️ Data or models not found. Run the pipeline first.")
    st.stop()

sample_meta = dataset["sample_metadata"]

# Disease selector
disease_key = st.sidebar.selectbox(
    "Select Disease Model",
    list(risk_scorers.keys()),
    format_func=lambda x: x.replace("_", " ").title(),
)

scorer = risk_scorers[disease_key]

# ── Model Performance ──────────────────────────────────────────────────────
st.markdown("## 📈 Model Performance")

if training_results and disease_key in training_results.get("risk_scorers", {}):
    res = training_results["risk_scorers"][disease_key]
    cv = res.get("cv_results", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CV AUC-ROC", f"{cv.get('mean_auc_roc', 0):.3f}")
    with col2:
        st.metric("CV AUC-PR", f"{cv.get('mean_auc_pr', 0):.3f}")
    with col3:
        st.metric("Test AUC", f"{res.get('test_auc', 0):.3f}")
    with col4:
        st.metric("Brier Score", f"{cv.get('mean_brier', 0):.3f}")

st.markdown("---")

# ── Global SHAP ───────────────────────────────────────────────────────────
st.markdown("## 🌐 Global Feature Importance (SHAP)")

with st.spinner("Computing SHAP values..."):
    explainer = SHAPExplainer(scorer.model, list(features.columns))
    explainer.compute_shap_values(features)
    global_importance = explainer.get_global_importance()

n_features = st.slider("Number of features to display", 5, 30, 15)
top_features = global_importance.head(n_features)

# Horizontal bar chart
fig = go.Figure(go.Bar(
    y=top_features["feature"].values[::-1],
    x=top_features["mean_abs_shap"].values[::-1],
    orientation="h",
    marker_color=px.colors.sequential.Viridis[:n_features][::-1] if n_features <= 10
    else ["#6366f1"] * n_features,
    text=[f"{v:.4f}" for v in top_features["mean_abs_shap"].values[::-1]],
    textposition="auto",
))

fig.update_layout(
    title="Mean |SHAP| Value — Feature Importance Ranking",
    xaxis_title="Mean |SHAP Value|",
    height=max(400, n_features * 28),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    xaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Feature Importance by Category ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Importance by Feature Category")

def categorize_feature(name):
    if name.startswith("clinical_"):
        return "Clinical"
    elif name.startswith("prs_"):
        return "Polygenic Risk Score"
    elif name.startswith("pc_"):
        return "Population Structure (PCA)"
    elif name.startswith("interact_"):
        return "Gene-Gene Interaction"
    elif "composite" in name or "interaction" in name:
        return "Clinical Composite"
    return "Other"

global_importance["category"] = global_importance["feature"].apply(categorize_feature)
category_importance = global_importance.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=False)

fig = go.Figure(go.Pie(
    labels=category_importance.index,
    values=category_importance.values,
    hole=0.4,
    marker_colors=["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#94a3b8"],
    textinfo="label+percent",
    textposition="outside",
))

fig.update_layout(
    title="Feature Importance by Category",
    height=450,
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
)
st.plotly_chart(fig, use_container_width=True)

# ── Population-Stratified Importance ───────────────────────────────────────
st.markdown("---")
st.markdown("## 🌍 Population-Stratified Feature Importance")
st.markdown("Do the same features matter for all populations?")

pop_importance = explainer.get_population_stratified_importance(
    features, sample_meta["population"].values
)

# Heatmap of top features × populations
n_top = 10
all_pops = sorted(pop_importance.keys())
top_feature_names = global_importance.head(n_top)["feature"].values

heatmap_data = np.zeros((n_top, len(all_pops)))
for j, pop in enumerate(all_pops):
    pop_df = pop_importance[pop].set_index("feature")
    for i, feat in enumerate(top_feature_names):
        if feat in pop_df.index:
            heatmap_data[i, j] = pop_df.loc[feat, "mean_abs_shap"]

fig = go.Figure(go.Heatmap(
    z=heatmap_data,
    x=all_pops,
    y=[f.replace("clinical_", "").replace("_", " ").title() for f in top_feature_names],
    colorscale="Viridis",
    text=np.round(heatmap_data, 4),
    texttemplate="%{text:.4f}",
    textfont={"size": 10},
))

fig.update_layout(
    title="Feature Importance Heatmap Across Populations",
    xaxis_title="Population",
    yaxis_title="Feature",
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "💡 **Interpretation**: Differences in this heatmap reveal whether the model "
    "depends on different features for different populations. Large differences "
    "could indicate population-specific risk patterns or potential model bias."
)

# ── Built-in Feature Importance ────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌳 Gradient Boosting Feature Importance")

gini_importance = scorer.get_feature_importances()
top_gini = gini_importance.head(n_features)

fig = go.Figure(go.Bar(
    x=top_gini["feature"].apply(lambda x: x.replace("clinical_", "").replace("_", " ").title()),
    y=top_gini["importance"],
    marker_color="#a78bfa",
))

fig.update_layout(
    title="Built-in Feature Importance (Gini)",
    yaxis_title="Importance",
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    xaxis=dict(tickangle=-45),
    yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "⚠️ This is a research demonstration. Not for clinical use."
    "</div>",
    unsafe_allow_html=True,
)
