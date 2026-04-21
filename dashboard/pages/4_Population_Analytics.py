"""
Population Analytics Page.

Population-level risk distributions, demographics, and allele frequency analysis.
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

from genomicguard.config import DISEASES, MODELS_DIR, POPULATION_LABELS, DashboardConfig
from genomicguard.data.generator import GenomicDataGenerator
from genomicguard.models.risk_scorer import PolygenicRiskScorer

st.set_page_config(page_title="Population Analytics", page_icon="🌍", layout="wide")

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
def load_population_data():
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

        risk_scorers = {}
        for disease in DISEASES:
            dk = disease.lower().replace(" ", "_")
            path = MODELS_DIR / f"risk_scorer_{dk}.joblib"
            if path.exists():
                risk_scorers[dk] = PolygenicRiskScorer.load(path)

        return dataset, features, risk_scorers
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None


st.markdown("# 🌍 Population Analytics")
st.markdown("Population-level risk distributions and demographic analysis.")
st.markdown("---")

dataset, features, risk_scorers = load_population_data()

if dataset is None:
    st.error("⚠️ Run the pipeline first.")
    st.stop()

sample_meta = dataset["sample_metadata"]
clinical = dataset["clinical"]
phenotypes = dataset["phenotypes"]
populations = sample_meta["population"].values

# ── Population Distribution ────────────────────────────────────────────────
st.markdown("## 📊 Cohort Composition")

pop_counts = pd.Series(populations).value_counts()
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Sample Sizes")
    for pop, count in pop_counts.items():
        label = POPULATION_LABELS.get(pop, pop)
        pct = count / len(populations) * 100
        st.markdown(f"**{label}** ({pop}): {count} ({pct:.1f}%)")

    st.metric("Total Samples", len(populations))

with col2:
    fig = go.Figure(go.Pie(
        labels=[POPULATION_LABELS.get(p, p) for p in pop_counts.index],
        values=pop_counts.values,
        hole=0.45,
        marker_colors=[POP_COLORS.get(p, "#888") for p in pop_counts.index],
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Population Distribution",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Risk Distributions ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📈 Risk Score Distributions")

disease_key = st.selectbox(
    "Select Disease",
    list(risk_scorers.keys()),
    format_func=lambda x: x.replace("_", " ").title(),
)

scorer = risk_scorers[disease_key]
risk_scores = scorer.predict_risk(features)

fig = go.Figure()
for pop in sorted(np.unique(populations)):
    mask = populations == pop
    fig.add_trace(go.Violin(
        y=risk_scores[mask],
        name=f"{pop} ({POPULATION_LABELS.get(pop, '')})",
        box_visible=True,
        meanline_visible=True,
        line_color=POP_COLORS.get(pop, "#888"),
        fillcolor=POP_COLORS.get(pop, "#888"),
        opacity=0.6,
    ))

fig.update_layout(
    title=f"Risk Score Distribution — {disease_key.replace('_', ' ').title()}",
    yaxis_title="Risk Score",
    height=450,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    showlegend=True,
    yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
)
st.plotly_chart(fig, use_container_width=True)

# Risk statistics table
st.markdown("### Risk Score Summary Statistics")
risk_stats = []
for pop in sorted(np.unique(populations)):
    mask = populations == pop
    scores = risk_scores[mask]
    risk_stats.append({
        "Population": POPULATION_LABELS.get(pop, pop),
        "Mean Risk": f"{scores.mean():.3f}",
        "Median Risk": f"{np.median(scores):.3f}",
        "Std Dev": f"{scores.std():.3f}",
        "Min": f"{scores.min():.3f}",
        "Max": f"{scores.max():.3f}",
        "High Risk (>0.6)": f"{(scores > 0.6).mean():.1%}",
    })
st.dataframe(pd.DataFrame(risk_stats), use_container_width=True, hide_index=True)

# ── Clinical Feature Distributions ────────────────────────────────────────
st.markdown("---")
st.markdown("## 🩺 Clinical Feature Distributions by Population")

clinical_feature = st.selectbox(
    "Select Clinical Feature",
    ["age", "bmi", "systolic_bp", "fasting_glucose", "hba1c",
     "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"],
    format_func=lambda x: x.replace("_", " ").title(),
)

fig = go.Figure()
for pop in sorted(np.unique(populations)):
    mask = populations == pop
    fig.add_trace(go.Box(
        y=clinical[clinical_feature].values[mask],
        name=pop,
        marker_color=POP_COLORS.get(pop, "#888"),
        boxmean=True,
    ))

fig.update_layout(
    title=f"{clinical_feature.replace('_', ' ').title()} Distribution by Population",
    yaxis_title=clinical_feature.replace("_", " ").title(),
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Disease Prevalence ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🦠 Disease Prevalence by Population")

prev_data = []
for disease in DISEASES:
    dk = disease.lower().replace(" ", "_")
    label_col = f"{dk}_label"
    if label_col in phenotypes.columns:
        for pop in sorted(np.unique(populations)):
            mask = populations == pop
            prev = phenotypes[label_col].values[mask].mean()
            prev_data.append({
                "Disease": disease,
                "Population": pop,
                "Prevalence": prev,
            })

if prev_data:
    prev_df = pd.DataFrame(prev_data)
    fig = px.bar(
        prev_df,
        x="Population",
        y="Prevalence",
        color="Disease",
        barmode="group",
        color_discrete_sequence=["#6366f1", "#f59e0b", "#10b981"],
    )
    fig.update_layout(
        title="Disease Prevalence by Population",
        yaxis_title="Prevalence",
        yaxis_tickformat=".0%",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PCA Population Structure ──────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🧬 Population Structure (PCA)")

if "pc_1" in features.columns and "pc_2" in features.columns:
    fig = go.Figure()
    for pop in sorted(np.unique(populations)):
        mask = populations == pop
        fig.add_trace(go.Scatter(
            x=features["pc_1"].values[mask],
            y=features["pc_2"].values[mask],
            mode="markers",
            name=f"{pop} ({POPULATION_LABELS.get(pop, '')})",
            marker=dict(
                color=POP_COLORS.get(pop, "#888"),
                size=5,
                opacity=0.6,
            ),
        ))

    fig.update_layout(
        title="PCA — Population Clustering (PC1 vs PC2)",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
        yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **PCA plot** shows genetic ancestry clustering. Clear separation between populations "
        "indicates the synthetic data correctly captures population structure, which is critical "
        "for ancestry-adjusted risk scoring."
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "⚠️ This is a research demonstration. Not for clinical use."
    "</div>",
    unsafe_allow_html=True,
)
