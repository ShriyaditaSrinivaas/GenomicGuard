"""
Visualization utilities for GenomicGuard.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def set_genomicguard_style():
    """Set consistent plot styling for GenomicGuard."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.family": "sans-serif",
    })


POPULATION_COLORS = {
    "EUR": "#6366f1",
    "AFR": "#f59e0b",
    "EAS": "#10b981",
    "SAS": "#ef4444",
    "AMR": "#8b5cf6",
}


def plot_risk_distribution(
    risk_scores: np.ndarray,
    populations: np.ndarray,
    title: str = "Risk Score Distribution by Population",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot risk score distributions stratified by population."""
    set_genomicguard_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for pop in sorted(np.unique(populations)):
        mask = populations == pop
        color = POPULATION_COLORS.get(pop, "#888888")
        ax.hist(
            risk_scores[mask],
            bins=30,
            alpha=0.5,
            label=pop,
            color=color,
            density=True,
        )

    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_fairness_comparison(
    group_metrics: Dict,
    metric_name: str = "auc_roc",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot side-by-side performance metrics across groups."""
    set_genomicguard_style()

    groups = list(group_metrics.keys())
    values = [group_metrics[g].get(metric_name, 0) or 0 for g in groups]
    colors = [POPULATION_COLORS.get(g, "#888888") for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(groups, values, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add reference line for overall
    overall_val = np.mean(values)
    ax.axhline(y=overall_val, color="#94a3b8", linestyle="--", alpha=0.7, label=f"Mean: {overall_val:.3f}")

    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"{metric_name.replace('_', ' ').title()} by Population")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    populations: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot calibration curves by population."""
    set_genomicguard_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

    from sklearn.calibration import calibration_curve

    for pop in sorted(np.unique(populations)):
        mask = populations == pop
        yt = y_true[mask]
        yp = y_prob[mask]

        if len(np.unique(yt)) >= 2:
            prob_true, prob_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy="uniform")
            color = POPULATION_COLORS.get(pop, "#888888")
            ax.plot(prob_pred, prob_true, "o-", color=color, label=pop, alpha=0.8)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curves by Population")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
