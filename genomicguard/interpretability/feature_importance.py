"""
Feature Importance Analysis.

Provides multiple approaches to understanding feature importance:
- Permutation importance (model-agnostic)
- SHAP-based importance
- Clinical relevance ranking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis.

    Combines multiple importance measures to provide a robust
    ranking of features, with special attention to clinical
    relevance and population-specific patterns.
    """

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self._results = {}

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Compute permutation importance.

        More robust than built-in feature importance for correlated features.
        """
        result = permutation_importance(
            self.model,
            X.values,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="roc_auc",
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        self._results["permutation"] = importance_df
        return importance_df

    def compute_population_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        populations: np.ndarray,
        n_repeats: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute permutation importance stratified by population.

        Reveals whether the model relies on different features
        for different population groups.
        """
        pop_importance = {}

        for pop in np.unique(populations):
            mask = populations == pop
            X_pop = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
            y_pop = y[mask]

            if len(np.unique(y_pop)) < 2:
                continue  # Skip if only one class

            result = permutation_importance(
                self.model,
                X_pop.values if isinstance(X_pop, pd.DataFrame) else X_pop,
                y_pop,
                n_repeats=n_repeats,
                random_state=42,
                scoring="roc_auc",
            )

            pop_importance[pop] = pd.DataFrame({
                "feature": self.feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        self._results["population_stratified"] = pop_importance
        return pop_importance

    def rank_clinical_relevance(
        self,
        shap_importance: Optional[pd.DataFrame] = None,
        permutation_importance_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create a unified clinical relevance ranking.

        Combines multiple importance measures and categorizes features
        by their clinical interpretability.
        """
        # Categorize features
        clinical_features = [f for f in self.feature_names if f.startswith("clinical_")]
        genetic_features = [f for f in self.feature_names if f.startswith("prs_") or f.startswith("pc_")]
        interaction_features = [f for f in self.feature_names if f.startswith("interact_")]
        composite_features = [f for f in self.feature_names if "composite" in f or "interaction" in f]

        categories = {}
        for f in self.feature_names:
            if f in clinical_features:
                categories[f] = "Clinical"
            elif f in genetic_features:
                categories[f] = "Genetic"
            elif f in interaction_features:
                categories[f] = "Gene Interaction"
            elif f in composite_features:
                categories[f] = "Clinical Composite"
            else:
                categories[f] = "Other"

        ranking = pd.DataFrame({
            "feature": self.feature_names,
            "category": [categories[f] for f in self.feature_names],
        })

        # Add importance scores if available
        if permutation_importance_df is not None:
            perm_scores = dict(zip(
                permutation_importance_df["feature"],
                permutation_importance_df["importance_mean"],
            ))
            ranking["permutation_importance"] = [
                perm_scores.get(f, 0) for f in self.feature_names
            ]

        if shap_importance is not None:
            shap_scores = dict(zip(
                shap_importance["feature"],
                shap_importance["mean_abs_shap"],
            ))
            ranking["shap_importance"] = [
                shap_scores.get(f, 0) for f in self.feature_names
            ]

        # Compute combined rank
        rank_cols = [c for c in ranking.columns if "importance" in c]
        if rank_cols:
            for col in rank_cols:
                ranking[f"{col}_rank"] = ranking[col].rank(ascending=False)
            rank_rank_cols = [c for c in ranking.columns if c.endswith("_rank")]
            ranking["combined_rank"] = ranking[rank_rank_cols].mean(axis=1)
            ranking = ranking.sort_values("combined_rank").reset_index(drop=True)

        return ranking

    def get_summary(self) -> Dict:
        """Return a summary of all importance analyses."""
        return {
            "n_features": len(self.feature_names),
            "methods_computed": list(self._results.keys()),
            "results": self._results,
        }
