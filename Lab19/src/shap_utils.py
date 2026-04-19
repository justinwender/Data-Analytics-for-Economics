"""Reusable SHAP analysis utilities for tree-based regressors.

Functions:
    explain_prediction(model, X, idx) -> matplotlib.figure.Figure
    global_importance(model, X, max_display=10) -> matplotlib.figure.Figure
    compare_importance(model, X, y, n_repeats=5, random_state=42) -> pd.DataFrame
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


def _check_tree_model(model: BaseEstimator) -> None:
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"Expected a fitted tree-based estimator with feature_importances_, got "
            f"{type(model).__name__}."
        )


def _build_shap_values(model: BaseEstimator, X: pd.DataFrame) -> shap.Explanation:
    _check_tree_model(model)
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])
    return shap.Explanation(values=raw, base_values=base, data=X.values,
                            feature_names=list(X.columns))


def explain_prediction(
    model: BaseEstimator,
    X: pd.DataFrame,
    idx: int,
) -> plt.Figure:
    """Return a SHAP waterfall plot for a single observation.

    Args:
        model: Fitted tree-based regressor (RandomForestRegressor, GradientBoosting, ...).
        X: Feature dataframe used at inference time.
        idx: Positional index of the observation to explain.

    Returns:
        The current matplotlib Figure with the waterfall plot.

    Raises:
        IndexError: If idx is out of range.
        TypeError: If the model is not a tree-based estimator.
    """
    if not (0 <= idx < len(X)):
        raise IndexError(f"idx={idx} is out of range for X of length {len(X)}.")

    explanation = _build_shap_values(model, X)
    single = shap.Explanation(
        values=explanation.values[idx],
        base_values=explanation.base_values,
        data=explanation.data[idx],
        feature_names=explanation.feature_names,
    )
    shap.plots.waterfall(single, show=False)
    return plt.gcf()


def global_importance(
    model: BaseEstimator,
    X: pd.DataFrame,
    max_display: int = 10,
) -> plt.Figure:
    """Return a SHAP beeswarm plot summarising global feature importance."""
    explanation = _build_shap_values(model, X)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    return plt.gcf()


def compare_importance(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Side-by-side ranking of MDI, permutation importance, and SHAP.

    Permutation importance is reported on (X, y); SHAP mean-|value| is computed on
    the passed-in X. Column values are ranks (1 = most important) so the frame
    can be sorted or diffed directly. Raw magnitudes are also attached with a
    ``_raw`` suffix for reference.
    """
    _check_tree_model(model)

    mdi = pd.Series(model.feature_importances_, index=X.columns, name="mdi_raw")

    perm = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    perm_mean = pd.Series(perm.importances_mean, index=X.columns, name="perm_raw")

    explanation = _build_shap_values(model, X)
    shap_mean = pd.Series(
        np.abs(explanation.values).mean(axis=0),
        index=X.columns,
        name="shap_raw",
    )

    raw = pd.concat([mdi, perm_mean, shap_mean], axis=1)
    ranks = raw.rank(ascending=False, method="min").astype(int)
    ranks.columns = ["mdi_rank", "perm_rank", "shap_rank"]
    return pd.concat([ranks, raw], axis=1).sort_values("shap_rank")


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X_full = pd.DataFrame(data.data, columns=data.feature_names)
    y_full = data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

    # Smoke tests on a small subset so the module imports quickly.
    sample = X_te.iloc[:200].reset_index(drop=True)

    explain_prediction(rf, sample, idx=0)
    plt.close()

    global_importance(rf, sample, max_display=8)
    plt.close()

    cmp = compare_importance(rf, sample, y_te[:200], n_repeats=3)
    print(cmp)
