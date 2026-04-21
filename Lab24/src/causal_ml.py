"""causal_ml.py - Manual Double Machine Learning + subgroup CATE utilities.

Course: ECON 5200, Lab 24.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def _default_nuisance() -> BaseEstimator:
    return RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)


def manual_dml(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    outcome_learner: BaseEstimator | None = None,
    treatment_learner: BaseEstimator | None = None,
    n_folds: int = 2,
    random_state: int = 42,
) -> dict:
    """Manual cross-fitted Partially Linear DML (PLR).

    Residualises both ``Y`` and ``D`` on ``X`` with K-fold cross-fitting, then
    estimates theta by the IV-style formula
    ``theta = sum(D_tilde * Y_tilde) / sum(D_tilde * D)``.

    Standard error uses the influence-function / sandwich estimator
    ``se = sqrt(var(psi)) / sqrt(n)`` where
    ``psi = D_tilde * (Y_tilde - theta * D_tilde) / mean(D_tilde * D)``.

    Args:
        Y: Outcome vector, shape ``(n,)``.
        D: Treatment vector, shape ``(n,)``. Binary or continuous.
        X: Covariate matrix, shape ``(n, p)``.
        outcome_learner: sklearn-compatible regressor for ``Y ~ X``.
        treatment_learner: sklearn-compatible regressor for ``D ~ X``.
        n_folds: K-fold cross-fit splits.
        random_state: Seed for the KFold.

    Returns:
        Dict with ``theta``, ``se``, ``ci_low``, ``ci_high``, ``Y_tilde``,
        ``D_tilde``.
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    if not (len(Y) == len(D) == len(X)):
        raise ValueError("Y, D, X must have the same length.")

    outcome = outcome_learner or _default_nuisance()
    treatment = treatment_learner or _default_nuisance()

    n = len(Y)
    Y_tilde = np.zeros(n)
    D_tilde = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        l_model = clone(outcome).fit(X[train_idx], Y[train_idx])
        m_model = clone(treatment).fit(X[train_idx], D[train_idx])
        Y_tilde[test_idx] = Y[test_idx] - l_model.predict(X[test_idx])
        D_tilde[test_idx] = D[test_idx] - m_model.predict(X[test_idx])

    denom = float(np.sum(D_tilde * D))
    if denom == 0:
        raise ValueError("Degenerate treatment residuals; denominator is zero.")
    theta = float(np.sum(D_tilde * Y_tilde) / denom)

    psi = D_tilde * (Y_tilde - theta * D_tilde) / (denom / n)
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z = 1.959963984540054  # two-sided 95% normal quantile
    return {
        "theta": theta,
        "se": se,
        "ci_low": theta - z * se,
        "ci_high": theta + z * se,
        "Y_tilde": Y_tilde,
        "D_tilde": D_tilde,
    }


def cate_by_subgroup(
    cate_predictions: np.ndarray,
    subgroup: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Summarise CATE distribution within each subgroup.

    Args:
        cate_predictions: Length-``n`` vector of CATE estimates.
        subgroup: Length-``n`` series or array of subgroup labels.

    Returns:
        DataFrame indexed by subgroup with columns ``n``, ``mean``, ``std``,
        ``p25``, ``p50``, ``p75``. Useful for comparing against a coarse
        subgroup-DML ATE.
    """
    df = pd.DataFrame({"cate": np.asarray(cate_predictions), "group": np.asarray(subgroup)})
    grouped = df.groupby("group", observed=True)["cate"]
    summary = grouped.agg(
        n="count",
        mean="mean",
        std="std",
        p25=lambda s: np.percentile(s, 25),
        p50="median",
        p75=lambda s: np.percentile(s, 75),
    )
    return summary


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, p = 4000, 20
    TRUE_ATE = 3.0
    X = rng.normal(size=(n, p))
    propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
    D = rng.binomial(1, propensity).astype(float)
    Y = TRUE_ATE * D + 2 * X[:, 0] + 1.5 * X[:, 1] + 1.0 * X[:, 2] + rng.normal(0, 1, n)

    out = manual_dml(Y, D, X)
    print(f"theta = {out['theta']:.3f}  (true = {TRUE_ATE})")
    print(f"95% CI = [{out['ci_low']:.3f}, {out['ci_high']:.3f}]")

    # Subgroup summary
    q = pd.qcut(X[:, 0], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    cate = np.full(n, out["theta"]) + 0.5 * X[:, 0]  # pretend-heterogeneity
    print(cate_by_subgroup(cate, q))
