"""Interactive Random Forest / SHAP dashboard for Lab 19.

Run with:
    cd Lab19
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from shap_utils import global_importance, compare_importance  # noqa: E402

st.set_page_config(page_title="Lab 19: RF + SHAP Explorer", layout="wide")
RANDOM_STATE = 42


@st.cache_data
def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    return X, data.target, list(data.feature_names)


@st.cache_resource
def train_models(n_estimators: int, max_features: int):
    X, y, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    ).fit(X_tr, y_tr)

    return X_tr, X_te, y_tr, y_te, ridge, rf, gbr


st.title("Lab 19 — Random Forest + SHAP Explorer")
st.caption(
    "California Housing regression. Adjust the RF hyperparameters in the sidebar "
    "and watch SHAP + model-comparison panels update. Models use random_state=42."
)

with st.sidebar:
    st.header("Random Forest hyperparameters")
    n_estimators = st.slider("n_estimators", 10, 500, 200, step=10)
    max_features = st.slider("max_features (integer count)", 1, 8, 4, step=1)
    max_display = st.slider("Beeswarm: top-N features", 4, 8, 8, step=1)
    ranking = st.radio("Importance metric",
                       options=["MDI", "Permutation", "SHAP (mean |value|)"])

X_tr, X_te, y_tr, y_te, ridge, rf, gbr = train_models(n_estimators, max_features)

# ---- KPIs ----
col1, col2, col3 = st.columns(3)
for col, name, model in [
    (col1, "Ridge", ridge),
    (col2, f"RF (trees={n_estimators}, m={max_features})", rf),
    (col3, "Gradient Boosting", gbr),
]:
    train_r2 = r2_score(y_tr, model.predict(X_tr))
    test_r2 = r2_score(y_te, model.predict(X_te))
    col.metric(name, f"Test R² = {test_r2:.3f}", delta=f"Train−Test gap: {train_r2 - test_r2:+.3f}")

# ---- Comparison bar ----
scores = pd.DataFrame({
    "Model": ["Ridge", f"RF ({n_estimators} trees)", "Gradient Boosting"],
    "Test R²": [r2_score(y_te, m.predict(X_te)) for m in (ridge, rf, gbr)],
})
st.plotly_chart(px.bar(scores, x="Model", y="Test R²", range_y=(0, 1),
                        title="Held-out Test R²"), use_container_width=True)

# ---- Importance panel ----
st.subheader("Feature importance")

sample = X_te.iloc[:400].reset_index(drop=True)
y_sample = y_te[:400]

if ranking == "MDI":
    imp = pd.Series(rf.feature_importances_, index=X_te.columns).sort_values()
    st.plotly_chart(
        px.bar(imp, orientation="h", title="Mean Decrease in Impurity (MDI)"),
        use_container_width=True,
    )
elif ranking == "Permutation":
    perm = permutation_importance(rf, sample, y_sample, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_te.columns).sort_values()
    st.plotly_chart(
        px.bar(imp, orientation="h", title="Permutation Importance (test subset)"),
        use_container_width=True,
    )
else:
    fig = global_importance(rf, sample, max_display=max_display)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    st.image(buf.getvalue(), caption="SHAP beeswarm on a test subset")
    plt.close("all")

# ---- Full ranking comparison ----
with st.expander("Side-by-side rank comparison (MDI vs. permutation vs. SHAP)"):
    cmp = compare_importance(rf, sample, y_sample, n_repeats=3, random_state=RANDOM_STATE)
    st.dataframe(cmp)

# ---- Inline SHAP waterfall for a single observation ----
st.subheader("Per-observation SHAP waterfall")
idx = st.slider("Test observation index (within first 400 rows)", 0, len(sample) - 1, 0)
explainer = shap.TreeExplainer(rf)
vals = explainer.shap_values(sample)
base = explainer.expected_value
if isinstance(base, (list, np.ndarray)):
    base = float(np.asarray(base).ravel()[0])
expl = shap.Explanation(values=vals[idx], base_values=base, data=sample.iloc[idx].values,
                        feature_names=list(sample.columns))
fig, _ = plt.subplots(figsize=(8, 4))
shap.plots.waterfall(expl, show=False)
buf = io.BytesIO()
plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=130)
st.image(buf.getvalue())
plt.close("all")

st.caption(
    "TreeExplainer vs. KernelExplainer: TreeExplainer computes exact SHAP values "
    "for tree ensembles in polynomial time by exploiting the tree structure. "
    "KernelExplainer is model-agnostic but much slower and approximate. Use "
    "TreeExplainer whenever your model is tree-based."
)
