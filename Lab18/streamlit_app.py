"""Interactive fraud-detection threshold dashboard for Lab 18.

Launch with:
    cd Lab18
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Threshold Dashboard", layout="wide")
RANDOM_STATE = 42


@st.cache_data
def load_fraud_data() -> pd.DataFrame:
    """Locate creditcard.csv. Falls back to kagglehub download if not local."""
    local = Path(__file__).parent / "creditcard.csv"
    if local.exists():
        return pd.read_csv(local)
    try:
        import kagglehub

        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        return pd.read_csv(Path(path) / "creditcard.csv")
    except Exception as e:
        st.error(f"Could not load creditcard.csv. Place it next to streamlit_app.py "
                 f"or install kagglehub. Error: {e}")
        st.stop()


@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df.drop(columns=["Class", "Time"]).copy()
    y = df["Class"].values
    X["Amount"] = StandardScaler().fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train, y_train)
    y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

    # Random Forest is slower on this dataset; use a modest budget.
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    return X_test, y_test, y_prob_lr, y_prob_rf


df = load_fraud_data()
X_test, y_test, y_prob_lr, y_prob_rf = train_models(df)

st.title("Fraud Detection — Threshold & Cost Dashboard")
st.caption(
    "Slide the threshold to see the confusion matrix, Precision / Recall / F1, and "
    "a dollar-cost metric update live. Compare Logistic Regression against Random "
    "Forest on both ROC-AUC and PR-AUC in the second panel."
)

with st.sidebar:
    st.header("Controls")
    tau = st.slider("Classification threshold \u03c4", 0.01, 0.99, 0.50, step=0.01)
    cost_fn = st.number_input("Cost of missed fraud (FN)", value=800, step=50)
    cost_fp = st.number_input("Cost of false alarm (FP)", value=12, step=1)
    model_choice = st.radio("Model for threshold panel",
                             options=["Logistic Regression", "Random Forest"])

y_prob = y_prob_lr if model_choice == "Logistic Regression" else y_prob_rf
y_pred = (y_prob >= tau).astype(int)

# ---- KPIs ----
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
# Cost = FN * cost_fn + FP * cost_fp (missed frauds dominate when cost_fn >> cost_fp).
total_cost = int(fn * cost_fn + fp * cost_fp)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Flagged", f"{int(tp + fp):,}")
k2.metric("Precision", f"{prec:.2%}")
k3.metric("Recall", f"{rec:.2%}")
k4.metric("F1", f"{f1:.3f}")
k5.metric("Dollar cost", f"${total_cost:,}")

st.caption(
    f"Confusion matrix (\u03c4 = {tau:.2f}, {model_choice}): "
    f"TP={tp}, FN={fn}, FP={fp}, TN={tn:,}"
)

# ---- Confusion matrix plot ----
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                        display_labels=["Legitimate", "Fraud"]).plot(
    ax=ax_cm, cmap="Blues", values_format=",")
ax_cm.set_title(f"Confusion Matrix (\u03c4 = {tau:.2f})")

# ---- Cost curve across thresholds ----
taus = np.arange(0.01, 0.99, 0.01)
preds_matrix = (y_prob[:, None] >= taus[None, :]).astype(int)
fn_curve = ((preds_matrix == 0) & (y_test[:, None] == 1)).sum(axis=0)
fp_curve = ((preds_matrix == 1) & (y_test[:, None] == 0)).sum(axis=0)
cost_curve = fn_curve * cost_fn + fp_curve * cost_fp

fig_cost, ax_cost = plt.subplots(figsize=(7, 4))
ax_cost.plot(taus, cost_curve, color="#2563eb", lw=2)
ax_cost.axvline(tau, color="gray", linestyle=":", label=f"Current \u03c4 = {tau:.2f}")
best_tau = float(taus[int(np.argmin(cost_curve))])
ax_cost.axvline(best_tau, color="#dc2626", linestyle="--",
                 label=f"Cost-min \u03c4 = {best_tau:.2f}")
ax_cost.set_xlabel("Threshold \u03c4")
ax_cost.set_ylabel("Total dollar cost")
ax_cost.set_title("Cost vs. threshold")
ax_cost.legend()
fig_cost.tight_layout()

col_a, col_b = st.columns([1, 1])
col_a.pyplot(fig_cm)
col_b.pyplot(fig_cost)

# ---- Model comparison panel ----
st.subheader("Model comparison — Logistic Regression vs. Random Forest")

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)

roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
pr_auc_lr = auc(rec_lr, prec_lr)
pr_auc_rf = auc(rec_rf, prec_rf)

fig_cmp, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr_lr, tpr_lr, color="#2563eb", lw=2, label=f"LR (AUC={roc_auc_lr:.3f})")
axes[0].plot(fpr_rf, tpr_rf, color="#f97316", lw=2, label=f"RF (AUC={roc_auc_rf:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_title("ROC Curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend()

axes[1].plot(rec_lr, prec_lr, color="#2563eb", lw=2, label=f"LR (PR-AUC={pr_auc_lr:.3f})")
axes[1].plot(rec_rf, prec_rf, color="#f97316", lw=2, label=f"RF (PR-AUC={pr_auc_rf:.3f})")
axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.5,
                 label=f"baseline = {y_test.mean():.4f}")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].legend()
fig_cmp.tight_layout()
st.pyplot(fig_cmp)

st.caption(
    "PR-AUC is the more informative metric on imbalanced data (0.17% fraud rate): "
    "ROC-AUC rewards True Negatives, which are abundant, so a mediocre ranker can "
    "still look good on ROC. PR-AUC ignores TN entirely and focuses on fraud-class "
    "performance, which is what a fraud team actually cares about."
)

st.caption(
    "How the cost curve is computed: for every threshold \u03c4 we build predictions "
    "``y_prob >= \u03c4``, then compute ``FN * cost_fn + FP * cost_fp``. The red dashed "
    "line marks the cost-minimising \u03c4, which is typically below the F1-maximising "
    "\u03c4 whenever ``cost_fn > cost_fp`` (as in most real fraud problems)."
)
