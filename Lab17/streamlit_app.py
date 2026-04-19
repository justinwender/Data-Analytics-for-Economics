"""Interactive NY Fed yield curve recession probability dashboard.

Run with:
    streamlit run streamlit_app.py
"""
import sys
from pathlib import Path
# Walk up from this file to find the gitignored config.py at the repo root.
_here = Path(__file__).resolve().parent
for candidate in [_here, *_here.parents]:
    if (candidate / "config.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
from config import FRED_API_KEY

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import fredapi
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="NY Fed Recession Probability", layout="wide")


@st.cache_data(ttl=60 * 60 * 24)
def load_fred_data():
    fred = fredapi.Fred(api_key=FRED_API_KEY)
    spread = fred.get_series("T10Y3M", observation_start="1970-01-01")
    recession = fred.get_series("USREC", observation_start="1970-01-01")
    spread_m = spread.resample("ME").last()
    recession_m = recession.resample("ME").max()
    return pd.DataFrame({"yield_spread": spread_m, "recession": recession_m}).dropna()


def build_lagged_frame(df: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    out = df.copy()
    out[f"spread_lag{lag_months}"] = out["yield_spread"].shift(lag_months)
    return out.dropna()


@st.cache_data
def fit_logit(X: np.ndarray, y: np.ndarray) -> dict:
    m = LogisticRegression(random_state=42)
    m.fit(X, y)
    return {"intercept": float(m.intercept_[0]), "coef": float(m.coef_[0][0])}


def predict_proba(intercept: float, coef: float, x: np.ndarray) -> np.ndarray:
    z = intercept + coef * x
    return 1.0 / (1.0 + np.exp(-z))


def bootstrap_bands(df_fit: pd.DataFrame, lag_col: str,
                    n_boot: int = 300, alpha: float = 0.10,
                    rng_seed: int = 42) -> pd.DataFrame:
    # Resample (x, y) pairs with replacement, refit logit each time, collect
    # P(recession|x) for every observed month. Bootstrap yields plausible
    # (intercept, coef) draws from the MLE's sampling distribution; applying
    # each draw to the observed spread gives a pointwise confidence envelope.
    # Narrow band = stable estimates. Wide band = sparse data in that region
    # of the predictor (e.g., deep inversions).
    rng = np.random.default_rng(rng_seed)
    n = len(df_fit)
    X = df_fit[[lag_col]].values
    y = df_fit["recession"].values

    probs = np.zeros((n_boot, n))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = LogisticRegression(random_state=42)
        m.fit(X[idx], y[idx])
        probs[b] = m.predict_proba(X)[:, 1]

    lo = np.quantile(probs, alpha / 2, axis=0)
    hi = np.quantile(probs, 1 - alpha / 2, axis=0)
    return pd.DataFrame({"lower": lo, "upper": hi}, index=df_fit.index)


st.title("NY Fed Yield Curve Recession Probability — Interactive")

df_raw = load_fred_data()

with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Prediction horizon (months)", 6, 18, 12, step=6)
    n_boot = st.slider("Bootstrap samples", 100, 500, 300, step=100)
    st.markdown("---")
    st.caption("Data: FRED T10Y3M (daily → month-end) and USREC (monthly).")

df_fit = build_lagged_frame(df_raw, horizon)
lag_col = f"spread_lag{horizon}"

fit = fit_logit(df_fit[[lag_col]].values, df_fit["recession"].values)
df_fit["prob"] = predict_proba(fit["intercept"], fit["coef"], df_fit[lag_col].values)

bands = bootstrap_bands(df_fit, lag_col, n_boot=n_boot)

current_spread = float(df_raw["yield_spread"].iloc[-1])
latest_input = float(df_fit[lag_col].iloc[-1])
latest_prob = float(df_fit["prob"].iloc[-1])

with st.sidebar:
    st.markdown("### Current state")
    st.metric("Spread today (pp)", f"{current_spread:+.2f}")
    st.metric(f"Spread {horizon}m ago (model input)", f"{latest_input:+.2f}")
    st.metric(f"P(Recession within {horizon}m)", f"{latest_prob * 100:.1f}%")
    st.metric("Odds ratio per +1pp spread", f"{np.exp(fit['coef']):.3f}")

plot_from = "2000"
d = df_fit.loc[plot_from:].copy()
b = bands.loc[plot_from:]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=d.index.tolist() + d.index[::-1].tolist(),
    y=(b["upper"] * 100).tolist() + (b["lower"][::-1] * 100).tolist(),
    fill="toself", fillcolor="rgba(31,119,180,0.18)", line=dict(width=0),
    hoverinfo="skip", name="90% bootstrap band"))
fig.add_trace(go.Scatter(x=d.index, y=d["prob"] * 100, name=f"P(Recession in {horizon}m)",
                         line=dict(color="#1f77b4", width=2.5)))

in_rec = d["recession"].values
dates = d.index
start = None
for i, v in enumerate(in_rec):
    if v == 1 and start is None:
        start = dates[i]
    elif v == 0 and start is not None:
        fig.add_vrect(x0=start, x1=dates[i], fillcolor="#d62728", opacity=0.18, line_width=0)
        start = None
if start is not None:
    fig.add_vrect(x0=start, x1=dates[-1], fillcolor="#d62728", opacity=0.18, line_width=0)

fig.update_layout(title=f"Yield Curve Recession Probability ({horizon}-month horizon)",
                  yaxis_title="Probability (%)", yaxis_range=[0, 100],
                  xaxis_title="Date", height=520, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.caption("Band is the pointwise 90% interval across bootstrap refits of the logistic "
           "regression on resampled (x, y) pairs. Narrower bands = more stable estimates. "
           "Expect wider bands at the tails of the yield-spread distribution (deep "
           "inversions or very steep curves are sparsely sampled).")
