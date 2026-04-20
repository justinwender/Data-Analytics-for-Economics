"""Interactive FRED decomposition dashboard for Lab 20.

Launch with:
    cd Lab20
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fredapi import Fred
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FRED_API_KEY  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))
from decompose import (  # noqa: E402
    block_bootstrap_trend,
    detect_breaks,
    run_mstl,
    run_stl,
    test_stationarity,
)

st.set_page_config(page_title="FRED Decomposition Explorer", layout="wide")


@st.cache_data(ttl=60 * 60 * 12)
def load_fred_series(series_id: str, start: str = "1960-01-01") -> pd.Series:
    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(series_id, observation_start=start).dropna()
    s.index = pd.DatetimeIndex(s.index)
    # Auto-detect the most common frequency
    inferred = pd.infer_freq(s.index) or "MS"
    s.index.freq = inferred
    s.name = series_id
    return s


st.title("FRED Decomposition Explorer")
st.caption("Lab 20: STL / MSTL / stationarity / PELT / block bootstrap on any FRED series.")

with st.sidebar:
    st.header("Data")
    series_id = st.text_input("FRED series id", value="RSXFSN",
                              help="Try RSXFSN (retail sales), GDPC1 (real GDP), INDPRO, CPIAUCSL.")
    start = st.text_input("Observation start", value="2000-01-01")

    st.header("Method")
    method = st.selectbox("Decomposition", ["STL", "MSTL"])
    log_transform = st.checkbox("Log-transform (multiplicative \u2192 additive)", value=True)
    robust = st.checkbox("Robust STL fitting", value=True)

    if method == "STL":
        period = st.slider("Seasonal period", 2, 52, 12, step=1)
    else:
        period_main = st.slider("Primary period", 2, 168, 24, step=1)
        period_secondary = st.slider("Secondary period", 6, 8760, 168, step=1)

    st.header("Diagnostics")
    pen = st.slider("PELT penalty", 1, 50, 10)
    run_bootstrap = st.checkbox("Compute block bootstrap bands", value=False)
    if run_bootstrap:
        n_boot = st.slider("Bootstrap replications", 50, 500, 150, step=50)
        block_size = st.slider("Block size", 2, 20, 8)

try:
    series = load_fred_series(series_id, start)
except Exception as e:
    st.error(f"Could not load FRED series '{series_id}': {e}")
    st.stop()

st.subheader(f"{series_id} \u2014 raw series")
fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                              name=series_id, line=dict(color="#2c3e50")))
fig_raw.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_raw, use_container_width=True)

# --- decomposition ---
if method == "STL":
    stl_res = run_stl(series, period=period, log_transform=log_transform, robust=robust)
    trend = stl_res.trend
    seasonal = stl_res.seasonal
    resid = stl_res.resid

    fig_dec = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             subplot_titles=("Trend", "Seasonal", "Residual"))
    fig_dec.add_trace(go.Scatter(x=trend.index, y=trend.values, line=dict(color="#e67e22")), row=1, col=1)
    fig_dec.add_trace(go.Scatter(x=seasonal.index, y=seasonal.values, line=dict(color="#27ae60")), row=2, col=1)
    fig_dec.add_trace(go.Scatter(x=resid.index, y=resid.values, line=dict(color="#c0392b")), row=3, col=1)
    fig_dec.update_layout(height=560, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_dec, use_container_width=True)
else:
    mstl_res = run_mstl(series, periods=[period_main, period_secondary], log_transform=log_transform)
    seasonal_df = mstl_res.seasonal
    fig_dec = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             subplot_titles=("Trend",
                                             f"Seasonal ({period_main})",
                                             f"Seasonal ({period_secondary})",
                                             "Residual"))
    fig_dec.add_trace(go.Scatter(x=mstl_res.trend.index, y=mstl_res.trend.values,
                                  line=dict(color="#e67e22")), row=1, col=1)
    fig_dec.add_trace(go.Scatter(x=seasonal_df.index, y=seasonal_df.iloc[:, 0].values,
                                  line=dict(color="#27ae60")), row=2, col=1)
    fig_dec.add_trace(go.Scatter(x=seasonal_df.index, y=seasonal_df.iloc[:, 1].values,
                                  line=dict(color="#8e44ad")), row=3, col=1)
    fig_dec.add_trace(go.Scatter(x=mstl_res.resid.index, y=mstl_res.resid.values,
                                  line=dict(color="#c0392b")), row=4, col=1)
    fig_dec.update_layout(height=700, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_dec, use_container_width=True)

# --- stationarity ---
st.subheader("Stationarity (ADF + KPSS)")
st_levels = test_stationarity(series)
st_diff = test_stationarity(series.diff().dropna())
df_results = pd.DataFrame({"levels": st_levels, "first difference": st_diff}).T
st.dataframe(df_results.style.format({
    "adf_stat": "{:.3f}", "adf_p": "{:.4f}",
    "kpss_stat": "{:.3f}", "kpss_p": "{:.4f}"}))

# --- structural breaks ---
st.subheader("Structural breaks (PELT)")
breaks = detect_breaks(series.pct_change().dropna() if series.min() > 0 else series,
                       pen=pen)
st.write(f"Breakpoints: {[b.strftime('%Y-%m-%d') for b in breaks]}")

fig_breaks = go.Figure()
fig_breaks.add_trace(go.Scatter(x=series.index, y=series.values,
                                 line=dict(color="#2c3e50", width=1.2), name=series_id))
for b in breaks:
    fig_breaks.add_vline(x=b, line_color="#c0392b", line_dash="dash", opacity=0.6)
fig_breaks.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_breaks, use_container_width=True)

# --- bootstrap bands ---
if run_bootstrap and method == "STL":
    st.subheader("Block bootstrap trend confidence band")
    bands = block_bootstrap_trend(
        series, period=period, n_bootstrap=n_boot, block_size=block_size,
        log_transform=log_transform,
    )
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=list(bands.index) + list(bands.index[::-1]),
        y=list(bands["upper"].values) + list(bands["lower"].values[::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.18)", line=dict(width=0),
        hoverinfo="skip", name="90% band"))
    fig_bb.add_trace(go.Scatter(x=bands.index, y=bands["trend"].values,
                                 line=dict(color="#e67e22", width=2), name="STL trend"))
    fig_bb.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_bb, use_container_width=True)
    st.caption(
        "Bands are pointwise quantiles across bootstrap replications where residuals are "
        "resampled in blocks, preserving autocorrelation. i.i.d. bootstrap would destroy "
        "autocorrelation and artificially shrink the bands. Expect wider bands around "
        "recessions because residuals are larger there."
    )
