"""decompose.py — Time Series Decomposition & Diagnostics Module.

Reusable functions for STL / MSTL decomposition, stationarity testing,
structural break detection, and block bootstrap uncertainty on economic
time series.

Course: ECON 5200, Lab 20.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import adfuller, kpss


def _ensure_positive(series: pd.Series) -> None:
    if (series <= 0).any():
        raise ValueError(
            "log_transform=True requires strictly positive values; "
            "series contains zeros or negatives."
        )


def run_stl(
    series: pd.Series,
    period: int = 12,
    log_transform: bool = True,
    robust: bool = True,
):
    """Apply STL decomposition with optional log-transform.

    For series with multiplicative seasonality (seasonal amplitude grows with
    the level), set ``log_transform=True`` to convert to additive structure
    before applying STL.

    Args:
        series: Time series with a DatetimeIndex and a set frequency.
        period: Seasonal period (12 for monthly, 4 for quarterly).
        log_transform: Log-transform before STL for multiplicative data.
        robust: Use robust fitting to down-weight outliers.

    Returns:
        A fitted STL result object with ``.trend``, ``.seasonal``, ``.resid``.

    Raises:
        ValueError: If ``log_transform=True`` and ``series`` has non-positive values.
    """
    if log_transform:
        _ensure_positive(series)
        working = np.log(series)
    else:
        working = series.copy()
    return STL(working, period=period, robust=robust).fit()


def run_mstl(
    series: pd.Series,
    periods: list[int],
    log_transform: bool = False,
):
    """Apply MSTL decomposition for multiple seasonal periods.

    Useful when a series has more than one seasonal cycle (for example hourly
    electricity demand has a 24-hour daily cycle and a 168-hour weekly cycle).

    Args:
        series: Time series with a DatetimeIndex.
        periods: List of seasonal periods to extract.
        log_transform: Log-transform for multiplicative data.

    Returns:
        A fitted MSTL result object. ``result.seasonal`` is a DataFrame with
        one column per requested period.
    """
    if log_transform:
        _ensure_positive(series)
        working = np.log(series)
    else:
        working = series.copy()
    return MSTL(working, periods=periods).fit()


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """Run ADF and KPSS and return a 2x2 decision-table verdict.

    ADF null: unit root (non-stationary). KPSS null: stationary. Combining both
    guards against either test's weaknesses:
      - reject ADF, fail to reject KPSS -> stationary
      - fail to reject ADF, reject KPSS -> non-stationary
      - reject both -> contradictory (usually a sign of structural break)
      - fail to reject both -> inconclusive (often short samples)

    Args:
        series: Time series to test.
        alpha: Significance level for both tests.

    Returns:
        Dictionary with keys ``adf_stat``, ``adf_p``, ``kpss_stat``,
        ``kpss_p``, ``verdict``.
    """
    clean = series.dropna()
    adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC", regression="c")
    kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")

    adf_rejects = adf_p < alpha
    kpss_rejects = kpss_p < alpha
    if adf_rejects and not kpss_rejects:
        verdict = "stationary"
    elif not adf_rejects and kpss_rejects:
        verdict = "non-stationary"
    elif adf_rejects and kpss_rejects:
        verdict = "contradictory"
    else:
        verdict = "inconclusive"

    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "verdict": verdict,
    }


def detect_breaks(series: pd.Series, pen: float = 10, model: str = "rbf") -> list[pd.Timestamp]:
    """Detect structural breaks with the PELT algorithm.

    Args:
        series: Time series with a DatetimeIndex.
        pen: Penalty parameter; higher values produce fewer breaks.
        model: Cost model passed to ``ruptures.Pelt`` (``l1``, ``l2``, ``rbf``).

    Returns:
        List of break timestamps. PELT's final "break" at ``len(series)`` is
        dropped because it just marks the end of the sample.
    """
    clean = series.dropna()
    algo = rpt.Pelt(model=model).fit(clean.values)
    breakpoints = algo.predict(pen=pen)
    # Drop the terminal index PELT returns, map the rest to dates
    return [clean.index[bp - 1] for bp in breakpoints if bp < len(clean)]


def block_bootstrap_trend(
    series: pd.Series,
    period: int,
    n_bootstrap: int = 200,
    block_size: int = 8,
    alpha: float = 0.10,
    log_transform: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Block bootstrap confidence bands for the STL trend.

    Resample overlapping blocks of the STL residuals, add them back to
    ``trend + seasonal``, and re-fit STL. The pointwise quantiles across
    bootstrap replications give a confidence band that preserves the
    autocorrelation of the residuals (which iid resampling destroys).

    Args:
        series: Time series with a DatetimeIndex and a set frequency.
        period: Seasonal period passed to STL.
        n_bootstrap: Number of bootstrap replications.
        block_size: Length of the resampled residual blocks.
        alpha: Total tail probability (``alpha=0.10`` gives a 90% band).
        log_transform: If True, work on ``log(series)`` and return trend in logs.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame indexed like the (log-)series with columns
        ``trend``, ``lower``, ``upper``.
    """
    if log_transform:
        _ensure_positive(series)
        working = np.log(series)
    else:
        working = series.copy()

    stl_fit = STL(working, period=period, robust=True).fit()
    n = len(working)
    rng = np.random.default_rng(random_state)

    boot = np.zeros((n_bootstrap, n))
    resid = stl_fit.resid.values
    for b in range(n_bootstrap):
        boot_resid = np.empty(n)
        idx = 0
        while idx < n:
            start = rng.integers(0, n - block_size + 1)
            end = min(idx + block_size, n)
            boot_resid[idx:end] = resid[start:start + (end - idx)]
            idx = end
        reconstructed = pd.Series(
            stl_fit.trend.values + stl_fit.seasonal.values + boot_resid,
            index=working.index,
        )
        reconstructed.index.freq = working.index.freq
        boot[b] = STL(reconstructed, period=period, robust=True).fit().trend.values

    lower = np.quantile(boot, alpha / 2, axis=0)
    upper = np.quantile(boot, 1 - alpha / 2, axis=0)
    return pd.DataFrame(
        {"trend": stl_fit.trend.values, "lower": lower, "upper": upper},
        index=working.index,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import FRED_API_KEY

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    retail = fred.get_series("RSXFSN", observation_start="2000-01-01").dropna()
    retail.index = pd.DatetimeIndex(retail.index)
    retail.index.freq = "MS"

    stl_res = run_stl(retail, period=12, log_transform=True)
    print("run_stl: trend range =", (stl_res.trend.min(), stl_res.trend.max()))

    gdp = fred.get_series("GDPC1", observation_start="1960-01-01").dropna()
    gdp.index = pd.DatetimeIndex(gdp.index)
    gdp.index.freq = "QS"

    print("Stationarity on GDP:", test_stationarity(gdp))
    print("Stationarity on GDP diff:", test_stationarity(gdp.diff().dropna()))

    print("Breaks in GDP growth:", detect_breaks(gdp.pct_change().dropna() * 100, pen=10))

    bands = block_bootstrap_trend(gdp, period=4, n_bootstrap=50, block_size=8)
    print("Bootstrap band head:")
    print(bands.head())
