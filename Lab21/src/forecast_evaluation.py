"""forecast_evaluation.py - MASE + expanding-window backtest utilities.

Course: ECON 5200, Lab 21.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    insample: np.ndarray,
    m: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    MASE scales the forecast's MAE by the MAE of an in-sample naive seasonal
    baseline (random walk when ``m=1``, seasonal random walk when ``m`` is the
    seasonal period). MASE < 1 means the model beats the naive benchmark.

    Args:
        actual: True out-of-sample values.
        forecast: Model predictions, same length as ``actual``.
        insample: Training-period observations used to calibrate the baseline.
        m: Seasonal step; 1 for random-walk baseline, 12 for monthly seasonal.

    Returns:
        MASE as a float.

    Raises:
        ValueError: If ``actual`` and ``forecast`` have different lengths, if
            ``insample`` is shorter than ``m``, or if the baseline MAE is zero.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    insample = np.asarray(insample, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"actual and forecast must match: {actual.shape} vs {forecast.shape}")
    if len(insample) <= m:
        raise ValueError(f"insample length {len(insample)} must exceed seasonal step m={m}")

    mae_forecast = float(np.mean(np.abs(actual - forecast)))
    mae_naive = float(np.mean(np.abs(insample[m:] - insample[:-m])))
    if mae_naive == 0:
        raise ValueError("Naive baseline MAE is zero; MASE is undefined.")
    return mae_forecast / mae_naive


def backtest_expanding_window(
    series: pd.Series,
    model_fn: Callable[[pd.Series], "object"],
    min_train: int = 120,
    horizon: int = 12,
    step: int = 12,
    seasonal_m: int = 1,
) -> pd.DataFrame:
    """Expanding-window backtest.

    Starting from the first ``min_train`` observations, fit the model, forecast
    the next ``horizon`` observations, record RMSE / MAE / MASE, then expand
    the training window by ``step`` observations and repeat until the end of
    the sample is reached.

    The ``model_fn`` argument is any callable that takes a training Series and
    returns a fitted object exposing ``.get_forecast(steps=...)`` (the
    statsmodels ARIMA / SARIMAX convention). Wrap anything else into a
    lightweight adapter before passing it in.

    Args:
        series: Full time series with a DatetimeIndex.
        model_fn: Factory that takes a training slice and returns a fitted model.
        min_train: Size of the initial training window.
        horizon: Forecast horizon at each step.
        step: Window-expansion size (also the cadence of refits).
        seasonal_m: Seasonal step for MASE; 1 disables seasonal naive.

    Returns:
        DataFrame indexed by training-window end date with columns
        ``train_size``, ``rmse``, ``mae``, ``mase``.
    """
    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be positive.")
    if min_train + horizon > len(series):
        raise ValueError("min_train + horizon exceeds length of series.")

    rows = []
    total = len(series)
    train_end = min_train
    while train_end + horizon <= total:
        train = series.iloc[:train_end]
        actual = series.iloc[train_end:train_end + horizon]

        fitted = model_fn(train)
        fc = fitted.get_forecast(steps=horizon)
        predicted = pd.Series(fc.predicted_mean, index=actual.index)

        errors = (actual - predicted).values
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        mase = compute_mase(actual.values, predicted.values, train.values, m=seasonal_m)

        rows.append({
            "train_end": train.index[-1],
            "train_size": len(train),
            "rmse": rmse,
            "mae": mae,
            "mase": mase,
        })
        train_end += step

    return pd.DataFrame(rows).set_index("train_end")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 360
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    trend = np.linspace(100, 200, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 1.5, n)
    series = pd.Series(trend + seasonal + noise, index=idx, name="synthetic")

    insample = series.iloc[:-24].values
    actual = series.iloc[-24:].values
    naive_forecast = np.full(24, series.iloc[-25])
    print("MASE (naive flat):", compute_mase(actual, naive_forecast, insample, m=12))

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def model_fn(train: pd.Series):
        return SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)

    bt = backtest_expanding_window(series, model_fn, min_train=120, horizon=12, step=12, seasonal_m=12)
    print(bt)
