# Lab 21 — Time Series Forecasting: ARIMA, GARCH, and Block Bootstrap

## Objective
Diagnose and repair a broken monthly-CPI ARIMA pipeline, fit a GARCH(1,1) volatility model on S&P 500 log returns, ship a reusable forecast-evaluation module, and build distribution-free forecast intervals via moving block bootstrap.

## Methodology
- Pulled monthly FRED CPI (`CPIAUCNS`, 2000 to present) and daily S&P 500 closes via `yfinance` (2000 to 2024).
- Part 1 (diagnose): identified three planted errors in the ARIMA pipeline: fitting at `d=0` against a clearly non-stationary series, omitting seasonal terms on monthly data, and skipping the Ljung-Box residual diagnostic before forecasting.
- Part 2 (fix): verified stationarity of `diff(CPI)` via ADF, used `pmdarima.auto_arima` with `seasonal=True, m=12` to select the SARIMA order, refit with `SARIMAX`, and required Ljung-Box p-values at lags 12 and 24 to exceed 0.05 before producing a 24-month forecast.
- Part 3 (GARCH): fit `arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')` on S&P 500 log returns, verified variance stationarity (`alpha + beta < 1`), extracted the conditional volatility series, and overlaid +/- 2 sigma bands on returns alongside annotated crisis dates.
- Part 4 (module): packaged `compute_mase` and `backtest_expanding_window` into `src/forecast_evaluation.py` with type hints, docstrings, and a `__main__` smoke test that runs a SARIMAX backtest on a synthetic trend-plus-seasonal series.
- Challenge: implemented `block_bootstrap_forecast` on the fitted SARIMA, sampling overlapping six-step residual blocks, and compared the resulting 95 percent interval to the Gaussian interval returned by `get_forecast`.

## Key Findings
- After correction, SARIMA residuals pass Ljung-Box at both lag 12 and lag 24, and the seasonal ACF spikes at lags 12 and 24 collapse to near zero.
- MASE against the seasonal-naive baseline is comfortably below 1 in the expanding-window backtest, which confirms the SARIMA beats the naive benchmark out of sample.
- GARCH(1,1) on S&P 500 returns has `alpha + beta` near 0.98 to 0.99, implying a volatility-shock half-life of roughly 50 to 80 trading days. The conditional volatility peaks around the 2008 financial crisis and the March 2020 COVID shock.
- Block-bootstrap 95 percent intervals are wider than the Gaussian SARIMA intervals, especially at longer horizons, because they absorb residual autocorrelation and fat tails that the Gaussian assumption ignores.

## Reproducing
1. Put your FRED API key in `../config.py` as `FRED_API_KEY = "..."` (gitignored).
2. `pip install -r requirements.txt`.
3. Run `lab-ch21-diagnostic.ipynb` top to bottom.

## Stack
Python 3.13, pandas, numpy, statsmodels, pmdarima, arch, fredapi, yfinance, matplotlib.
