# Lab 20 — Time Series Diagnostics and Advanced Decomposition

## Objective
Diagnose and fix common failure modes in time-series decomposition and stationarity testing, then extend the corrected analysis with MSTL, block bootstrap, and per-regime stationarity diagnostics. Ship everything as a reusable Python module and an interactive Streamlit dashboard over the FRED API.

## Methodology
- Pulled FRED series `RSXFSN` (not-seasonally-adjusted retail sales), `GDPC1` (real GDP), and a simulated hourly electricity demand series with daily and weekly cycles.
- Part 1 (diagnose): corrected a broken STL decomposition where additive STL was applied to multiplicative retail-sales data. Log-transformed before STL and verified the seasonal amplitude ratio fell into the 0.7 to 1.3 range.
- Part 2 (diagnose): corrected a misspecified ADF test that used `regression='n'` on trending GDP. Switched to `regression='ct'`, added KPSS, and applied the 2x2 decision table to confirm GDP is non-stationary.
- Part 3 (extend): applied `MSTL` to the electricity demand series with `periods=[24, 168]`, confirmed that the residual standard deviation matched the simulated noise level and each seasonal column captured the correct period.
- Part 4 (extend): implemented a moving block bootstrap on the STL residuals of log real GDP to produce a 90 percent confidence band on the trend. Preserved autocorrelation via overlapping eight-quarter blocks.
- Part 5 (extend): used `ruptures.Pelt` with an RBF cost to detect structural breaks in quarterly GDP growth, then ran ADF and KPSS per regime to surface segments where stationarity conclusions diverge from the full-sample view.
- Part 6 (module): packaged the workflow into `src/decompose.py` with `run_stl`, `run_mstl`, `test_stationarity`, `detect_breaks`, and `block_bootstrap_trend`. Every function has type hints, docstrings, and explicit error handling; the `__main__` block runs a smoke test against live FRED data.
- AI expansion: built `streamlit_app.py`, a FRED Decomposition Explorer that lets the user pick any FRED series, choose STL or MSTL, tune periods and penalties, run the 2x2 stationarity test, overlay PELT breakpoints, and optionally compute block-bootstrap trend bands.

## Key Findings
- Applying additive STL directly to retail sales produces a seasonal component whose amplitude ratio between the last and first year exceeds 2, a textbook multiplicative-seasonality failure. Log-transforming drops the ratio into the 0.7 to 1.3 range.
- On real GDP levels, ADF with `regression='ct'` fails to reject a unit root and KPSS rejects stationarity, so the 2x2 verdict is non-stationary. First-differencing flips both tests, which matches the standard I(1) classification of GDP.
- MSTL reproduces the injected noise standard deviation almost exactly on the electricity simulation, while STL with a single period badly mixes the weekly cycle into the residual.
- The bootstrap band on log real GDP is visibly wider around the 2008 and 2020 recessions than during expansions, because residuals there are larger and more autocorrelated.
- PELT detects structural breaks around the 1980 disinflation, the 2008 financial crisis, and the 2020 pandemic. Per-regime stationarity tests confirm that GDP growth is stationary within each regime even though the full sample looks non-stationary on KPSS.

## Reproducing
1. Put your FRED API key in `../config.py` as `FRED_API_KEY = "..."` (gitignored).
2. `pip install -r requirements.txt`.
3. Run `lab-ch20-diagnostic.ipynb` top to bottom.
4. For the dashboard: from the `Lab20/` folder run `streamlit run streamlit_app.py`.

## Stack
Python 3.13, pandas, numpy, statsmodels, ruptures, plotly, streamlit, fredapi, matplotlib.
