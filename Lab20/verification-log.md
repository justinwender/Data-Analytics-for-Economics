# Verification Log — Lab 20

## P.R.I.M.E. prompt

```
[Prep]    Senior Python data scientist specialising in time series analysis,
          FRED data, and production-grade dashboards.
[Request] Extend a lab-built src/decompose.py with run_mstl and
          block_bootstrap_trend. Then build a Streamlit app that lets a user
          enter a FRED series id, pick STL or MSTL, adjust period / robust
          flags, run ADF + KPSS, overlay PELT breakpoints, and compute
          bootstrap trend bands on demand.
[Iterate] Use statsmodels, ruptures, plotly, streamlit, fredapi. Preserve
          the FRED_API_KEY pattern (import from gitignored config.py).
          Auto-detect frequency on arbitrary FRED pulls. Cache FRED calls.
[Mechanism] Comment on why block bootstrap preserves autocorrelation and
            why i.i.d. resampling breaks it; why STL vs MSTL differ; why
            log-transform is the fix for multiplicative data.
[Evaluate] Dashboard verdicts on GDP (non-stationary in levels, stationary
           in first differences) must match the notebook's 2x2 verdicts.
           run_mstl on simulated demand must return a two-column seasonal.
           block_bootstrap_trend bands must be wider at 2008Q4 than 2019Q4.
```

## What the AI generated
- `src/decompose.py` extended with `run_mstl` and `block_bootstrap_trend`, each with docstrings, type hints, and argument validation.
- `streamlit_app.py` with FRED id input, auto-frequency detection, STL/MSTL toggle, period sliders, PELT penalty slider, a stationarity table, an overlay of breakpoints on the raw series, and an optional bootstrap-band panel.

## What I changed
- Replaced a raw `np.random.randint` loop with `np.random.default_rng(seed).integers` so bootstrap results are reproducible from a seed argument.
- Dropped PELT's terminal breakpoint (which just marks the end of the series) from `detect_breaks` before returning dates, so the output is always "real" breakpoints.
- Forced `series.index.freq` to the inferred frequency inside `load_fred_series` because statsmodels refuses to fit STL on an un-frequencied DatetimeIndex.
- Switched the AI-generated `st.cache` decorators to `st.cache_data` / `st.cache_resource` (the old API is deprecated).
- Added an `_ensure_positive` guard so `log_transform=True` fails fast with a clear error when the series has zeros or negatives.

## What I verified
- `python src/decompose.py` runs its `__main__` block against live FRED data and reports stationarity verdicts matching the notebook.
- Notebook cell 25 module-test block passes all asserts.
- In the dashboard: `GDPC1` at defaults returns `non-stationary` in levels and `stationary` in first differences, which matches the notebook's 2x2 result.
- Bootstrap-band width at 2008Q4 is ~2x the width at 2019Q4, which matches the Part 4 verification checkpoint.
