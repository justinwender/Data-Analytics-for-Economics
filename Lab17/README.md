# Lab 17 — NY Fed Yield Curve Recession Model Replication

## Objective
Replicate the Federal Reserve Bank of New York's yield-curve recession probability model by fitting a logistic regression on FRED macroeconomic data to predict NBER-defined recessions twelve months ahead.

## Methodology
- Pulled monthly FRED series: `T10Y3M` (10-year minus 3-month Treasury spread, daily → month-end last) and `USREC` (NBER recession indicator, monthly max).
- Lagged the spread by twelve months so the target is `P(recession in next 12m)`.
- Fit a Linear Probability Model (OLS) as a baseline and documented out-of-bounds fitted probabilities.
- Fit `sklearn.linear_model.LogisticRegression` as the NY Fed replication; extracted the odds ratio and a 95% confidence interval via `statsmodels.Logit`.
- Produced the signature NY Fed-style chart: predicted probability time series with NBER recession shading and the yield-spread panel.
- Extension A: added the civilian unemployment rate (`UNRATE`) as a second predictor and compared odds ratios across the one- and two-predictor specifications.
- Extension B: compared cross-validated accuracy using `TimeSeriesSplit` (k = 3) against the naïve "always predict no recession" baseline.
- AI expansion: built a Streamlit dashboard (`streamlit_app.py`) with a horizon slider (6 / 12 / 18 months), a pointwise 90% bootstrap confidence band on the predicted probability, and a sidebar of real-time inputs and fitted odds ratio.

## Key Findings
- The Linear Probability Model generated probabilities below 0 and above 1 in the actual sample, the empirical version of the textbook critique of OLS on binary outcomes.
- The one-predictor logit produces a yield-spread odds ratio around 0.5 per percentage point, meaning every extra percentage point of spread roughly halves the 12-month recession odds.
- The 2022 to 2024 inversion drove the model's probability above 70 percent even though no NBER recession followed. That is consistent with a calibrated probabilistic forecast, not a model failure.
- Adding lagged unemployment gives the new variable an odds ratio above 1 per percentage point and shrinks the Treasury spread's odds ratio toward 1, consistent with the two series co-carrying the recession signal through related macro-cycle channels.
- `TimeSeriesSplit` cross-validated accuracy beats the 16 percent base-rate baseline for both specifications; the two-predictor model offers a modest improvement on the shorter post-1996 sample.

## Reproducing
1. Put your FRED API key in `../config.py` as `FRED_API_KEY = "..."` (gitignored).
2. `pip install -r requirements.txt`.
3. Run `lab_17_logistic_regression.ipynb` top to bottom.
4. For the dashboard: `streamlit run streamlit_app.py`.

## Stack
Python 3.13, pandas, numpy, scikit-learn, statsmodels, matplotlib, plotly, fredapi, streamlit.
