# Assignment 5 — The Sovereign Risk Engine

IMF Global Financial Stability Division exercise: diagnose overfitting in a brute-force OLS growth forecaster, rebuild it with Ridge and Lasso regularization, turn the same data into a binary crisis classifier with logistic regression, and make a capacity-constrained and cost-sensitive operating-point decision.

## What's in this folder
- `assignment5_sovereign_risk_engine.ipynb` — the full four-phase notebook.

## Methodology
- Shared data pipeline: thirty-seven WDI indicators for roughly 150 countries, averaged 2013 to 2019, with countries or indicators missing more than 40 percent of their panel dropped and the remainder median-imputed. Outcome: `gdp_growth_pc`. Binary label: `crisis = 1` if `gdp_growth_pc < 0`. 70/30 train/test split with `random_state=42`; features standardized on the training partition only.
- Phase 1 (regularization): fit OLS, `RidgeCV`, and `LassoCV` on the continuous growth target; reported training R², test R², test RMSE, and non-zero coefficient counts. Plotted the full Lasso coefficient path across log-λ and identified the first predictor to enter the model.
- Phase 2 (classification): deliberately fit the Linear Probability Model on the Lasso-selected features to surface out-of-bounds predicted probabilities, then fit `LogisticRegression` on the same features, reported odds ratios with the top-ranked predictor interpreted in plain English, and produced a side-by-side LPM versus logistic sigmoid visualization on the strongest standardized predictor.
- Phase 3 (operational evaluation): compared the logistic regression at τ = 0.5 to a naïve "no crisis" baseline, generated the confusion matrix and classification report, plotted ROC and Precision-Recall curves with AUC annotations, swept τ across [0.01, 0.99] to identify the capacity-constrained threshold (≤ 5 missions) and the F1-optimal threshold, and wrote a recommendation memo for the Division Chief.
- Phase 4 (P.R.I.M.E. AI expansion): constructed explicit Prep / Request / Iterate / Mechanism-check / Evaluate prompts for two tasks. Task 4.1 is a 200-resample bootstrap of LassoCV producing a selection-frequency bar chart with 50 percent and 80 percent reference lines. Task 4.2 is a threshold-sweep cost curve under a FN:FP cost ratio of 50 billion to 2 million, with the cost-minimising τ annotated and compared against the capacity-constrained and F1-optimal thresholds.

## Key findings
- OLS trained on all thirty-plus WDI indicators overfits catastrophically: training R² sits near its ceiling while test R² collapses, and the gap tracks the p/n ratio as the bias-variance framework predicts.
- Ridge and Lasso both buy large variance reductions for small bias increases. Ridge is the safer operational default for continuous growth forecasting; Lasso is preferable when the deliverable is a ranked shortlist of indicators for stakeholder briefings.
- On the binary crisis outcome the LPM produces predicted probabilities outside [0, 1] on real data, confirming that the failure is operational (uninterpretable output), not cosmetic. The logistic regression bounds predictions to [0, 1] and produces odds ratios that the IMF team can report to policymakers.
- The accuracy paradox is stark: a naïve "no crisis" baseline lands at roughly 0.75 accuracy by ignoring the minority class entirely, which is exactly the class the early-warning system exists to detect.
- Under a 5-mission capacity constraint the model catches roughly two of every three test-set crises at a precision that keeps mission resources focused. The F1-optimal threshold lifts recall further but exceeds the capacity; the cost-minimising threshold is lower still and recommends expanding capacity.
- Bootstrap Lasso stability confirms that only a small subset of indicators (typically inflation, government debt, and a CPIA governance score) is selected in ≥ 80 percent of resamples. Several Phase-1 Lasso selections are fragile under resampling, which means they are correlated alternatives, not independently important predictors.

## Reproducing
1. `pip install wbgapi scikit-learn statsmodels matplotlib seaborn numpy pandas`.
2. Run `assignment5_sovereign_risk_engine.ipynb` top to bottom. First run makes ~37 live World Bank API calls across the seven-year window.

## Stack
Python 3.13, pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, wbgapi.
