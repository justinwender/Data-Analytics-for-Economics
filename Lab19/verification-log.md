# Verification Log — Lab 19

## P.R.I.M.E. prompt given to the AI Co-Pilot

```
[Prep]    Senior Python data scientist specialising in SHAP, scikit-learn, and
          dashboarding.
[Request] Build two artifacts for a graduate-level econometrics lab:
          (1) src/shap_utils.py with explain_prediction, global_importance,
              compare_importance.
          (2) A Streamlit dashboard letting the user adjust n_estimators and
              max_features, see SHAP beeswarm and waterfall update, compare
              RF vs Ridge vs GBR, and toggle MDI / permutation / SHAP rankings.
[Iterate] Use shap.TreeExplainer, sklearn, plotly, streamlit. No deprecated APIs.
          Match variable names used in the notebook (X_train, X_test, y_train,
          y_test, best_rf). Cache expensive training so the sliders are responsive.
[Mechanism] Inline comments on why TreeExplainer beats KernelExplainer for tree
            ensembles, and on how permutation importance differs from MDI.
[Evaluate] Run the module's __main__ block as a smoke test. Confirm dashboard
           metrics match the notebook's Test-R² table. Flag any SHAP ranking
           that diverges sharply from MDI so a reviewer can sanity check.
```

## What the AI generated
- `src/shap_utils.py` with `_build_shap_values` helper plus the three public functions, each with type hints and docstrings.
- `streamlit_app.py` with sidebar sliders (`n_estimators`, `max_features`, `max_display`), a KPI row of Test R² per model, a model-comparison bar chart, a switchable importance panel, a collapsible rank comparison, and a per-observation waterfall selector.

## What I changed
- Tightened the `compare_importance` output to return rank columns plus raw magnitudes rather than only raw magnitudes; ranks are easier to diff visually.
- Swapped `shap.initjs()` calls (not needed in headless Streamlit) for `show=False` + `bbox_inches="tight"` rendering through a BytesIO buffer.
- Added `st.cache_data`/`st.cache_resource` so retraining only fires when a slider actually moves.
- Dropped a default `max_features='sqrt'` option from the slider and forced integer values so the control maps 1-to-1 to feature count.
- Replaced a non-deterministic `n_jobs=-1` in the RF smoke test with `n_repeats=3` for the CI-friendly subset so the module imports quickly.

## What I verified
- `python src/shap_utils.py` runs its `__main__` block to completion and prints a ranking frame.
- Dashboard Test R² values at default sliders match the notebook's tuned-RF number to within floating-point noise.
- SHAP rankings from the Streamlit dashboard match the `compare_importance` output in the notebook.
- Waterfall plot for the highest-prediction row highlights MedInc and Latitude as the dominant positive contributors, which is consistent with both the beeswarm and the permutation-importance table.
