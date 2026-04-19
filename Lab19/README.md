# Lab 19 — Tree-Based Models: Random Forests on California Housing

## Objective
Compare Decision Tree, Ridge, Random Forest, and Gradient Boosting on the California Housing dataset; fix a train-versus-test evaluation bug; diagnose a causal overclaim from feature importance; and ship a SHAP-backed explainability layer plus a reusable Python module and an interactive Streamlit dashboard.

## Methodology
- Pulled the 20,640-row California Housing dataset from `sklearn.datasets` with an 80/20 train-test split and a fixed `random_state=42`.
- Diagnosed a planted bug where Random Forest was being scored on the training set and re-ran the comparison on held-out data.
- Critiqued a second deliberate flaw: a policy recommendation derived directly from MDI feature importance. Added a permutation-importance analysis on the test set and documented MDI's bias toward high-cardinality features.
- Tuned Random Forest with `GridSearchCV` over `n_estimators`, `max_depth`, and `max_features`, then benchmarked it alongside Ridge, default RF, and a Gradient Boosting Regressor on test RMSE and R².
- Added a SHAP layer using `TreeExplainer`: three per-observation waterfall plots (high prediction, low prediction, large residual), a beeswarm for global attribution, and a side-by-side ranking of MDI versus SHAP.
- Packaged the SHAP tooling as `src/shap_utils.py` with `explain_prediction`, `global_importance`, and `compare_importance`, each with type hints, docstrings, and a `__main__` smoke test.
- Built a Streamlit dashboard (`streamlit_app.py`) with sliders for `n_estimators` and `max_features`, switchable MDI / permutation / SHAP rankings, KPI cards for each model, and a per-observation waterfall selector.

## Key Findings
- Fixed Random Forest Test R² lands around 0.80, roughly 20 points above Ridge and 5 to 8 points above a single depth-unlimited tree. The original 0.97 number was pure in-sample memorisation.
- MDI overweights continuous location features (Latitude, Longitude) relative to permutation importance and SHAP. All three metrics agree that MedInc is the dominant predictor, but they disagree on the next tier.
- Tuned RF shaves roughly 0.02 to 0.03 off default RF's test RMSE, while Gradient Boosting (depth 5, 200 trees, lr 0.1) is the overall winner on this dataset.
- SHAP confirms that MedInc, Latitude, and AveRooms drive most of the variation in predictions, with strong interactions between income and room count.
- None of the importance metrics carry causal content. The lab's Part 2 critique makes the distinction between prediction and policy explicit.

## Reproducing
1. `pip install -r requirements.txt`.
2. Run `lab-ch19-diagnostic.ipynb` top to bottom.
3. For the interactive explorer, from the `Lab19/` folder run `streamlit run streamlit_app.py`.

## Stack
Python 3.13, scikit-learn, pandas, numpy, matplotlib, shap, plotly, streamlit.
