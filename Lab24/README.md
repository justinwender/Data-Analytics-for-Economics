# Lab 24 — Causal ML: Double Machine Learning and Causal Forests

## Objective
Diagnose and repair a broken manual Double Machine Learning (DML) cross-fitting pipeline, verify the fix on a known data-generating process, estimate the ATE of 401(k) eligibility on net financial assets using `DoubleML`, run sensitivity analysis for unmeasured confounders, and scale up to individual-level CATEs with `CausalForestDML`.

## Methodology
- Pulled the 9,915-household 401(k) panel via `doubleml.datasets.fetch_401K` with `net_tfa` as the outcome and `e401` as the treatment.
- Part A (diagnose): identified and fixed three planted bugs in the manual DML implementation — (1) the same fold used for nuisance training and residual computation, eliminating cross-fitting; (2) missing residualization of the treatment, leaving an open X → D path; (3) the wrong estimator for theta, using `np.mean(V_tilde * Y_tilde)` instead of the IV-style ratio `sum(D_tilde * Y_tilde) / sum(D_tilde * D)`.
- Part A (verify): on a simulated DGP with TRUE_ATE = 5.0 and 100 covariates, the corrected estimator recovers theta within 0.5 of the true value.
- Part B (package): fit `DoubleMLPLR` with 5-fold cross-fitting and `RandomForestRegressor(n_estimators=200, max_depth=5)` for both nuisance models; reported point estimate, standard error, 95 percent CI, and p-value.
- Part B (sensitivity): ran `sensitivity_analysis(cf_y=0.03, cf_d=0.03)` and interpreted the robustness value against the estimate and against the 95 percent CI crossing zero.
- Part C (extend): fit `CausalForestDML` with 500 trees, `min_samples_leaf=20`, `max_depth=10`, and `cv=5`. Extracted individual CATE predictions and 95 percent intervals, plotted the CATE histogram, and profiled the top-quartile high-response subgroup against the rest of the sample.
- Extension: compared Causal Forest CATE distribution to subgroup DML by income quartile using boxplots; quantified the within-quartile versus between-quartile spread to test whether the quartile-level ATE is representative of within-quartile heterogeneity.
- Packaged `manual_dml` (with sandwich standard errors) and `cate_by_subgroup` into `src/causal_ml.py` with type hints, docstrings, and a `__main__` smoke test on a 4,000-observation simulated DGP.

## Key Findings
- The corrected manual DML recovers the true ATE on the simulated DGP within +/- 0.3, matching the verification checkpoint. The broken version missed by more than 1.0 because of data leakage and the wrong closed-form.
- DoubleMLPLR estimates the 401(k) ATE on `net_tfa` in the range of $8,000 to $11,000 with a tight 95 percent CI that excludes zero.
- Sensitivity analysis at the `cf_y = cf_d = 0.03` benchmark leaves the estimate clearly above zero; the robustness value is above 0.03, meaning benchmarked omitted-variable bias is not enough to flip the sign.
- The Causal Forest CATE distribution is right-skewed with substantial within-quartile spread; the top quartile of treatment responders has meaningfully higher income, larger families, and higher rates of prior IRA ownership than the rest of the sample.
- Between-quartile CATE differences are comparable to within-quartile spread, which means coarse quartile-level subgroup DML would misrepresent the true heterogeneity. Targeting on continuous Causal Forest CATEs dominates targeting on quartile bins.

## Reproducing
1. `pip install -r requirements.txt`.
2. Run `lab-ch24-diagnostic.ipynb` top to bottom.

## Stack
Python 3.13, pandas, numpy, scikit-learn, doubleml, econml, matplotlib.
