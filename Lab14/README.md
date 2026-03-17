# AI Capex Diagnostic Modeling: Detecting & Correcting Heteroscedasticity

## Objective

This lab moves beyond model estimation to **model validation**—diagnosing structural violations of OLS assumptions that systematically inflate precision estimates and distort inference. Using Nvidia's 2026 AI capital expenditure data, I diagnosed severe heteroscedasticity and multicollinearity in a baseline OLS model, then corrected inference using HC3 robust standard errors to recover the true statistical confidence in the deployment metrics.

## What I Did

Using **Nvidia's 2026 AI Capex and Deployment Diagnostics Dataset**, I operationalized the econometric principle that *assumptions violated are inferences corrupted*. Rather than assuming homoscedastic errors, I tested whether variance varied systematically with capital expenditure levels, and whether predictor multicollinearity inflated coefficient uncertainty.

### Key Findings

- **Heteroscedasticity Detected**: Residual variance expanded dramatically at high capex tiers, violating the homoscedasticity assumption
- **False Precision Problem**: Naive OLS underestimated standard errors by 15–40%, producing artificially low p-values and false statistical significance
- **HC3 Correction**: Robust standard errors appropriately widened confidence intervals, revealing true significance of deployment metrics
- **Multicollinearity Risk**: VIF analysis identified moderate collinearity in hardware utilization measures (VIF ≈ 4–6), below the critical threshold but worth monitoring

## Technical Approach

### Heteroscedasticity Diagnosis

- **Breusch-Pagan Test**: Formal statistical test to detect heteroscedasticity by regressing squared residuals on predictors
- **White Test**: Robust joint test that doesn't assume specific functional form of heteroscedasticity
- **Visual Forensics**: Residuals vs. Fitted Values scatterplot to inspect non-random error patterns across prediction ranges
- **Scale-Location Plot**: Square-root standardized residuals to visualize the expanding variance cone

### Multicollinearity Assessment

- **Variance Inflation Factor (VIF)**: Computed for all predictors to quantify how multicollinearity inflates standard errors
- **Interpretation Rule**: VIF < 5 (acceptable), 5–10 (moderate, monitor), > 10 (severe, problematic)
- **Correlation Matrix**: Examined pairwise correlations among hardware deployment metrics

### Robust Inference Correction

- **HC3 Heteroscedasticity-Consistent Estimator**: Applied sandwich estimator to recover correct standard errors without respecifying the model
- **Side-by-Side Comparison**: Displayed Naive OLS vs. HC3 Robust summaries to quantify the correction magnitude
- **Coefficient Stability**: Confirmed coefficients remained unchanged; only standard errors and p-values were corrected

### Implementation

- **Framework**: statsmodels' `OLS()`, `het_breuschpagan()`, `het_white()` for diagnostic tests
- **Robust Estimators**: `.get_robustcov_results(cov_type='HC3')` for sandwich estimation
- **Multicollinearity**: `variance_inflation_factor()` from statsmodels.stats.outliers_influence
- **Visualization**: Interactive Plotly dashboards for residual inspection and threshold-based VIF coloring

## Why This Matters: The Reliability Crisis in Empirical Economics

In applied econometrics, violating OLS assumptions is not a minor technical detail—it is a validity crisis. When heteroscedasticity goes uncorrected:

- **Inference Breaks**: Standard errors are biased. Confidence intervals are too narrow. Type I errors occur at rates far above α.
- **Publication Bias Masquerade**: False positives pass peer review because their p-values appear significant despite flawed estimation
- **Policy Errors**: Policymakers allocate resources based on "significant" effects that dissolve under rigorous scrutiny
- **Replication Failures**: Results don't hold when different data or corrected methods are applied

Robust standard errors are not a luxury—they are a *minimal requirement* for trustworthy inference. By diagnosing and correcting heteroscedasticity, we separate genuine insights from statistical artifacts. This is the discipline that separates evidence-based economics from empirical theater.

---

**Dataset**: Nvidia AI Capital Expenditure & Deployment Diagnostics (2026)
**Methods**: Breusch-Pagan Test, White Test, HC3 Robust Standard Errors, VIF Analysis
**Language**: Python 3
**Libraries**: pandas, statsmodels, numpy, plotly
