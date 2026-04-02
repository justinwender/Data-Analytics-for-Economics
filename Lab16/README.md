# High-Dimensional GDP Growth Forecasting with Regularized Regression

## Objective

Forecast cross-country GDP per capita growth using 50+ World Development Indicators to demonstrate the failure of OLS in high-dimensional settings and evaluate Ridge and Lasso regularization as principled alternatives with cross-validated penalty selection.

## Methodology

- Downloaded 35+ WDI indicators via the `wbgapi` Python API spanning trade, macroeconomics, education, infrastructure, health, finance, natural resources, agriculture, and governance for 120+ countries over the 2013-2019 pre-COVID period.
- Averaged indicators across the time dimension to produce a single cross-sectional observation per country, dropped countries and indicators with excessive missingness (>40%), and imputed remaining gaps with cross-country medians.
- Split countries 70/30 into training and test sets and standardized all features to zero mean and unit variance using `StandardScaler` (fit on training data only) to ensure valid penalization.
- Fit an OLS baseline via `LinearRegression` to establish the overfitting benchmark, then applied `RidgeCV` (L2 penalty) and `LassoCV` (L1 penalty) with 5-fold cross-validation over a log-spaced grid of lambda values.
- Visualized the full Lasso coefficient path using `lasso_path` to trace how predictors enter the model as the penalty decreases, and built a side-by-side model comparison table reporting training R-squared, test R-squared, test MSE, and predictor counts.

## Key Findings

- OLS severely overfitted the training data, producing a large training R-squared but a low or negative test R-squared, confirming the expected failure mode when the predictor-to-observation ratio is high.
- Both Ridge and Lasso substantially improved out-of-sample performance by introducing bias in exchange for reduced variance.
- Lasso achieved test R-squared comparable to Ridge while retaining only a sparse subset of predictors, demonstrating that the majority of WDI indicators are conditionally redundant for forecasting growth once a few key variables are included.
- The Lasso path revealed which predictors enter the model first as the penalty relaxes, identifying the single strongest conditional contributors to cross-country growth variation.
- A zero Lasso coefficient reflects predictive redundancy given the correlation structure of the data, not economic irrelevance or the absence of a causal relationship.
