# Lab: The Bias-Variance Tradeoff
# Understanding Underfitting and Overfitting via NVIDIA Revenue Forecasting

# %% [markdown] 
# Step 1: Data Ingestion and Visual Exploratory Data Analysis (EDA)
# Establish your analytical environment by importing all required mathematical and machine learning libraries.
# Load the NVIDIA dataset and create a high data-ink ratio scatter plot to visually observe the non-linear,
# exponential curvature of the revenue growth.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Step 1: Ingestion of Modernized NVIDIA Dataset
# Data reflects the massive AI infrastructure capex boom of 2024-2026

data = {
    'Time_Index': np.array([1, 2, 3, 4, 5, 6, 7, 8]),
    'Total_Revenue_Billions': np.array([26.04, 30.04, 35.10, 39.33, 44.06, 46.74, 57.00, 68.10])
}
df = pd.DataFrame(data)
X = df[['Time_Index']]
y = df['Total_Revenue_Billions']

# Visual EDA (Adhering to Data-Ink Ratio principles)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Actual Revenue')
plt.title('NVIDIA Total Revenue (FY25-FY26)', fontsize=14)
plt.xlabel('Sequential Quarter Index', fontsize=12)
plt.ylabel('Revenue (Billions USD)', fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# %% [markdown]
# Step 2: The Baseline High-Bias Model (Underfitting)
# Instantiate a standard LinearRegression model and fit it to predict revenue strictly using the one-dimensional Time Index.
# Calculate the training Mean Squared Error (MSE) and plot the regression line to observe how a rigid, straight line
# systematically underfits the exponential reality.
# %%

# Step 2: High Bias (Underfitting) Linear Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)

print(f"Linear Training MSE (High Bias): {mse_linear:.2f}")

plt.plot(X, y_pred_linear, color='blue', linestyle='--', label='Linear Fit (High Bias)')

# %% [markdown]
# Step 3: The High-Variance Model (Overfitting via Polynomial Expansion)
# Cure the bias by dramatically increasing complexity. Mathematically expand your single feature into a 7-dimensional
# feature matrix using PolynomialFeatures. Fit a new OLS regression to this space. Notice how the training MSE approaches
# zero as the polynomial curve snakingly contorts to memorize historical noise.

# %%
# Step 3: High Variance (Overfitting) Polynomial Model
poly_features = PolynomialFeatures(degree=7, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"Polynomial Degree-7 Training MSE (High Variance): {mse_poly:.2f}")

# Plotting the smooth polynomial curve
X_smooth = np.linspace(1, 8.5, 100).reshape(-1, 1)
X_smooth_poly = poly_features.transform(X_smooth)
y_smooth_poly = poly_reg.predict(X_smooth_poly)

plt.plot(X_smooth, y_smooth_poly, color='red', label='Polynomial Fit (High Variance)')
plt.legend()
plt.show()

# %% [markdown]
# Step 4: The Epistemological Collapse (Extrapolation)
# Create a synthetic prediction array for the immediate, unseen next quarter (Index 9). Pass this index through your
# trained 7th-degree polynomial model to force a forecast, witnessing the algorithm hallucinate based on its terminal trajectory.

# %%
# Step 4: The Extrapolation Collapse
X_future = np.array([[9]])  # Forecasting Q1 FY27
X_future_poly = poly_features.transform(X_future)
future_pred = poly_reg.predict(X_future_poly)

print(f"\nHallucinated Q1 FY27 Revenue Prediction: ${future_pred[0]:.2f} Billion")

# %% [markdown] 
# Step 5: Rigorous Evaluation via K-Fold Cross-Validation
# Abandon the illusion of the training MSE. Deploy K-Fold Cross-Validation to sequentially rotate holdout sets,
# rigorously testing the model on data it hasn't seen during training. Compare the massive cross-validated error to
# your artificially low training error to quantify the true variance gap.

# %%
# Step 5: K-Fold Cross Validation
cv_scores = cross_val_score(poly_reg, X_poly, y, cv=4, scoring='neg_mean_squared_error')
mean_cv_mse = -cv_scores.mean()

print(f"K-Fold Cross-Validated MSE (The True Operational Error): {mean_cv_mse:.2f}")

# %% [markdown]
# Step 6: Ridge Regression (L2 Regularization) — Constraining Variance
# The overfitted degree-7 polynomial has near-zero training MSE but catastrophic cross-validated error,
# because its coefficients are astronomically large and tuned to noise. Ridge Regression fixes this by
# adding a penalty term — lambda * sum(beta^2) — to the OLS loss function. This forces the optimizer
# to shrink coefficient magnitudes, trading a small increase in bias for a large reduction in variance.
# RidgeCV runs 4-Fold CV internally to select the optimal alpha (the lambda penalty strength).

# %%
from sklearn.linear_model import RidgeCV

# Step 6: Ridge Regression with 4-Fold Cross-Validation Alpha Selection

# Candidate alpha values: small alpha = weak regularization, large alpha = heavy shrinkage.
# RidgeCV scores each alpha via 4-fold CV and retains the one minimizing held-out error.
alphas = np.logspace(-3, 5, 200)

# The polynomial feature matrix from Step 3 is reused — Ridge operates in the same
# expanded 7-dimensional space but penalizes large coefficients instead of ignoring them.
ridge_cv = RidgeCV(alphas=alphas, cv=4, scoring='neg_mean_squared_error')
ridge_cv.fit(X_poly, y)

# Optimal alpha selected by RidgeCV — this is the penalty strength that best balanced
# bias and variance across the 4 cross-validation folds.
optimal_alpha = ridge_cv.alpha_
print(f"Optimal Ridge Alpha (L2 Penalty): {optimal_alpha:.4f}")

# Training MSE for Ridge — will be higher than the overfit OLS (near-zero) training MSE,
# because Ridge intentionally accepts more in-sample error to reduce out-of-sample variance.
y_pred_ridge = ridge_cv.predict(X_poly)
mse_ridge_train = mean_squared_error(y, y_pred_ridge)
print(f"Ridge Training MSE: {mse_ridge_train:.2f}")

# Cross-validated MSE for Ridge — the true apples-to-apples comparison with Step 5.
# A lower CV MSE than the unregularized polynomial means Ridge generalizes better.
ridge_cv_scores = cross_val_score(ridge_cv, X_poly, y, cv=4, scoring='neg_mean_squared_error')
mse_ridge_cv = -ridge_cv_scores.mean()
print(f"Ridge 4-Fold CV MSE: {mse_ridge_cv:.2f}")

print(f"\n--- Variance Reduction Summary ---")
print(f"Unregularized Poly CV MSE : {mean_cv_mse:.2f}  (chaotic, high variance)")
print(f"Ridge Regularized CV MSE  : {mse_ridge_cv:.2f}  (constrained, lower variance)")
print(f"Variance Reduction        : {mean_cv_mse - mse_ridge_cv:.2f}")

# Smooth curves for plotting — same x-grid as Step 3
X_smooth_poly = poly_features.transform(X_smooth)
y_smooth_ridge = ridge_cv.predict(X_smooth_poly)
y_smooth_poly_replot = poly_reg.predict(X_smooth_poly)  # OLS poly for comparison

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', zorder=5, label='Actual Revenue')

# Overfitted OLS polynomial — large coefficients cause wild oscillation outside data range
plt.plot(X_smooth, y_smooth_poly_replot, color='red', linestyle='--', label='OLS Poly Degree-7 (High Variance)')

# Ridge curve — L2 penalty shrinks coefficients, producing a smoother, more stable trajectory
plt.plot(X_smooth, y_smooth_ridge, color='green', linewidth=2, label=f'Ridge Regularized (alpha={optimal_alpha:.2f})')

plt.title('Ridge vs. OLS Polynomial: Variance Suppression via L2 Regularization', fontsize=13)
plt.xlabel('Sequential Quarter Index', fontsize=12)
plt.ylabel('Revenue (Billions USD)', fontsize=12)
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# --- Evaluation: How to Interpret the MSE Comparison ---
#
# The unregularized degree-7 polynomial achieves near-zero TRAINING MSE because it memorizes
# every data point — its coefficients grow enormous to fit each observation exactly. But the
# 4-Fold CV MSE from Step 5 exposes the real cost: when tested on unseen folds, that
# memorization collapses into massive prediction error.
#
# Ridge forces a different objective. Instead of minimizing only sum((y - y_hat)^2), it minimizes
# sum((y - y_hat)^2) + alpha * sum(beta^2). The alpha * sum(beta^2) term directly penalizes
# large coefficient values. As alpha increases, the optimizer is increasingly compelled to keep
# coefficients small and smooth — shrinking the model's ability to contort around noise.
#
# What to look for:
#   - Ridge TRAINING MSE will be higher than OLS poly training MSE. This is expected and correct:
#     Ridge is deliberately accepting more bias to buy variance reduction.
#   - Ridge CV MSE should be substantially lower than OLS poly CV MSE. That gap is the payoff —
#     it represents real improvement in predictive stability on unseen data.
#   - The optimal alpha printed above tells you how aggressively the data required regularization.
#     A large alpha means the unregularized coefficients were extreme; the penalty had to work hard.
#
# The core tradeoff: a model that is slightly wrong everywhere (Ridge) is more useful in practice
# than a model that is perfect on the training set and catastrophically wrong everywhere else (OLS poly).
