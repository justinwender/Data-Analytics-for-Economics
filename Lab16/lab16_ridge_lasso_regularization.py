# Lab 16: Ridge and Lasso Regularization
# Macroeconomic Feature Selection via World Bank API

# %% [markdown]
# ## Step 1: Data Ingestion via API and The Scaling Prerequisite
# Why we are running this code: Instead of relying on static files, we will use the wbgapi library to query the
# World Bank's live databases. We will fetch our target variable (GDP Growth) alongside a matrix of
# macroeconomic indicators. After pivoting and cleaning the API response, we must mathematically map all
# variables to a uniform scale using StandardScaler(). This ensures the regularization penalties treat a
# percentage metric exactly the same as a metric measured in billions of dollars.

# %%
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, lasso_path

# 1. Define the Macroeconomic Indicators (The Features and the Target)
# NY.GDP.MKTP.KD.ZG = GDP growth (annual %) -> TARGET (y)
indicators = [
    'NY.GDP.MKTP.KD.ZG',    # GDP Growth
    'FP.CPI.TOTL.ZG',       # Inflation
    'BX.KLT.DINV.WD.GD.ZS', # Foreign Direct Investment
    'NE.TRD.GNFS.ZS',       # Trade (% of GDP)
    'GC.DOD.TOTL.GD.ZS',    # Government Debt
    'SL.UEM.TOTL.ZS'        # Unemployment Rate
    # Note: In the full lab, students will append 100+ indicator codes here.
]

print("Querying World Bank API...")
# 2. Fetch data for all countries for a recent stable year
df_api = wb.data.DataFrame(indicators, time=2022, skipBlanks=True, columns='series')

# 3. Data Cleaning Pipeline
# Drop economies with missing data to ensure a clean matrix for Scikit-Learn
df_clean = df_api.dropna()

# 4. Isolate Target (y) and Features (X)
y = df_clean['NY.GDP.MKTP.KD.ZG']
X_raw = df_clean.drop(columns=['NY.GDP.MKTP.KD.ZG'])

# CRITICAL PEDAGOGICAL STEP: Feature Scaling
# Unscaled data invalidates the mathematical geometry of the penalty.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"API Ingestion Complete. Feature matrix shape: {X_scaled.shape}")

# %% [markdown]
# ## Step 2: The Baseline Failure (Unconstrained OLS)
# Why we are running this code: To understand the cure, we must prove the disease. We will fit a standard,
# unconstrained LinearRegression model to our highly collinear data. In a high-dimensional space (high p, low
# n), OLS will heavily overfit, providing a deceptively perfect score that will instantly collapse in the real world.

# %%
# Step 2: The Baseline OLS (Demonstrating Overfitting in High Dimensions)
ols_model = LinearRegression().fit(X_scaled, y)

print("--- OLS REGRESSION RESULTS ---")
print(f"Standard OLS Training R-squared: {ols_model.score(X_scaled, y):.4f}")
print("Notice how the model effectively 'memorizes' the dataset.")

# %% [markdown]
# ## Step 3: Ridge Regression with Built-in Cross Validation
# Why we are running this code: We now apply the L2 Penalty (Ridge) to stabilize the ill-conditioned matrix.
# Ridge shrinks the coefficients to manage the variance caused by multicollinearity. We will mathematically
# verify that while Ridge shrinks coefficients, it rarely drives them to exactly zero.

# %%
# We pass a logarithmic space of alphas to test multiple orders of magnitude.
alphas_to_test = np.logspace(-3, 4, 100)
ridge_model = RidgeCV(alphas=alphas_to_test, cv=5).fit(X_scaled, y)

print("--- RIDGE REGRESSION RESULTS ---")
print(f"Optimal L2 Penalty (Alpha): {ridge_model.alpha_:.4f}")

# Verify the geometric reality of L2: no coefficients are EXACTLY zero.
zeros_in_ridge = np.sum(ridge_model.coef_ == 0)
print(f"Number of coefficients driven to exactly zero by Ridge: {zeros_in_ridge} out of {X_raw.shape[1]}")

# %% [markdown]
# ## Step 4: Lasso Regression and Automated Sparsity
# Why we are running this code: We transition to the L1 Penalty (Lasso). Because of its geometric shape, Lasso
# enforces "algorithmic sparsity"—it literally forces the coefficients of collinear or useless variables to
# absolute zero, keeping only the true structural drivers.

# %%
# LassoCV defaults to a path of 100 alphas. We specify cv=5.
# max_iter is increased to 10000 to prevent Coordinate Descent convergence failures.
lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_scaled, y)

print("--- LASSO REGRESSION RESULTS ---")
print(f"Optimal L1 Penalty (Alpha): {lasso_model.alpha_:.4f}")

# Calculate algorithmic sparsity
surviving_features = np.sum(lasso_model.coef_ != 0)
eliminated_features = np.sum(lasso_model.coef_ == 0)

print(f"\nFeatures Retained (Signal): {surviving_features}")
print(f"Features Eliminated (Noise): {eliminated_features}")

# Extracting the names of the surviving macroeconomic indicators
active_vars = X_raw.columns[lasso_model.coef_ != 0]
print("\nTop Surviving Macroeconomic Indicators:")
for var, coef in zip(active_vars, lasso_model.coef_[lasso_model.coef_ != 0]):
    print(f"{var}: {coef:.4f}")

# %% [markdown]
# ## Step 5: Visual Forensics — Plotting the Lasso Path
# Why we are running this code: As data scientists, we must visually explain to policymakers how the algorithm
# made its decisions. We will plot the trajectory of every macroeconomic coefficient as the L1 penalty grows
# stricter. The lines that survive the longest represent the most robust drivers of GDP growth.

# %%
# The lasso_path function computes coefficients along a full path of alphas.
alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y, alphas=alphas_to_test, max_iter=10000)

plt.figure(figsize=(12, 7))

# Plotting the coefficient paths against the negative log of alpha
for i in range(coefs_lasso.shape[0]):
    plt.plot(-np.log10(alphas_lasso), coefs_lasso[i, :], alpha=0.7)

# Draw a vertical line indicating the optimal alpha found via Cross-Validation
plt.axvline(x=-np.log10(lasso_model.alpha_), color='black', linestyle='--', label='Optimal Alpha (Cross-Validated)')

plt.title('The Lasso Path: Automated Feature Selection of World Bank Indicators', fontsize=14)
plt.xlabel('-Log(Alpha) [Increasing Penalty Strictness]', fontsize=12)
plt.ylabel('Standardized Beta Coefficients', fontsize=12)
plt.axhline(0, color='black', linewidth=1.5)
plt.legend()

# Maximize data-ink ratio: remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(alpha=0.2)
plt.show()
