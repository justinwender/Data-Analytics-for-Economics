#%% [markdown]
### Step 1: Environment Setup
# Install all required packages and import the necessary libraries. We use wbgapi to pull World Bank data directly, scikit-learn for regression models, and matplotlib for visualization. Setting a random seed ensures reproducibility.

#%%
# ============================================================
# SETUP -- Run this cell first
# ============================================================

# Uncomment and run once if packages are missing:
# !pip install wbgapi scikit-learn matplotlib seaborn numpy pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import wbgapi as wb

# Reproducibility
np.random.seed(42)

print("Setup complete ✓")
# %% [markdown]
### Step 2: Download World Bank Data
# We define a dictionary of 35+ WDI indicator codes spanning trade, macroeconomics, education, infrastructure, health, finance, natural resources, agriculture, and governance. These are the same indicators IMF and World Bank economists use in growth diagnostics. We download 7 years of data (2013–2019, pre-COVID) via the WDI API.

# %%
# ============================================================
# PART 1A: Download World Bank Data
# ============================================================

# WDI indicator codes for our predictors
INDICATORS = {
    # Growth outcome (our y)
    'NY.GDP.PCAP.KD.ZG': 'gdp_growth_pc',

    # Trade & Openness
    'NE.TRD.GNFS.ZS':    'trade_pct_gdp',
    'BX.KLT.DINV.WD.GD.ZS': 'fdi_inflows_pct_gdp',
    'TM.TAX.MRCH.SM.AR.ZS': 'tariff_rate_avg',
    'BX.GSR.ROYL.CD':    'royalties_receipts',

    # Macroeconomics
    'FP.CPI.TOTL.ZG':    'inflation_cpi',
    'GC.DOD.TOTL.GD.ZS': 'govt_debt_pct_gdp',
    'GC.XPN.TOTL.GD.ZS': 'govt_expenditure_pct_gdp',
    'BN.CAB.XOKA.GD.ZS': 'current_account_pct_gdp',
    'FR.INR.RINR':       'real_interest_rate',
    'PA.NUS.FCRF':       'exchange_rate_official',

    # Education & Human Capital
    'SE.SEC.ENRR':       'secondary_enrollment_gross',
    'SE.TER.ENRR':       'tertiary_enrollment_gross',
    'SE.ADT.LITR.ZS':    'adult_literacy_rate',
    'SE.XPD.TOTL.GD.ZS': 'education_expenditure_pct_gdp',
    'SL.UEM.TOTL.ZS':    'unemployment_rate',

    # Infrastructure & Technology
    'IT.NET.USER.ZS':    'internet_users_pct',
    'IT.CEL.SETS.P2':    'mobile_subscriptions_per100',
    'EG.ELC.ACCS.ZS':    'electricity_access_pct',
    'IS.ROD.PAVE.ZS':    'paved_roads_pct',

    # Health & Demographics
    'SP.DYN.LE00.IN':    'life_expectancy',
    'SH.DYN.MORT':       'infant_mortality_per1000',
    'SP.POP.GROW':       'population_growth',
    'SP.URB.TOTL.IN.ZS': 'urbanization_pct',
    'SH.XPD.CHEX.GD.ZS': 'health_expenditure_pct_gdp',

    # Finance & Banking
    'FS.AST.DOMS.GD.ZS': 'domestic_credit_pct_gdp',
    'CM.MKT.LCAP.GD.ZS': 'market_cap_pct_gdp',
    'FB.ATM.TOTL.P5':    'atms_per100k',
    'FD.AST.PRVT.GD.ZS': 'private_credit_pct_gdp',

    # Natural Resources
    'NY.GDP.TOTL.RT.ZS': 'natural_resource_rents_pct_gdp',
    'EG.FEC.RNEW.ZS':    'renewable_energy_pct',
    'EN.ATM.CO2E.PC':    'co2_emissions_per_capita',

    # Agriculture
    'NV.AGR.TOTL.ZS':    'agriculture_pct_gdp',
    'AG.LND.ARBL.ZS':    'arable_land_pct',

    # Governance (World Bank Governance Indicators)
    'IQ.CPA.TRAD.XQ':    'trade_cpia',
    'IQ.CPA.FINS.XQ':    'financial_management_cpia',
    'IQ.CPA.PROP.XQ':    'property_rights_cpia',
}

OUTCOME_VAR = 'gdp_growth_pc'
indicator_list = list(INDICATORS.keys())

print(f"Downloading {len(indicator_list)} indicators for all countries, 2013–2019...")
print("(This may take 30–60 seconds -- API call to World Bank)")

try:
    raw_data = wb.data.DataFrame(
        indicator_list,
        time=range(2013, 2020),  # 2013–2019
        skipBlanks=True,
        labels=False
    )
    raw_data.columns = [INDICATORS[c] if c in INDICATORS else c for c in raw_data.columns]
    print(f"Raw data shape: {raw_data.shape}")
    print("Download successful ✓")
except Exception as e:
    print(f"API error: {e}")
    print("Loading fallback data from CSV...")
# %% [markdown]
### Step 3: Build the Analysis Dataset
# We average all indicators across the 2013–2019 period for each country, producing a single cross-sectional observation per country. Countries with more than 40% missing values are dropped. Remaining gaps are filled with the cross-country median -- a standard approach in cross-country empirics that avoids the selection bias of listwise deletion.

# %%
# ============================================================
# PART 1B: Build the Analysis Dataset
# ============================================================

# Average over time dimension
if isinstance(raw_data.index, pd.MultiIndex):
    averaged_by_series = raw_data.mean(axis=1)
    country_data = averaged_by_series.unstack(level='series')
    country_data = country_data.rename(columns=INDICATORS)
else:
    country_data = raw_data.copy()

# Drop countries with too many missing values (keep countries with >= 60% non-missing)
threshold = 0.60
country_data = country_data.dropna(thresh=int(threshold * country_data.shape[1]))

# Drop indicators with too many missing values across countries
country_data = country_data.dropna(axis=1, thresh=int(threshold * len(country_data)))

# Final fill: impute remaining missing values with the cross-country median
country_data = country_data.fillna(country_data.median())

print(f"Final dataset: {len(country_data)} countries × {country_data.shape[1]} indicators")
print(f"\nSample countries: {list(country_data.index[:5])}")
print(f"\nIndicators retained: {list(country_data.columns)}")
print(f"\nGDP growth summary:")
print(country_data[OUTCOME_VAR].describe().round(2))
# %% [markdown]
### Step 4: Train-Test Split and OLS Baseline
# We split countries (not time periods) into training and test sets -- a 70/30 split. This simulates the realistic challenge an IMF forecaster faces: can the model generalize to countries it has never seen? Features are standardized to zero mean and unit variance, which is critical for Ridge and Lasso since these methods penalize coefficient magnitude.

# %%
# ============================================================
# PART 1C: Train-Test Split & OLS Baseline
# ============================================================

# Separate outcome (y) from predictors (X)
feature_cols = [c for c in country_data.columns if c != OUTCOME_VAR]

X = country_data[feature_cols].values
y = country_data[OUTCOME_VAR].values
feature_names = feature_cols

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"Training countries: {X_train.shape[0]}")
print(f"Test countries: {X_test.shape[0]}")
print(f"Number of predictors: {X_train.shape[1]}")
print(f"Predictor-to-observation ratio (train): p/n = {X_train.shape[1]}/{X_train.shape[0]} = {X_train.shape[1]/X_train.shape[0]:.2f}")
print()
print("If p/n > 0.5, OLS is at serious risk of overfitting.")

# Standardize features (critical for Ridge and Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only!
X_test_scaled  = scaler.transform(X_test)         # apply same transform to test

print("\nFeatures standardized (zero mean, unit variance) ✓")
# %% [markdown]
### Step 5: OLS Baseline -- Demonstrating the Failure Mode
# OLS will overfit when p/n is large. We predict this before looking at results: expect high training R² but much lower (possibly negative) test R². OLS assigns non-zero coefficients to every single predictor -- even variables unlikely to drive growth -- because it has no mechanism to say "this variable adds noise, not signal."

# %%
# ============================================================
# PART 1D: OLS Baseline -- Demonstrating the Failure Mode
# ============================================================

ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

# Training performance
y_train_pred_ols = ols_model.predict(X_train_scaled)
ols_train_r2  = r2_score(y_train, y_train_pred_ols)
ols_train_mse = mean_squared_error(y_train, y_train_pred_ols)

# Test performance
y_test_pred_ols = ols_model.predict(X_test_scaled)
ols_test_r2  = r2_score(y_test, y_test_pred_ols)
ols_test_mse = mean_squared_error(y_test, y_test_pred_ols)

print("=" * 45)
print("OLS BASELINE RESULTS")
print("=" * 45)
print(f"Training R²:  {ols_train_r2:.3f}")
print(f"Test R²:      {ols_test_r2:.3f}")
print(f"")
print(f"Training MSE: {ols_train_mse:.3f}")
print(f"Test MSE:     {ols_test_mse:.3f}")
print(f"")
print(f"Gap (Train R² - Test R²): {ols_train_r2 - ols_test_r2:.3f}")
print("=" * 45)
print()
print("Interpretation:")
print(f"  OLS fits the training data {ols_train_r2:.0%} -- but only explains")
print(f"  {max(ols_test_r2, 0):.0%} of test variance. This is overfitting.")
# %% [markdown]
### Step 6: Ridge Regression with Cross-Validated Lambda
# Ridge adds a squared-magnitude penalty (λ||β||²) to the OLS objective, shrinking all coefficients toward zero but keeping every predictor in the model. RidgeCV tries a grid of λ values and selects the one that minimizes cross-validation error. Note that scikit-learn calls λ "alpha" for historical reasons.

# %%
# ============================================================
# PART 2A: Ridge Regression with Cross-Validated Lambda
# ============================================================

# Grid of lambda (alpha) values to try -- log-spaced from 0.01 to 1000
lambda_grid = np.logspace(-2, 3, 50)

# TODO: Create a RidgeCV with 5-fold CV and the lambda_grid above
ridge_cv = RidgeCV(alphas=lambda_grid, cv=5)  # ← complete this line

# TODO: Fit ridge_cv on the scaled training data
ridge_cv.fit(X_train_scaled, y_train)  # ← complete this line

# Evaluate on test set
y_test_pred_ridge = ridge_cv.predict(X_test_scaled)
ridge_test_r2  = r2_score(y_test, y_test_pred_ridge)
ridge_test_mse = mean_squared_error(y_test, y_test_pred_ridge)

print("=" * 45)
print("RIDGE REGRESSION RESULTS")
print("=" * 45)
print(f"Optimal λ* (CV-selected): {ridge_cv.alpha_:.4f}")
print(f"Non-zero coefficients:    {np.sum(ridge_cv.coef_ != 0)} of {X_train.shape[1]}")
print(f"Test R²:                  {ridge_test_r2:.3f}")
print(f"Test MSE:                 {ridge_test_mse:.3f}")
print()
print(f"vs. OLS: Test R² = {ols_test_r2:.3f}, Test MSE = {ols_test_mse:.3f}")
# %% [markdown]
### Step 7: LassoCV -- Automated Feature Selection
# Lasso adds an absolute-value penalty (λ||β||₁) that drives many coefficients to exactly zero, performing automatic feature selection. Complete the fit_lasso_cv() function below, then examine which predictors survive and which are eliminated.

# %%
# ============================================================
# PART 2B: LassoCV -- Automated Feature Selection
# ============================================================

def fit_lasso_cv(X_train, y_train, X_test, y_test, cv=5):
    """
    Fit LassoCV to select optimal regularization parameter
    and evaluate on test set.

    Parameters
    ----------
    X_train : np.ndarray -- Standardized training features
    y_train : np.ndarray -- Training outcome (GDP growth)
    X_test  : np.ndarray -- Standardized test features
    y_test  : np.ndarray -- Test outcome
    cv      : int -- Number of cross-validation folds

    Returns
    -------
    lasso_model : LassoCV -- Fitted model with optimal alpha
    test_r2     : float   -- R² on held-out test set
    test_mse    : float   -- MSE on held-out test set
    """
    # TODO: Create a LassoCV with cv folds and max_iter=10_000
    lasso_model = LassoCV(cv=cv, max_iter=10_000, random_state=42)

    # TODO: Fit the model on training data
    lasso_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = lasso_model.predict(X_test)

    # TODO: Compute test R² and test MSE
    test_r2  = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    return lasso_model, test_r2, test_mse


# Call the function
lasso_cv_model, lasso_test_r2, lasso_test_mse = fit_lasso_cv(
    X_train_scaled, y_train, X_test_scaled, y_test, cv=5
)

n_nonzero = np.sum(lasso_cv_model.coef_ != 0)

print("=" * 45)
print("LASSO REGRESSION RESULTS")
print("=" * 45)
print(f"Optimal λ* (CV-selected): {lasso_cv_model.alpha_:.4f}")
print(f"Non-zero coefficients:    {n_nonzero} of {X_train.shape[1]}")
print(f"Test R²:                  {lasso_test_r2:.3f}")
print(f"Test MSE:                 {lasso_test_mse:.3f}")
print()
print("Selected predictors (non-zero Lasso coefficients):")
selected_features = [
    (feature_names[i], lasso_cv_model.coef_[i])
    for i in range(len(feature_names))
    if lasso_cv_model.coef_[i] != 0
]
for name, coef in sorted(selected_features, key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:<40} coef = {coef:+.4f}")
# %% [markdown]
### Step 8: The Lasso Path -- Who Enters First?
# The Lasso Path traces all coefficient estimates as λ varies from large (everything zero) to small (approaching OLS). The first variable to enter -- the one whose coefficient leaves zero at the highest penalty -- is the single strongest predictor of GDP growth. The vertical dashed line marks our CV-selected λ*. Colored lines are the selected predictors; gray lines are those that Lasso zeroed out.

# %%
# ============================================================
# PART 2C: The Lasso Path -- Who Enters First?
# ============================================================

alphas_path, coefs_path, _ = lasso_path(
    X_train_scaled, y_train,
    eps=1e-3,
    n_alphas=100,
)

optimal_alpha = lasso_cv_model.alpha_

fig, ax = plt.subplots(figsize=(12, 7))

active_features_idx = np.where(lasso_cv_model.coef_ != 0)[0]

for i in range(len(feature_names)):
    if i in active_features_idx:
        ax.plot(np.log(alphas_path), coefs_path[i],
                linewidth=2, label=feature_names[i])
    else:
        ax.plot(np.log(alphas_path), coefs_path[i],
                linewidth=0.8, color='lightgray', alpha=0.6)

ax.axvline(np.log(optimal_alpha), color='red', linestyle='--', linewidth=2,
           label=f'CV-selected λ* = {optimal_alpha:.4f}')
ax.axhline(0, color='black', linewidth=0.5)

ax.set_xlabel('log(λ)  ←  Larger penalty (sparser)  |  Smaller penalty (denser)  →',
               fontsize=12)
ax.set_ylabel('Coefficient value (standardized units)', fontsize=12)
ax.set_title(
    'Lasso Path: GDP Growth Predictors as λ Varies\n'
    'Gray lines = zeroed-out predictors | Colored lines = selected at λ*',
    fontsize=13
)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_path_gdp_growth.png', dpi=150, bbox_inches='tight')
plt.show()
print("Lasso Path saved to lasso_path_gdp_growth.png")
# %% [markdown]
### Step 9: Model Comparison Table
# Build a side-by-side comparison of OLS, Ridge, and Lasso. Pay attention to the training R² vs. test R² gap for each method, the number of non-zero predictors, and the test MSE. This table is the core deliverable that demonstrates the bias-variance tradeoff in action.

# %%
# ============================================================
# PART 2D: Model Comparison Table
# ============================================================

comparison = pd.DataFrame({
    'Method': ['OLS', 'Ridge (RidgeCV)', 'Lasso (LassoCV)'],
    'Lambda (α*)': [
        'N/A (no penalty)',
        f'{ridge_cv.alpha_:.4f}',
        f'{lasso_cv_model.alpha_:.4f}'
    ],
    'Non-zero Predictors': [
        X_train.shape[1],
        X_train.shape[1],
        np.sum(lasso_cv_model.coef_ != 0)
    ],
    'Training R²': [
        f'{r2_score(y_train, ols_model.predict(X_train_scaled)):.3f}',
        f'{r2_score(y_train, ridge_cv.predict(X_train_scaled)):.3f}',
        f'{r2_score(y_train, lasso_cv_model.predict(X_train_scaled)):.3f}'
    ],
    'Test R²': [
        f'{ols_test_r2:.3f}',
        f'{ridge_test_r2:.3f}',
        f'{lasso_test_r2:.3f}'
    ],
    'Test MSE': [
        f'{ols_test_mse:.3f}',
        f'{ridge_test_mse:.3f}',
        f'{lasso_test_mse:.3f}'
    ]
})

print(comparison.to_string(index=False))
print()
print("Key observations:")
print(f"  • OLS training R² >> test R²: evidence of overfitting (high variance)")
print(f"  • Ridge and Lasso reduce the train-test gap")
print(f"  • Lasso selects only {np.sum(lasso_cv_model.coef_ != 0)} of {X_train.shape[1]} predictors")
print(f"  • The other {X_train.shape[1] - np.sum(lasso_cv_model.coef_ != 0)} are predictively redundant,")
print(f"    not necessarily economically unimportant")
# %% [markdown]
### Step 10: Open-Ended Interpretation (Written Responses)

# %% [markdown]
### Question 1 -- Interpreting Lasso Zeros
# Look at the predictors that Lasso zeroed out in Step 7. Suppose a World Bank colleague says: "Your Lasso model proves that paved_roads_pct is economically irrelevant to GDP growth -- we should remove it from all future analysis."
# 
#   The colleague is wrong because a Lasso coefficient of zero indicates conditional predictive redundancy, not economic irrelevance. Lasso selects among correlated predictors and retains only one representative from each
#    cluster; if paved_roads_pct is highly correlated with other development proxies like electricity_access_pct or internet_users_pct, Lasso will zero it out because the information it carries is already captured by the 
#   retained variable. This reflects the correlation structure of the predictors, not a causal relationship (or lack thereof) between roads and growth. Lasso is a prediction tool, not a causal identification strategy, so
#   a zero coefficient tells us the variable adds no marginal predictive power given the other variables in the model, not that it has no effect on GDP growth.  
# Write 2–3 sentences explaining why your colleague is wrong. Your answer should use the terms conditional predictive redundancy, correlation structure, and causal relationship (or their equivalents).
# %% [markdown]
### Question 2 -- Ridge vs. Lasso: When to Use Which?
# Your dataset has ~30–50 predictors and ~80–100 countries in training data. Many WDI indicators are correlated (e.g., internet users and electricity access are both proxies for development level).
# 
# For this dataset, Lasso is preferable because the many correlated WDI indicators (e.g., internet users, electricity access, mobile subscriptions all proxying development level) mean most predictors are redundant, and 
#   Lasso handles this by zeroing out the redundant ones rather than keeping all of them shrunken like Ridge. The model comparison table confirms this: Lasso achieves comparable test R² and MSE to Ridge while using far 
#   fewer predictors, delivering both a better bias-variance tradeoff and a more interpretable model for policymakers. 
# Should you prefer Ridge or Lasso for this specific dataset? Justify your answer using (a) the structure of the data (correlated predictors, p/n ratio), and (b) your model comparison table results.

# %% [markdown]
### Step 11: Extension -- Change the Outcome Variable (Optional)
# Instead of GDP growth, predict a different development outcome such as infant_mortality_per1000, urbanization_pct, or secondary_enrollment_gross. Does Lasso select different predictors? Does the model fit better or worse? What does this tell you about which indicators are "general" vs. "specific" to particular development outcomes?

# %% [markdown]
### Step 12: Interactive Plotly Dashboard
# This section builds a two-panel interactive dashboard using Plotly:
#   Panel 1 (left): Lasso Path with a draggable lambda slider that highlights which
#       predictors are active at any chosen penalty level and reports the resulting R².
#   Panel 2 (right): Grouped bar chart comparing the absolute coefficient magnitudes
#       of OLS, Ridge, and Lasso side by side so you can see how regularization
#       reshapes the model's "attention" across predictors.

# %%
# ============================================================
# PART 3: Interactive Plotly Dashboard
# ============================================================

# Uncomment and run once if plotly is missing:
# !pip install plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# 3A: Recompute the Lasso path data we need for the interactive plot
# ------------------------------------------------------------------
# alphas_path and coefs_path were already computed in Step 8.
# coefs_path shape: (n_features, n_alphas)
# alphas_path shape: (n_alphas,) -- sorted from largest to smallest lambda

log_alphas = np.log(alphas_path)  # x-axis values for the path plot

# ------------------------------------------------------------------
# 3B: For each lambda on the path, fit a Lasso at that exact alpha
#     and compute the test R² so the slider can display it.
# ------------------------------------------------------------------
# We evaluate R² on the TEST set at each alpha along the path.
# This lets the user see how out-of-sample performance changes with lambda.

r2_at_each_alpha = []
for a in alphas_path:
    # Fit a Lasso at this specific alpha
    temp_lasso = Lasso(alpha=a, max_iter=10_000)
    temp_lasso.fit(X_train_scaled, y_train)
    r2_val = r2_score(y_test, temp_lasso.predict(X_test_scaled))
    r2_at_each_alpha.append(r2_val)
r2_at_each_alpha = np.array(r2_at_each_alpha)

# ------------------------------------------------------------------
# 3C: Build Panel 1 -- Lasso Path with lambda slider
# ------------------------------------------------------------------
# We create one Plotly "frame" per alpha value. Each frame:
#   - Colors active coefficients (non-zero at that alpha) and grays out the rest
#   - Updates the title annotation to show the current lambda and R²
# Because Plotly sliders work by swapping entire frames, we pre-render
# every frame and attach them to the figure via fig.frames.

# Subsample the alpha grid so the slider is responsive (every 2nd point)
step = 2
slider_indices = list(range(0, len(alphas_path), step))

# Create the two-panel subplot layout:
#   - Left panel: Lasso Path (line chart with slider)
#   - Right panel: Coefficient comparison (grouped bar chart, static)
# shared_xaxes=False because the panels have completely different x-axes.
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'Lasso Path: Coefficients vs log(λ)',
        'Coefficient Magnitudes: OLS vs Ridge vs Lasso'
    ],
    column_widths=[0.55, 0.45],  # give the path plot a bit more room
    horizontal_spacing=0.12
)

# --- Static background traces for Panel 1 (all coefficient paths) ---
# We draw every feature's path as a thin gray line first, then overlay
# the "active" highlighted version in the frames.
for i in range(len(feature_names)):
    fig.add_trace(
        go.Scatter(
            x=log_alphas,
            y=coefs_path[i],
            mode='lines',
            line=dict(color='lightgray', width=1),
            name=feature_names[i],
            showlegend=False,
            hoverinfo='name+y'
        ),
        row=1, col=1
    )

# Vertical line at CV-selected lambda* (always visible)
fig.add_vline(
    x=np.log(optimal_alpha), line_dash='dash', line_color='red',
    annotation_text=f'CV λ*={optimal_alpha:.4f}',
    annotation_position='top left',
    row=1, col=1
)

# Horizontal zero line
fig.add_hline(y=0, line_color='black', line_width=0.5, row=1, col=1)

# --- Build slider frames ---
# Each frame contains highlighted traces for features that are non-zero
# at that particular alpha, plus an annotation showing lambda and R².
frames = []
slider_steps = []

for idx in slider_indices:
    a = alphas_path[idx]
    r2_here = r2_at_each_alpha[idx]
    # Identify which features have non-zero coefficients at this alpha
    nonzero_mask = coefs_path[:, idx] != 0
    n_active = int(nonzero_mask.sum())

    # Build highlighted traces -- one colored line per active feature
    frame_traces = []
    for i in range(len(feature_names)):
        if nonzero_mask[i]:
            frame_traces.append(
                go.Scatter(
                    x=log_alphas,
                    y=coefs_path[i],
                    mode='lines',
                    line=dict(width=2.5),
                    name=feature_names[i],
                    showlegend=False,
                    hoverinfo='name+y'
                )
            )

    # Vertical marker line at the current slider lambda
    frame_traces.append(
        go.Scatter(
            x=[np.log(a), np.log(a)],
            y=[coefs_path.min(), coefs_path.max()],
            mode='lines',
            line=dict(color='blue', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    frames.append(go.Frame(
        data=frame_traces,
        name=str(idx),
        # The "traces" field tells Plotly WHICH trace indices to update.
        # We add highlighted traces starting after the static gray lines
        # (indices len(feature_names) onward). This avoids overwriting the
        # static background.
        traces=list(range(len(feature_names), len(feature_names) + len(frame_traces)))
    ))

    # Each slider step activates its corresponding frame by name.
    slider_steps.append(dict(
        args=[[str(idx)], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
        label=f'λ={a:.3f}',
        method='animate'
    ))

# Add placeholder traces that the frames will overwrite.
# We need max possible traces: len(feature_names) highlighted + 1 marker line.
# Frames with fewer active features will simply leave extra placeholders blank.
max_highlighted = len(feature_names) + 1  # +1 for the blue marker line
for _ in range(max_highlighted):
    fig.add_trace(
        go.Scatter(x=[], y=[], mode='lines', showlegend=False, hoverinfo='skip'),
        row=1, col=1
    )

fig.frames = frames

# ------------------------------------------------------------------
# 3D: Build Panel 2 -- Grouped bar chart of coefficient magnitudes
# ------------------------------------------------------------------
# Extract coefficient vectors from each fitted model.
# ols_model.coef_ : OLS coefficients (from sklearn LinearRegression)
# ridge_cv.coef_  : Ridge coefficients (from RidgeCV, fitted with optimal alpha)
# lasso_cv_model.coef_ : Lasso coefficients (from LassoCV, fitted with optimal alpha)

ols_coefs   = np.abs(ols_model.coef_)
ridge_coefs = np.abs(ridge_cv.coef_)
lasso_coefs = np.abs(lasso_cv_model.coef_)

# Sort features by OLS magnitude (descending) so the chart reads top-to-bottom
sort_order = np.argsort(ols_coefs)[::-1]
sorted_names  = [feature_names[i] for i in sort_order]
sorted_ols    = ols_coefs[sort_order]
sorted_ridge  = ridge_coefs[sort_order]
sorted_lasso  = lasso_coefs[sort_order]

# Three grouped bars per feature: OLS (blue), Ridge (orange), Lasso (green)
fig.add_trace(
    go.Bar(
        y=sorted_names, x=sorted_ols,
        name='OLS', orientation='h',
        marker_color='#636EFA',
        # Hover shows exact coefficient value for policy interpretation
        hovertemplate='%{y}: |β| = %{x:.4f}<extra>OLS</extra>'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Bar(
        y=sorted_names, x=sorted_ridge,
        name='Ridge', orientation='h',
        marker_color='#EF553B',
        hovertemplate='%{y}: |β| = %{x:.4f}<extra>Ridge</extra>'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Bar(
        y=sorted_names, x=sorted_lasso,
        name='Lasso', orientation='h',
        marker_color='#00CC96',
        hovertemplate='%{y}: |β| = %{x:.4f}<extra>Lasso</extra>'
    ),
    row=1, col=2
)

# ------------------------------------------------------------------
# 3E: Layout, slider, and annotations
# ------------------------------------------------------------------
# The slider callback mechanism:
#   Plotly sliders work via the "animate" method. Each slider step has an
#   args list: the first element is the frame name to jump to, and the
#   second is a dict controlling animation speed (duration=0 for instant).
#   When the user drags the slider, Plotly swaps in the pre-built frame,
#   replacing only the traces listed in frame.traces (the highlighted
#   lines), while the static gray background traces remain untouched.

fig.update_layout(
    # Slider widget attached to the bottom of Panel 1
    sliders=[dict(
        active=len(slider_indices) // 2,  # start in the middle of the lambda range
        currentvalue=dict(
            prefix='Selected: ',
            visible=True,
            xanchor='center'
        ),
        pad=dict(t=60),
        steps=slider_steps,
        x=0.0, len=0.50,  # slider spans the left panel only
        xanchor='left'
    )],
    # Global figure settings
    height=700,
    width=1400,
    title_text=(
        'Interactive Regularization Dashboard: GDP Growth Predictors<br>'
        '<sup>Left: drag slider to explore Lasso sparsity at different λ  |  '
        'Right: compare coefficient magnitudes across methods</sup>'
    ),
    barmode='group',  # side-by-side bars in Panel 2
    legend=dict(x=0.75, y=1.0, bgcolor='rgba(255,255,255,0.8)'),
    template='plotly_white'
)

# Axis labels for Panel 1
fig.update_xaxes(title_text='log(λ)  ← sparser | denser →', row=1, col=1)
fig.update_yaxes(title_text='Coefficient value (standardized)', row=1, col=1)

# Axis labels for Panel 2; reverse y-axis so largest OLS coef is on top
fig.update_xaxes(title_text='|Coefficient| (standardized)', row=1, col=2)
fig.update_yaxes(autorange='reversed', row=1, col=2)

# Add an annotation with the test R² comparison beneath the bar chart
fig.add_annotation(
    text=(
        f'Test R² -- OLS: {ols_test_r2:.3f}  |  '
        f'Ridge: {ridge_test_r2:.3f}  |  '
        f'Lasso: {lasso_test_r2:.3f}'
    ),
    xref='paper', yref='paper',
    x=0.78, y=-0.08,
    showarrow=False,
    font=dict(size=12, color='black'),
    bgcolor='lightyellow',
    bordercolor='gray',
    borderwidth=1
)

# Render the dashboard -- opens in browser or displays inline in Jupyter
fig.show()

print("\nDashboard rendered")
print("\n" + "=" * 65)
print("HOW TO INTERPRET THIS DASHBOARD")
print("=" * 65)
print("""
PANEL 1 -- Lasso Path with Lambda Slider:
  - Each line is one predictor's coefficient as lambda changes.
  - Drag the slider LEFT (higher lambda) -> more coefficients hit zero (sparser model).
  - Drag RIGHT (lower lambda) -> coefficients grow toward OLS values (denser, riskier).
  - The RED dashed line marks the CV-optimal lambda* -- the best bias-variance tradeoff.
  - OVERFITTING SIGNAL: if you slide lambda well below lambda*, the model includes many
    predictors but test R2 drops -- that is the variance cost of a complex model.
  - OPTIMAL REGION: the zone around the red dashed line, where only a handful
    of predictors are non-zero and test R2 is near its peak.

PANEL 2 -- Coefficient Comparison (OLS vs Ridge vs Lasso):
  - Blue (OLS) bars are often wildly large -- OLS overestimates effects because
    it cannot distinguish signal from noise in high-dimensional settings.
  - Orange (Ridge) bars are uniformly shrunk toward zero -- Ridge keeps every
    predictor but dampens their magnitudes proportionally.
  - Green (Lasso) bars are sparse: many are exactly zero. The surviving green
    bars are the predictors Lasso judges to be the strongest *conditional*
    contributors to GDP growth.
  - POLICY INSIGHT: features where Lasso retains a large green bar AND Ridge
    agrees (large orange bar) are the most robust growth predictors. Features
    where OLS is large but Lasso is zero were likely absorbing collinear noise.
""")
# %%
