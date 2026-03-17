#%%
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

#%% [markdown]
# # Step 1: Environment Initialization and Baseline Bivariate Regression
# Import your analytical libraries and load the dataset directly into a pandas DataFrame. Run a naive, simple bivariate regression predicting Sale_Price strictly using Property_Age. Locate the coefficient for Property_Age and observe its sign to determine what it incorrectly implies about aging homes.

#%%
df = pd.read_csv(Path(__file__).parent.parent / 'Data' /'Zillow_California_2026_Hedonic.csv')

# %%
naive_model = smf.ols('Sale_Price ~ Property_Age', data=df).fit()
print(naive_model.summary())
print("\nNaive Age Coefficient:", naive_model.params['Property_Age'])

#%% [markdown]
# # Step 2: The Multivariate Expansion (Controlling for Confounders)
# Run a multiple regression incorporating both features: Property_Age and Distance_to_Tech_Hub. Observe the new coefficient for Property_Age to see how its magnitude shifts or flips signs entirely once the location confounder is absorbed by the hyperplane.

#%%
multi_model = smf.ols('Sale_Price ~ Property_Age + Distance_to_Tech_Hub', data=df).fit()
print(multi_model.summary())
print("\nMultivariate Age Coefficient:", multi_model.params['Property_Age'])

#%% [markdown]
# # Step 3: Manual Execution of the Frisch-Waugh-Lovell Theorem
# Prove how the algorithm isolated that exact coefficient using the FWL three-step method. First, regress the target (Sale_Price) strictly on the confounder (Distance_to_Tech_Hub) and extract the residuals. Second, regress the feature of interest (Property_Age) on the confounder and extract those residuals. Third, run a simple regression predicting the Price Residuals strictly using the Age Residuals, adding - 1 to your formula string to remove the constant intercept.

#%%
# 3a: Partial out distance from Price
res_y_model = smf.ols('Sale_Price ~ Distance_to_Tech_Hub', data=df).fit()
df['Price_Residuals'] = res_y_model.resid
# %%
# 3b: Partial out distance from Age
res_x_model = smf.ols('Property_Age ~ Distance_to_Tech_Hub', data=df).fit()
df['Age_Residuals'] = res_x_model.resid
# %%
# 3c: Regress Residuals on Residuals (-1 removes the intercept for exact mathematical matching)
fwl_model = smf.ols('Price_Residuals ~ Age_Residuals - 1', data=df).fit()
print("\nFWL Isolated Age Coefficient:", fwl_model.params['Age_Residuals'])

#%% [markdown]
# # Step 4: The Epistemological Proof
# Compare the coefficient generated in Step 3 to the coefficient for Property_Age generated in the massive multiple regression in Step 2. They should be identical out to multiple decimal places.

#%%
print("\n" + "="*60)
print("COEFFICIENT COMPARISON: THE FWL PROOF")
print("="*60)
print(f"Multivariate Model (Step 2) - Property_Age Coefficient:  {multi_model.params['Property_Age']:.10f}")
print(f"FWL Method (Step 3) - Age_Residuals Coefficient:         {fwl_model.params['Age_Residuals']:.10f}")
print(f"Difference:                                               {abs(multi_model.params['Property_Age'] - fwl_model.params['Age_Residuals']):.2e}")
print("="*60)

# %%
