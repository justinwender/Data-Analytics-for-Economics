# %%
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import missingno as msno
import category_encoders as ce

# Step 1: Ingestion 
df = pd.read_csv(Path(__file__).parent.parent / 'Data' / 'messy_hr_economics.csv')

# %%
# Step 2: Visual Forensics
msno.matrix(df, figsize=(12, 6), sparkline=False)

# Instructor Note: Students should visually observe that missing 'bonus_pay'
# perfectly aligns with missing 'performance_rating'.
# This structural alignment indicates MAR (Missing at Random).
# %%
# Step 3: Grouped Conditional Imputation
# Imputing the median salary based on department to preserve variance structures
df['base_salary'] = df.groupby('department')['base_salary'].transform(lambda x: x.fillna(x.median()))
# %%
# Step 4: The Dummy Variable Trap (Intentional Failure)
# Remove rows with missing base_salary for modeling
df_model = df.dropna(subset=['base_salary']).reset_index(drop=True)

dummies_trap = pd.get_dummies(df_model['department'], prefix='dept', dtype=int)
X_trap = pd.concat([df_model[['tenure_years']].reset_index(drop=True), dummies_trap.reset_index(drop=True)], axis=1)

# Adding the constant intercept creates perfect multicollinearity
X_trap = sm.add_constant(X_trap)
y = df_model['base_salary'].reset_index(drop=True)

# This will trigger a severe multicollinearity warning or a LinAlgError
model_trap = sm.OLS(y, X_trap).fit()
print(model_trap.summary())
# %%
# Step 5: Escaping the Trap (k-1 methodology)
# drop_first=True establishes the reference category
dummies_safe = pd.get_dummies(df_model['department'], prefix='dept', drop_first=True, dtype=int)
X_safe = pd.concat([df_model[['tenure_years']].reset_index(drop=True), dummies_safe.reset_index(drop=True)], axis=1)
X_safe = sm.add_constant(X_safe)
model_safe = sm.OLS(y, X_safe).fit()
print(model_safe.summary())

# Step 5b: Target Encoding High Cardinality
# Condensing 800 ZIP codes into a single continuous vector representing average salary
encoder = ce.TargetEncoder(cols=['office_zip'])
df['zip_encoded'] = encoder.fit_transform(df['office_zip'], df['base_salary'])
# %%
