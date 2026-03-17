#%% [markdown]
# # Phase 1: The Correlation Trap (Manual Forensics)
# We begin by loading real U.S. macroeconomic data from FRED. You will construct a raw correlation matrix, estimate a naive regression, and then diagnose multicollinearity using VIF.

#%% [markdown]
# ## Step 1: Data Ingestion (The Observable Macro System)
# We pull monthly time-series data directly from FRED. Fill in the missing code in class.

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests
import sys
sys.path.insert(0, '..')
from config import FRED_API_KEY

start = "2010-01-01"
end = "2024-12-31"
# my API key is in a gitignored file so I can use it without pasting/deleting
# to run code, you will need to use an API key
# the url ingestion wasn't working so I had to do it this way
API_KEY = FRED_API_KEY

series = {
    "CPIAUCSL": "cpi",
    "UNRATE": "unrate",
    "FEDFUNDS": "fedfunds",
    "INDPRO": "indpro",
    "RSAFS": "retail_sales",
    "DGS10": "dgs10",
    "PAYEMS": "payrolls",
    "M2SL": "m2"
}

def get_fred_data(series_id, api_key):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    response = requests.get(url, params=params)
    data = response.json()
    if 'observations' in data:
        dates = [obs['date'] for obs in data['observations']]
        values = [float(obs['value']) if obs['value'] != '.' else np.nan for obs in data['observations']]
        return pd.Series(values, index=pd.to_datetime(dates))
    return None

df_list = []
for code, name in series.items():
    s = get_fred_data(code, API_KEY)
    if s is not None:
        s.name = name
        df_list.append(s)

df = pd.concat(df_list, axis=1)
df = df.loc[start:end]  # filter by date range
df = df.dropna()   # remove missing rows

print(df.head())
print(df.shape)

#%% [markdown]
# ## Step 2: The Raw Correlation Matrix ("Everything is Correlated")
# Create a heatmap using raw levels. Your job is to identify which relationships may be statistical illusions caused by trend or common macro dynamics.

#%%
plt.figure(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Raw Correlation Matrix (FRED Macro Data)")
plt.tight_layout()
plt.show()

#%% [markdown]
# ## Concept Check: Spurious Correlation
# You will likely observe strong correlations among CPI, retail sales, payrolls, and M2. This does not automatically imply direct causation. In macroeconomics, variables often move together because of shared trends, inflation, or common demand/supply shocks.

#%% [markdown]
# ## Step 3: The Naive Regression (A Teaching Trap on Purpose)
# Estimate a regression with CPI level as the dependent variable. We are intentionally building a fragile model to learn why it can mislead us.

#%%
y = df['cpi']

X = df[[
    "unrate", "fedfunds", "indpro",
    "retail_sales", "dgs10", "payrolls", "m2"
]]

X = sm.add_constant(X)   # add intercept

model = sm.OLS(y, X,).fit()
print(model.summary())

#%% [markdown]
# ### 🧠 In-Class Discussion
# Why might a high R-squared be misleading in this model? Which variables may be moving together because of trend, not causal structure?

#%% [markdown]
# ## Step 4: VIF Forensics (Multicollinearity Diagnostic)
# Now audit the predictor matrix. Fill in the VIF construction manually.

#%%
X_vif = df[[
    "unrate", "fedfunds", "indpro",
    "retail_sales", "dgs10", "payrolls", "m2"
]].copy()

X_vif = sm.add_constant(X_vif)   # add constant

vif_table = pd.DataFrame()
vif_table["feature"] = X_vif.columns
vif_table["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print(vif_table.sort_values("VIF", ascending=False))

#%% [markdown]
# ### Required Action (in class):
# Drop the highest-VIF variable, recompute VIF, and re-estimate the regression. Repeat until your non-constant predictors are in an acceptable range (we will choose a threshold together).

#%% [markdown]
# ## Step 5: Mechanism Check (Transforming the Data)
# Convert level variables into Year-over-Year growth rates and compare the correlation structure again.

#%%
df_t = df.copy()

for col in ["cpi", "payrolls", "retail_sales", "indpro", "m2"]:
    df_t[f"{col}_yoy"] = df_t[col].pct_change(12)

df_t["inflation_yoy"] = df_t['cpi_yoy']

use_cols = [
    "inflation_yoy", "unrate", "fedfunds", "dgs10",
    "indpro_yoy", "retail_sales_yoy", "payrolls_yoy", "m2_yoy"
]

df_t = df_t[use_cols].dropna()   # select columns + drop NA

plt.figure(figsize=(10, 8))
sns.heatmap(df_t.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix (Transformed Macro Variables)")
plt.tight_layout()
plt.show()

#%% [markdown]
# ## Step 6: Causal Forensics (DAG Reasoning)
# A student sees a positive correlation between inflation and the federal funds rate and concludes:
#
# "Raising interest rates causes inflation."
#
# Action: Draw a DAG and provide at least one alternative causal explanation using:
# - Policy reaction / reverse causality
# - Common shocks (demand shock, supply shock)
# - Omitted variable confounding

#%%
import networkx as nx

G = nx.DiGraph()

# X = policy rate, Y = inflation, Z = confounder
G.add_edges_from([
    ("Z (Demand Shock)", "X (Fed Funds Rate)"),
    ("Z (Demand Shock)", "Y (Inflation)"),
    ("X (Fed Funds Rate)", "Y (Inflation)")  # optional direct causal path
])

pos = {
    "Z (Demand Shock)": (0, 1),
    "X (Fed Funds Rate)": (-1, 0),
    "Y (Inflation)": (1, 0)
}

plt.figure(figsize=(8, 5))
nx.draw(
    G, pos, with_labels=True,
    node_size=3200, font_size=10,
    arrows=True, arrowstyle='-|>', arrowsize=20,
    width=2
)
plt.title("DAG Example: X, Y, and Z Confounder")
plt.axis("off")
plt.show()

# %%
