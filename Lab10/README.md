# Lab 10: Correlation, Causality, and Spurious Regression

## Objective

This lab investigates a critical threat to econometric inference: why macroeconomic variables produce strong correlations and high-fit regression models that nonetheless harbor little causal truth. We expose the machinery of spurious correlation—trending variables, confounders, reverse causality, and policy reaction mechanisms—and demonstrate diagnostic methods to separate real relationships from statistical illusions.

---

## Dataset

**Real U.S. Monthly Macroeconomic Data from FRED (2010–2024)**

We construct a feature matrix of eight series:

- **CPIAUCSL** (CPI): Consumer price index, a primary inflation measure.
- **UNRATE** (Unemployment Rate): Labor market slack.
- **FEDFUNDS** (Fed Funds Rate): The Federal Reserve's policy rate.
- **INDPRO** (Industrial Production): Real economic activity.
- **RSAFS** (Retail Sales): Nominal aggregate demand.
- **DGS10** (10-Year Treasury Yield): Long-term risk-free rate.
- **PAYEMS** (Total Nonfarm Payrolls): Employment growth.
- **M2SL** (M2 Money Stock): Nominal money supply.

Data is fetched from the St. Louis Federal Reserve Economic Data (FRED) API and covers 180 monthly observations over 15 years of post-financial-crisis recovery, COVID shock, and inflation surge.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core analytics language |
| **pandas** | Data manipulation and alignment |
| **NumPy** | Numerical computation |
| **statsmodels** | OLS regression, VIF diagnostics |
| **seaborn / matplotlib** | Correlation heatmaps and visualizations |
| **networkx** | DAG visualization for causal critique |
| **requests** | FRED API calls |

---

## Methodology: Six-Step Forensic Approach

### Step 1: Data Ingestion (The Observable Macro System)
Pull monthly time-series data directly from FRED via API. Align observations by date, filter to our window, and drop incomplete rows. This gives us a clean 180×8 matrix of real macroeconomic signals.

### Step 2: The Raw Correlation Matrix ("Everything is Correlated")
Construct a correlation heatmap of level variables. We will observe strong positive correlations among CPI, retail sales, payrolls, and M2 (all typically > 0.90). This does *not* prove causation; it reflects shared trends and common demand/supply shocks moving all prices and quantities upward in unison.

### Step 3: The Naive Regression (A Teaching Trap on Purpose)
Estimate an OLS model with CPI as the dependent variable and all seven other macroeconomic series as predictors. We deliberately build a fragile model: high R², seemingly significant coefficients, and false confidence in predictive power. This model will fail in real forecasting and obscure causal structure.

### Step 4: VIF Forensics (Multicollinearity Diagnostic)
Compute Variance Inflation Factors for each predictor. We expect to find VIF > 10 for most variables—a diagnostic red flag signaling that the predictors are nearly collinear. This multicollinearity inflates standard errors, destabilizes coefficients, and renders causal inference unreliable. We then iteratively drop the highest-VIF variable and re-estimate until the model is parsimonious.

### Step 5: Mechanism Check (Transforming the Data)
Convert level variables into year-over-year (YoY) growth rates. This transformation:
- **Removes long-term trends** (which are often non-stationary).
- **Isolates cyclical dynamics** (what economists truly care about).
- **Reduces spurious correlation** caused by both variables drifting upward together.

We reconstruct the correlation matrix using growth rates and observe that spurious relationships weaken or vanish, while genuine short-term relationships may strengthen.

### Step 6: Causal Forensics (DAG Reasoning)
Draw a causal DAG to critique the naive interpretation. A student observes positive correlation between inflation and the fed funds rate and concludes: *"The Fed raises rates, causing inflation."* We present an alternative DAG showing:
- **Reverse causality**: The Fed *reacts to* inflation (policy responds to conditions).
- **Common confounder**: A demand shock drives both inflation and policy rate increases simultaneously.
- **Omitted supply factors**: Oil shocks, wage pressures, or supply-chain disruptions drive inflation; the Fed raises rates *in response*.

---

## The Theoretical Core: Why Inflation and Interest Rates Confuse Us

### The Question: Correlation ≠ Causation in Policy Analysis

A positive correlation between inflation (π) and the federal funds rate (i) appears to support the claim:

> *"Raising interest rates causes inflation."*

This conclusion is backwards. Here is why:

#### 1. **Policy Reaction Function (Reverse Causality)**

The Federal Reserve explicitly targets inflation (and employment). The policy reaction function is:

$$i_t = r^* + \pi_t + \alpha(\pi_t - \pi^*) + \beta(u_t - u^*)$$

where:
- i_t = nominal fed funds rate
- π_t = realized inflation
- π* = inflation target (≈2%)
- u_t = unemployment rate
- u* = natural rate

**The causality runs from inflation to the policy rate, not vice versa.** When inflation rises above target, the Fed *raises* rates to cool demand and bring inflation back down. The observed positive correlation reflects the Fed's *systematic response*, not a causal effect of rates on inflation.

#### 2. **Common Shocks: Inflation-First Dynamics**

Consider a sudden demand shock (e.g., fiscal stimulus, global demand surge):

```
Demand Shock → Inflation rises → Fed tightens (raises rates)
```

Both variables move upward together, but the Fed is *chasing*, not causing. The observable positive correlation masks the true temporal sequence: inflation leads, policy responds.

#### 3. **Supply Shocks: Stagflationary Confusion**

In the presence of supply shocks (oil price spike, supply-chain disruption), inflation and rates can move *in opposite directions* over short windows, yet remain correlated at longer frequencies due to Fed reaction:

- **Immediate**: Oil shock → inflation rises, unemployment rises → Fed's policy stance is ambiguous (raise rates to fight inflation, or cut to support employment?).
- **Medium-term**: As the Fed tightens to control inflation, demand slows → unemployment rises further.

The positive correlation between inflation and rates emerges only after the Fed has had time to respond, not because rates cause inflation.

#### 4. **Observational Confounding in Macro Data**

In randomized experiments, we can isolate causal effects. In macroeconomics, we observe only *realized outcomes*. A high inflation period is precisely when the Fed raises rates—confounding the "effect" of rate increases with the "effect" of having been in an inflationary regime.

To separate causation from correlation, we would need:
- **Exogenous variation** in the Fed's rate path (e.g., an unexpected policy shock).
- **Timing**: Evidence that rate changes *precede* inflation changes by several months.
- **Mechanism checks**: Adding supply proxies (oil prices) to see if the inflation–rate relationship persists.

---

## How to Run This Lab

### Prerequisites
1. **FRED API Key**: Obtain a free API key from [https://fred.stlouisfed.org/docs/api/](https://fred.stlouisfed.org/docs/api/).
2. **Configuration**: Save your key in a `config.py` file in the parent directory:
   ```python
   FRED_API_KEY = "your_api_key_here"
   ```

### Execution
Run the notebook in Jupyter or execute the script:
```bash
python lab10_correlation_causality_spurious_regression.py
```

The script will:
1. Fetch data from FRED.
2. Generate correlation heatmaps.
3. Fit OLS and compute VIF diagnostics.
4. Transform variables and compare results.
5. Visualize a causal DAG.

---

## Key Takeaways

1. **Correlation ≠ Causation**: Strong correlations in observational macro data often reflect shared trends, common shocks, or reverse causality—not direct causal effects.

2. **Multicollinearity Masks Truth**: High VIF values signal that predictors move together. Dropping redundant variables stabilizes inference but requires economic reasoning, not just statistical thresholds.

3. **Transformation Matters**: Converting level variables to growth rates removes spurious trend-driven correlation and isolates cyclical relationships that may reflect true causal mechanisms.

4. **DAGs Are Essential**: Drawing the causal graph forces us to articulate why variables move together. This discipline separates econometric stories from economic truth.

5. **Policy Reaction Is Everywhere**: Central banks, firms, and households respond to macro conditions. This endogeneity is the primary threat to causal inference in macro data. Static correlations cannot distinguish between *policy effects* and *policy responses*.

---

## Extensions and Further Investigation

- **Granger Causality**: Test whether past values of one variable help predict another, controlling for the variable's own history. Does the fed funds rate Granger-cause inflation, or vice versa?
- **VAR Analysis**: Estimate a vector autoregression to recover impulse responses—the effect of an exogenous rate shock on inflation controlling for feedback.
- **Structural Breaks**: Check if relationships shift across crisis periods (2008, 2020). Instability signals structural dependence on regime.
- **Supply vs. Demand Shocks**: Add oil prices and import prices to isolate the inflation source. If inflation is supply-driven, Fed tightening may not reduce it—a critical policy lesson.

---

## Author Notes

This lab demonstrates why descriptive statistics alone are insufficient for economic reasoning. A casual glance at correlation and regression coefficients invites overconfidence. True mastery requires interrogating the data with skepticism, transforming it to isolate mechanisms, and reasoning through causal mechanisms using economic theory and institutional knowledge.

