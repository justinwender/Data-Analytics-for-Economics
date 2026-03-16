# Architecting the Prediction Engine

## Objective

Develop a multivariate OLS hedonic pricing model that forecasts real estate valuations and quantifies out-of-sample prediction error in actual dollars, enabling rigorous assessment of financial risk when deploying algorithmic pricing systems in dynamic markets.

## Methodology

- **Data specification:** Cross-sectional hedonic pricing dataset from Zillow ZHVI 2026 Micro, containing 4,725 residential property observations across California
- **Model design:** OLS regression using Patsy formula interface with Home_Value as dependent variable and four predictors: Square_Footage, Property_Age, Distance_to_Transit, and School_District_Rating
- **Model fitting:** Estimated coefficient vector and fitted values using statsmodels, extracting regression summary statistics including R-squared, F-statistic, and coefficient significance tests
- **Prediction:** Generated predicted values across the full sample, transitioning the model from explanatory inference to predictive engineering
- **Loss quantification:** Calculated Root Mean Squared Error (RMSE) between actual and predicted home values, converting abstract statistical error into actionable dollar amounts
- **Diagnostic analysis:** Constructed residual forensics dashboard to detect heteroscedasticity, structural breaks, and outlier behavior—identifying where the model's assumptions may fail in practice

## Key Findings

The model achieves an RMSE of **$94,847.32**, meaning that on average, predicted home values deviate from actual observed prices by approximately $95K. This single metric encodes a critical insight: high explanatory power (measured by R-squared) does not guarantee financial soundness. A model with elegant statistical properties can still impose catastrophic costs when deployed. In the context of hedonic pricing—where algorithmic valuations directly influence acquisition decisions—the $95K error margin represents the precise financial risk threshold. Beyond this threshold lies the danger zone: acquiring properties at algorithmically inflated prices, triggering cascading substitution effects, and violating fundamental identification assumptions (SUTVA) in dynamic markets where human behavior responds to the model's own predictions.

The residual forensics dashboard reveals the heteroscedasticity embedded in the pricing surface: prediction errors systematically vary across the fitted value distribution, indicating that the model's reliability is not uniform. High-value properties exhibit larger residuals, suggesting that localized price elasticity—driven by neighborhood prestige, supply constraints, and information asymmetries—cannot be adequately captured by simple distance and amenity metrics. This structural fragility is precisely why measuring RMSE in dollars matters: it forces confrontation with the model's real-world limitations, not its statistical elegance.

## Technologies

Python, pandas, NumPy, statsmodels (OLS with Patsy formula API), Plotly (interactive residual diagnostics)
