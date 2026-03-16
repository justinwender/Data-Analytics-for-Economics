#%%
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Step 1: Ingestion from external source
df = pd.read_csv(Path(__file__).parent.parent / 'Data' /'Zillow_ZHVI_2026_Micro.csv')
# %%
# Step 2: Defining the formula
# Utilizing the R-style patsy formula interface allows for elegant, readable model specification
formula ='Home_Value ~ Square_Footage+Property_Age+Distance_to_Transit+School_District_Rating'
# %%
# Step 3: Fitting the model and printing the summary
model = smf.ols(formula = formula, data = df)
results = model.fit()
print(results.summary())
# %%
# Step 4: Generating predictions
# We extract the predicted values vector to transition from explanation to prediction
y_pred = results.predict()
# %%
# Step 5: Calculate RMSE between the actuals and the predictions
model_rmse = rmse(df['Home_Value'], y_pred)
print(f"\nThe Predictive RMSE is: ${model_rmse:,.2f}")
# %%
# Step 6: Residual Forensics Dashboard
# Interactive visualization to detect heteroscedasticity and structural breaks

def create_residual_forensics_dashboard(results, title="Residual Forensics Dashboard"):
    """
    Create an interactive residual forensics scatter plot for OLS model diagnostics.

    Extracts fitted values and residuals from statsmodels regression results,
    identifies outliers, and visualizes them against fitted values to detect
    heteroscedasticity and structural breaks.

    Parameters:
    -----------
    results : statsmodels.regression.linear_model.RegressionResults
        The fitted OLS model object from statsmodels
    title : str
        Title for the dashboard

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive scatter plot with outlier highlighting
    """

    # Extract fitted values from the results object
    # The .fittedvalues attribute contains ŷ_i = X_i * β̂ for each observation
    fitted_values = results.fittedvalues

    # Extract residuals from the results object
    # The .resid attribute contains ε_i = y_i - ŷ_i (actual - predicted)
    residuals = results.resid

    # Calculate standard deviation of residuals for outlier detection
    # Outliers exceed 2σ, roughly identifying the most extreme ~5% (normal approximation)
    residual_std = np.std(residuals)
    outlier_threshold = 2 * residual_std

    # Create boolean mask: True for outliers (|resid| > 2σ), False for normal observations
    is_outlier = np.abs(residuals) > outlier_threshold

    # Prepare data for Plotly visualization
    plot_data = pd.DataFrame({
        'Fitted Values': fitted_values,
        'Residuals': residuals,
        'Is Outlier': is_outlier,
        'Abs Residual': np.abs(residuals)
    })

    # Create scatter plot with conditional coloring
    # X-axis: fitted predicted values | Y-axis: residual errors
    fig = px.scatter(
        plot_data,
        x='Fitted Values',
        y='Residuals',
        color='Is Outlier',
        color_discrete_map={
            True: '#DC143C',    # Crimson (stark red) for outliers
            False: '#4472C4'    # Professional blue for normal points
        },
        labels={
            'Fitted Values': 'Fitted Predicted Values (ŷ)',
            'Residuals': 'Residual Errors (y - ŷ)',
            'Is Outlier': 'Outlier Status'
        },
        hover_data={
            'Is Outlier': False,  # Hide boolean for cleaner hover
            'Abs Residual': ':.4f'
        },
        title=title
    )

    # Add zero-line (perfect prediction baseline)
    # Positive residuals = underestimate; negative = overestimate
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='rgba(128, 128, 128, 0.6)',
        annotation_text='Zero Residual Line',
        annotation_position='right',
        annotation_font_color='gray'
    )

    # Add confidence bands (±2σ) to visualize outlier thresholds
    fig.add_hline(
        y=outlier_threshold,
        line_dash='dot',
        line_color='rgba(220, 20, 60, 0.4)',
        line_width=1
    )
    fig.add_hline(
        y=-outlier_threshold,
        line_dash='dot',
        line_color='rgba(220, 20, 60, 0.4)',
        line_width=1
    )

    # Configure layout
    fig.update_layout(
        hovermode='closest',
        height=600,
        width=1000,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            title='Observation Type',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    # Enhance marker appearance
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        )
    )

    return fig


# Generate the residual forensics dashboard
residual_plot = create_residual_forensics_dashboard(results,
                                                     title="Hedonic Pricing Model: Residual Forensics")
residual_plot.show()
# %%
